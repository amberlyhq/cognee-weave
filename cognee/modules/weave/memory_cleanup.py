"""Tenant-scoped native entity consolidation after an accepted memory job."""

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.weave.agent_memory import repository_scope
from cognee.modules.weave.agent_memory_contracts import MemoryCleanupResponse, NativeCleanupRun
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.memory_sources import assert_native_completed
from cognee.modules.weave.models import WeaveMemoryCleanup, WeaveMemoryJob
from cognee.modules.weave.organizations import set_weave_organization_scope
from cognee.modules.weave.scope import native_organization


async def cleanup_memory_job(organization_id, repository_id, job_id, request):
    """Replay completed receipts; a failed attempt retries only native cleanup."""
    from cognee.memify_pipelines.consolidate_entities import consolidate_entities_pipeline

    async with weave_operation_lock(organization_id, wait=False) as acquired:
        if not acquired:
            raise LookupError("Customer source indexing is busy")
        binding, _ = await repository_scope(
            organization_id, repository_id, request.lifecycle_generation
        )
        identity = (organization_id, job_id, request.mode)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.get(WeaveMemoryJob, (organization_id, job_id))
            if job is None or job.github_repository_id != repository_id:
                raise LookupError("Memory job not found")
            if job.status != "completed":
                raise LookupError("Memory job is not completed")
            if job.payload.get("lifecycle_generation") != request.lifecycle_generation:
                raise LookupError("Memory job lifecycle changed")
            cleanup = await session.get(WeaveMemoryCleanup, identity)
            if cleanup is not None and cleanup.status == "completed":
                return MemoryCleanupResponse.model_validate(cleanup.receipt)
            if cleanup is None:
                cleanup = WeaveMemoryCleanup(
                    organization_id=organization_id,
                    github_repository_id=repository_id,
                    job_id=job_id,
                    lifecycle_generation=request.lifecycle_generation,
                    mode=request.mode,
                    status="processing",
                )
                session.add(cleanup)
                await session.commit()
        user = await get_user(binding.service_user_id)
        token = native_organization.set(organization_id)
        try:
            async with scoped_database_context_variables(
                binding.dataset_id,
                user.id,
                llm_config=get_weave_llm_config(),
                embedding_config=get_weave_embedding_config(),
            ):
                result = await consolidate_entities_pipeline(
                    dataset=binding.dataset_id,
                    user=user,
                    dry_run=request.mode == "preview",
                    run_in_background=False,
                )
        finally:
            native_organization.reset(token)
        assert_native_completed(result)
        runs = [
            NativeCleanupRun(
                pipeline_run_id=run.pipeline_run_id,
                dataset_id=run.dataset_id,
                status=run.status,
            )
            for run in result.values()
        ]
        if any(run.dataset_id != binding.dataset_id for run in runs):
            raise RuntimeError("Native cleanup returned a foreign dataset")
        response = MemoryCleanupResponse(
            organization_id=organization_id,
            github_repository_id=repository_id,
            job_id=job_id,
            mode=request.mode,
            native_runs=runs,
        )
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            cleanup = await session.get(WeaveMemoryCleanup, identity)
            cleanup.status = "completed"
            cleanup.receipt = response.model_dump(mode="json")
            await session.commit()
        return response
