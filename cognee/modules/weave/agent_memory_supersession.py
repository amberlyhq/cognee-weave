"""Host-requested cancellation restores committed native state, without re-qualifying."""

import hashlib

from sqlalchemy import select

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_user
from cognee.modules.weave.agent_memory import repository_scope
from cognee.modules.weave.agent_memory_contracts import MemorySupersedeResponse
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.legacy_memory_notes import note_source_key
from cognee.modules.weave.memory_retrieval import native_source_tag
from cognee.modules.weave.memory_sources import forget_source, source_data_id, sync_source
from cognee.modules.weave.models import (
    WeaveMemoryJob,
    WeaveMemoryNote,
    WeaveMemoryOperation,
    WeaveMemorySource,
)
from cognee.modules.weave.note_code_links import sync_note_code_links
from cognee.modules.weave.organizations import set_weave_organization_scope
from cognee.modules.weave.scope import native_organization
from cognee.tasks.ingestion.data_item import DataItem


async def supersede_memory_job(organization_id, repository_id, job_id, generation):
    async with weave_operation_lock(organization_id, wait=False) as acquired:
        if not acquired:
            raise LookupError("Customer source indexing is busy")
        binding, _ = await repository_scope(organization_id, repository_id, generation)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.get(WeaveMemoryJob, (organization_id, job_id))
            if job is not None and job.github_repository_id != repository_id:
                raise LookupError("Memory job not found")
            if job is None:
                # A cancellation can beat the first delivery; remember its tombstone.
                job = WeaveMemoryJob(
                    organization_id=organization_id,
                    github_repository_id=repository_id,
                    job_id=job_id,
                    payload={},
                    payload_hash="",
                    status="superseded",
                )
                session.add(job)
                await session.commit()
            if job.status in {"completed", "superseded"}:
                return MemorySupersedeResponse(
                    organization_id=organization_id,
                    github_repository_id=repository_id,
                    job_id=job_id,
                    status=job.status,
                )
            job.status = "superseding"
            pending = list(
                await session.scalars(
                    select(WeaveMemoryOperation).where(
                        WeaveMemoryOperation.organization_id == organization_id,
                        WeaveMemoryOperation.job_id == job_id,
                        WeaveMemoryOperation.status.in_(["processing", "superseding"]),
                    )
                )
            )
            for operation in pending:
                operation.status = "superseding"
            await session.commit()
        user = await get_user(binding.service_user_id)
        llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
        token = native_organization.set(organization_id)
        try:
            async with scoped_database_context_variables(
                binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
            ):
                for operation in pending:
                    await restore_committed_note(
                        binding, user, operation, llm=llm, embedding=embedding
                    )
        finally:
            native_organization.reset(token)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.get(WeaveMemoryJob, (organization_id, job_id))
            job.status = "superseded"
            await session.commit()
        return MemorySupersedeResponse(
            organization_id=organization_id,
            github_repository_id=repository_id,
            job_id=job_id,
            status="superseded",
        )


async def restore_committed_note(binding, user, operation, *, llm, embedding):
    org, repo = binding.organization_id, operation.github_repository_id
    source_key = (operation.receipt or {}).get("source_key") or await note_source_key(
        binding, repo, operation.note_id
    )
    source_identity = (org, repo, source_key)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, org)
        note = await session.get(WeaveMemoryNote, (org, repo, operation.note_id))
        source = await session.get(WeaveMemorySource, source_identity)
        data = await session.get(Data, source_data_id(*source_identity))
        if data is not None and (data.dataset_id, data.owner_id, data.tenant_id) != (
            binding.dataset_id,
            user.id,
            binding.tenant_id,
        ):
            raise ValueError("Foreign native memory source")
    if note is None or note.status == "retired":
        if source is not None:
            await forget_source(binding, user, source)
    else:
        digest = hashlib.sha256(note.content.encode()).hexdigest()
        tag = native_source_tag(source_data_id(*source_identity), digest)
        metadata = {
            "memory_kind": "agent_note",
            "note_id": str(note.note_id),
            "github_repository_id": repo,
            "source_sha": note.source_sha,
            "source_paths": note.source_paths,
            "provenance": note.provenance,
        }
        ready = (
            source is not None
            and data is not None
            and source.content_hash == digest
            and source.status in {"completed", "processing_note"}
        )
        if not ready:
            revision = (
                max(note.version, (source.artifact_revision or 0) + (source.content_hash != digest))
                if source
                else note.version
            )
            await sync_source(
                binding,
                user,
                repo,
                source_key,
                DataItem(
                    data=note.content,
                    label=f"memory-note-{note.note_id}.txt",
                    external_metadata=metadata,
                ),
                digest,
                llm=llm,
                embedding=embedding,
                artifact_revision=revision,
                self_improvement=False,
                node_set=[tag],
            )
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, org)
            source = await session.get(WeaveMemorySource, source_identity)
            source.status = "processing_note"
            source.dataset_id = binding.dataset_id
            previous = operation.receipt or {}
            if previous.get("previous_source_hash") == digest:
                # Restore the original source eligibility too. In particular an
                # unqualified legacy source must not become current by rollback.
                source.qualification = previous.get("previous_source_qualification")
            else:
                source.qualification = {"memory_note": True, "native_source_tag": tag}
            await session.commit()
        await sync_note_code_links(binding, repo, source_key, note.source_paths)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, org)
        if note is not None and note.status == "active":
            source = await session.get(WeaveMemorySource, source_identity)
            source.status = "completed"
            data = await session.get(Data, source.data_id)
            data.external_metadata = metadata
        current = await session.get(WeaveMemoryOperation, (org, operation.operation_id))
        current.status = "superseded"
        await session.commit()
