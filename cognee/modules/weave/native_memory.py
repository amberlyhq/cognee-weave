"""Tenant/repository routing around Cognee's unchanged public memory operations.

No extraction tasks, prompts, graph models, or ranking algorithms belong here.
"""

import json
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_user
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.scope import native_organization

NATIVE_PIPELINE_VERSION = "weave-native-memory.v1"


def repository_dataset_name(organization_id: UUID, repository_id: int) -> str:
    if repository_id <= 0:
        raise ValueError("Repository ID must be positive")
    return f"weave-memory-{organization_id.hex}-{repository_id}"


async def repository_dataset(binding, repository_id: int):
    """Resolve by trusted identity, never a caller-supplied dataset UUID."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        return await session.scalar(
            select(Dataset).where(
                Dataset.name == repository_dataset_name(binding.organization_id, repository_id),
                Dataset.owner_id == binding.service_user_id,
                Dataset.tenant_id == binding.tenant_id,
            )
        )


async def forget_repository(binding, repository_id: int) -> None:
    dataset = await repository_dataset(binding, repository_id)
    if dataset is None:
        return
    user = await get_user(binding.service_user_id)
    await _forget_dataset(binding, dataset, user)


async def _forget_dataset(binding, dataset, user):
    import cognee

    token = native_organization.set(binding.organization_id)
    try:
        async with scoped_database_context_variables(dataset.id, user.id):
            await cognee.forget(dataset_id=dataset.id, user=user)
    finally:
        native_organization.reset(token)


async def forget_organization_memory(binding) -> None:
    """Include interrupted builds, not just successfully published snapshots."""
    engine = get_relational_engine()
    prefix = f"weave-memory-{binding.organization_id.hex}-"
    async with engine.get_async_session() as session:
        datasets = list(
            await session.scalars(
                select(Dataset).where(
                    Dataset.owner_id == binding.service_user_id,
                    Dataset.tenant_id == binding.tenant_id,
                    Dataset.name.startswith(prefix),
                )
            )
        )
    user = await get_user(binding.service_user_id)
    for dataset in datasets:
        await _forget_dataset(binding, dataset, user)


async def remember_repository(binding, request, repository):
    import cognee
    from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset

    # Caller holds the operation lock and has marked the snapshot running.
    # Replacement uses native deletion; failed rebuilds are not served as current.
    embedding = get_weave_embedding_config()
    llm = get_weave_llm_config()
    await forget_repository(binding, request.github_repository_id)
    user = await get_user(binding.service_user_id)
    dataset = await create_authorized_dataset(
        repository_dataset_name(binding.organization_id, request.github_repository_id), user
    )
    async with scoped_database_context_variables(
        dataset.id, user.id, llm_config=llm, embedding_config=embedding
    ):
        result = await cognee.remember(
            str(repository),
            dataset_id=dataset.id,
            user=user,
            llm_config=llm,
            embedding_config=embedding,
        )
        if result.status != "completed":
            raise RuntimeError("Native remember did not complete")
        return result


def snapshots_ready(records) -> bool:
    return bool(records) and all(
        record.pipeline_version == NATIVE_PIPELINE_VERSION
        and record.status == "indexed"
        and record.indexed_sha is not None
        and record.indexed_sha == record.requested_sha
        for record in records
    )


async def recall_repository_memory(organization_id, request):
    import cognee
    from cognee.modules.weave.contracts import RecallResponse
    from cognee.modules.weave.indexing import weave_operation_lock
    from cognee.modules.weave.organizations import get_organization_binding
    from cognee.modules.weave.recall import _load_snapshots, _repository_reference, _unavailable

    # Freeze selected revisions during retrieval. Repository writes take the
    # corresponding shared organization lock, so cannot race this read.
    async with weave_operation_lock(organization_id):
        binding = await get_organization_binding(organization_id)
        if binding is None:
            return _unavailable(
                organization_id, request, "unavailable", "organization_not_provisioned"
            )
        records = await _load_snapshots(
            organization_id, request.github_repository_ids, request.primary_github_repository_id
        )
        if not snapshots_ready(records):
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        selected = {record.github_repository_id for record in records}
        required = set(request.github_repository_ids)
        if request.primary_github_repository_id is not None:
            required.add(request.primary_github_repository_id)
        if not required.issubset(selected):
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        datasets = [
            await repository_dataset(binding, record.github_repository_id) for record in records
        ]
        if any(dataset is None for dataset in datasets):
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        user = await get_user(binding.service_user_id)
        embedding = get_weave_embedding_config()
        llm = get_weave_llm_config()
        async with scoped_database_context_variables(
            datasets[0].id, user.id, llm_config=llm, embedding_config=embedding
        ):
            result = await cognee.recall(
                request.query,
                dataset_ids=[dataset.id for dataset in datasets],
                user=user,
                top_k=request.top_k,
                llm_config=llm,
                embedding_config=embedding,
            )
        # Preserve native output: these are NOT source-verified symbol candidates.
        memory = json.dumps(
            [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in result
            ],
            ensure_ascii=False,
        )
        if len(memory) > 64000:
            raise ValueError("Native recall exceeds the response size boundary")
        references = [_repository_reference(record) for record in records]
        shas = {record.indexed_sha for record in records}
        return RecallResponse(
            status="available",
            organization_id=organization_id,
            mode=request.mode,
            indexed_default_sha=next(iter(shas)) if len(shas) == 1 else None,
            repositories=references,
            native_memory=memory,
        )
