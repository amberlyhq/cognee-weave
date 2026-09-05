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

NATIVE_PIPELINE_VERSION = "weave-native-memory.v2"


async def customer_dataset(binding):
    async with get_relational_engine().get_async_session() as session:
        return await session.scalar(
            select(Dataset).where(
                Dataset.id == binding.dataset_id,
                Dataset.owner_id == binding.service_user_id,
                Dataset.tenant_id == binding.tenant_id,
            )
        )


async def customer_snapshots(organization_id):
    from cognee.modules.weave.models import WeaveRepositorySnapshot
    from cognee.modules.weave.organizations import set_weave_organization_scope

    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        return list(
            await session.scalars(
                select(WeaveRepositorySnapshot)
                .where(
                    WeaveRepositorySnapshot.organization_id == organization_id,
                    WeaveRepositorySnapshot.deleted_at.is_(None),
                )
                .order_by(WeaveRepositorySnapshot.github_repository_id)
            )
        )


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
    from cognee.modules.weave.memory_sources import forget_source, source_records

    user = await get_user(binding.service_user_id)
    async with scoped_database_context_variables(binding.dataset_id, user.id):
        for record in await source_records(binding, repository_id):
            await forget_source(binding, user, record)
    # Legacy datasets are retained during indexing migration. An explicit
    # repository removal must still remove that repository's legacy memory.
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
    from cognee.modules.weave.memory_sources import forget_source, source_records

    async with scoped_database_context_variables(binding.dataset_id, user.id):
        for record in await source_records(binding):
            await forget_source(binding, user, record)
    for dataset in datasets:
        await _forget_dataset(binding, dataset, user)


async def remember_repository(binding, request, repository):
    from cognee.modules.weave.memory_sources import forget_source, source_records, sync_source
    from cognee.modules.weave.repository_sources import prepare_repository_sources

    # Caller holds the operation lock and has marked the snapshot running.
    # Replace individual native sources, never the customer dataset.
    embedding = get_weave_embedding_config()
    llm = get_weave_llm_config()
    user = await get_user(binding.service_user_id)
    dataset = await customer_dataset(binding)
    if dataset is None:
        raise LookupError("Customer dataset not found")
    async with scoped_database_context_variables(
        dataset.id, user.id, llm_config=llm, embedding_config=embedding
    ):
        sources = await prepare_repository_sources(
            repository, request, user=user, dataset_id=dataset.id
        )
        wanted = {key for key, _, _ in sources}
        for record in await source_records(binding, request.github_repository_id):
            if not record.source_key.startswith("review:") and record.source_key not in wanted:
                await forget_source(binding, user, record)
        actions = []
        for key, item, fingerprint in sources:
            actions.append(
                await sync_source(
                    binding,
                    user,
                    request.github_repository_id,
                    key,
                    item,
                    fingerprint,
                    llm=llm,
                    embedding=embedding,
                )
            )
        return actions


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
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.organizations import get_organization_binding
    from cognee.modules.weave.recall import _repository_reference, _unavailable

    # Freeze selected revisions during retrieval. Repository writes take the
    # corresponding shared organization lock, so cannot race this read.
    async with weave_operation_lock(organization_id):
        binding = await get_organization_binding(organization_id)
        if binding is None:
            return _unavailable(
                organization_id, request, "unavailable", "organization_not_provisioned"
            )
        # Native recall searches the customer dataset, so disclose and validate
        # every repository receipt, including incomplete first-time indexes.
        records = await customer_snapshots(organization_id)
        sources = await source_records(binding)
        if not snapshots_ready(records) or any(s.status != "completed" for s in sources):
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        selected = {record.github_repository_id for record in records}
        required = set(request.github_repository_ids)
        if request.primary_github_repository_id is not None:
            required.add(request.primary_github_repository_id)
        if not required.issubset(selected):
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        dataset = await customer_dataset(binding)
        if dataset is None:
            return _unavailable(organization_id, request, "unavailable", "no_indexed_repository")
        user = await get_user(binding.service_user_id)
        embedding = get_weave_embedding_config()
        llm = get_weave_llm_config()
        async with scoped_database_context_variables(
            dataset.id, user.id, llm_config=llm, embedding_config=embedding
        ):
            result = await cognee.recall(
                request.query,
                dataset_ids=[dataset.id],
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
