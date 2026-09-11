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

NATIVE_PIPELINE_VERSION = "weave-native-memory.v6"


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
    return f"weave-memory-{organization_id.hex}-{repository_id}-code-v1"


async def repository_dataset(binding, repository_id: int, *, legacy: bool = False):
    """Resolve by trusted identity, never a caller-supplied dataset UUID."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        return await session.scalar(
            select(Dataset).where(
                Dataset.name
                == (
                    repository_dataset_name(
                        binding.organization_id, repository_id
                    ).removesuffix("-code-v1")
                    if legacy
                    else repository_dataset_name(binding.organization_id, repository_id)
                ),
                Dataset.owner_id == binding.service_user_id,
                Dataset.tenant_id == binding.tenant_id,
            )
        )


async def forget_repository(
    binding, repository_id: int, *, preserve_reviews: bool = False
) -> None:
    from cognee.modules.weave.memory_sources import forget_source, source_records

    user = await get_user(binding.service_user_id)
    async with scoped_database_context_variables(binding.dataset_id, user.id):
        for record in await source_records(binding, repository_id):
            if preserve_reviews and record.source_key.startswith("review:"):
                continue
            await forget_source(binding, user, record)
    # Remove the dedicated native dataset for this repository, including an
    # interrupted build or a dataset created by the original directory pipeline.
    for legacy in (False, True):
        dataset = await repository_dataset(binding, repository_id, legacy=legacy)
        if dataset is not None:
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


def repository_provenance(request):
    from cognee.tasks.code_graph.models import RepositoryProvenance

    return RepositoryProvenance(
        organization_id=request.organization_id,
        github_repository_id=request.github_repository_id,
        repository_owner=request.repository_owner,
        repository_name=request.repository_name,
        indexed_sha=request.requested_sha,
        pipeline_version=NATIVE_PIPELINE_VERSION,
        extraction_version=request.extraction_version,
    )


async def remember_repository(binding, request, repository):
    import cognee
    from cognee.modules.weave.memory_sources import assert_native_completed
    from cognee.modules.weave.repository_sources import prepare_repository_directory

    # The native code pipeline updates facts/edges and sweeps removed code.
    # Its snapshot identity handles repeats and A -> B -> A without deleting
    # the dataset or the review lessons stored alongside the code.
    user = await get_user(binding.service_user_id)
    dataset = await customer_dataset(binding)
    if dataset is None:
        raise LookupError("Customer dataset is not ready")
    from cognee.modules.weave.memory_sources import source_records, forget_source

    # Retire source receipts from the older per-file loader, preserving reviews.
    async with scoped_database_context_variables(binding.dataset_id, user.id):
        for source in await source_records(binding, request.github_repository_id):
            if not source.source_key.startswith("review:"):
                await forget_source(binding, user, source)
    repository = prepare_repository_directory(repository, request)
    paths = sorted(
        p.relative_to(repository).as_posix()
        for p in repository.rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    import hashlib

    file_hashes = {
        path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
        for path in paths
    }
    token = native_organization.set(binding.organization_id)
    try:
        async with scoped_database_context_variables(dataset.id, user.id):
            from cognee.modules.weave.knowledge_lifecycle import (
                invalidate_changed_knowledge,
                stamp_file_hashes,
            )

            await invalidate_changed_knowledge(
                binding,
                request.github_repository_id,
                request.requested_sha,
                file_hashes,
            )
            result = await cognee.remember(
                str(repository),
                dataset_id=dataset.id,
                user=user,
                content_type="code",
                repository_provenance=repository_provenance(request),
                index_vectors=False,
                self_improvement=False,
                run_in_background=False,
            )
            assert_native_completed(result)
            from cognee.modules.weave.code_files import sync_code_files
            from cognee.modules.weave.review_code_links import sync_review_code_links

            await sync_code_files(binding, repository_provenance(request), paths)
            await stamp_file_hashes(binding, request.github_repository_id, file_hashes)
            await sync_review_code_links(binding, request.github_repository_id)
            return result
    finally:
        native_organization.reset(token)


async def memory_datasets(binding, records):
    """Resolve only this customer's native repository and review datasets."""
    from cognee.modules.weave.memory_sources import source_records

    sources = await source_records(binding)
    # A partially migrated V2 source must never leak stale repository content
    # into the review dataset used by native cross-repository recall.
    if any(not s.source_key.startswith("review:") for s in sources):
        return []
    dataset = await customer_dataset(binding)
    return [dataset] if dataset is not None else []


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
    from cognee.modules.weave.contracts import RecallResponse, RecallDiagnostic
    from cognee.modules.weave.indexing import weave_operation_lock
    from cognee.modules.weave.organizations import get_organization_binding
    from cognee.modules.weave.recall import _repository_reference, _unavailable

    # Freeze revisions during retrieval, but never queue optional review context
    # behind a long-running index or deletion of the customer dataset.
    async with weave_operation_lock(organization_id, wait=False) as acquired:
        if not acquired:
            return _unavailable(
                organization_id, request, "unavailable", "backend_unavailable"
            )
        binding = await get_organization_binding(organization_id)
        if binding is None:
            return _unavailable(
                organization_id, request, "unavailable", "organization_not_provisioned"
            )
        # Native recall searches owned customer datasets, so disclose and validate
        # every repository receipt, including incomplete first-time indexes.
        records = await customer_snapshots(organization_id)
        if not snapshots_ready(records):
            return _unavailable(
                organization_id, request, "unavailable", "no_indexed_repository"
            )
        selected = {record.github_repository_id for record in records}
        required = set(request.github_repository_ids)
        if request.primary_github_repository_id is not None:
            required.add(request.primary_github_repository_id)
        if not required.issubset(selected):
            return _unavailable(
                organization_id, request, "unavailable", "no_indexed_repository"
            )
        datasets = await memory_datasets(binding, records)
        if not datasets:
            return _unavailable(
                organization_id, request, "unavailable", "no_indexed_repository"
            )
        user = await get_user(binding.service_user_id)
        code_datasets = datasets
        # Code navigation works immediately and never invokes an LLM or embeddings.
        code_query = {"operation": "query_facts", "limit": request.top_k}
        if request.mode == "impact_context":
            code_query = {
                "operation": "impact_analysis",
                "names": request.seeds or [request.query],
                "max_depth": request.depth,
                "max_nodes": request.top_k,
            }
        elif request.seeds:
            code_query["names"] = request.seeds
        elif request.mode != "repository_context":
            code_query["name"] = request.query
        async with scoped_database_context_variables(code_datasets[0].id, user.id):
            result = await cognee.recall(
                request.query,
                scope="code",
                code_query=code_query,
                dataset_ids=[dataset.id for dataset in code_datasets],
                user=user,
                top_k=request.top_k,
            )
        # Partial review learning must not hide usable code or expose partial memory.
        from cognee.modules.weave.memory_sources import source_records

        sources = await source_records(binding)
        from cognee.modules.weave.review_code_links import linked_review_context

        linked = await linked_review_context(binding, result, sources, request.top_k)
        if linked:
            result += [{"qualified_review_context": linked}]
        from cognee.modules.weave.memory_retrieval import (
            eligible_memory_sources,
            recall_current_memory,
        )

        semantic = eligible_memory_sources(binding, sources, required)
        diagnostics = []
        if semantic:
            try:
                embedding, llm = get_weave_embedding_config(), get_weave_llm_config()
                async with scoped_database_context_variables(
                    binding.dataset_id,
                    user.id,
                    llm_config=llm,
                    embedding_config=embedding,
                ):
                    result += await recall_current_memory(
                        binding,
                        user,
                        request.query,
                        semantic,
                        repository_ids=required,
                        top_k=request.top_k,
                        llm=llm,
                        embedding=embedding,
                    )
            except Exception:
                # Advisory paid memory must not discard successful code navigation.
                diagnostics.append(RecallDiagnostic(code="backend_unavailable"))
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
            diagnostics=diagnostics,
        )
