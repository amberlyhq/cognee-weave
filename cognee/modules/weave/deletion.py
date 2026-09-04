from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from sqlalchemy import select, text, update

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.code_retriever import (
    CODE_NODE_TYPES,
    invalidate_code_graph_snapshot_cache,
)
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.weave.contracts import DeleteResponse, SurfaceEdge, SurfaceResponse
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.models import WeaveOrganizationBinding, WeaveRepositorySnapshot
from cognee.modules.weave.organizations import (
    OrganizationBinding,
    get_organization_binding,
    set_weave_organization_scope,
)
from cognee.modules.weave.recall import (
    _candidate,
    _matches_snapshot,
    _properties,
    _repository_reference,
)


class SurfaceNotFound(LookupError):
    """An absent and a foreign target deliberately share this error."""

    def __init__(self):
        super().__init__("Resource not found")


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _surface_snapshots(
    organization_id: UUID,
    repository_ids: list[int],
) -> list[WeaveRepositorySnapshot]:
    if len(repository_ids) > 20 or any(item <= 0 for item in repository_ids):
        raise SurfaceNotFound()
    requested = set(repository_ids)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        query = select(WeaveRepositorySnapshot).where(
            WeaveRepositorySnapshot.organization_id == organization_id,
            WeaveRepositorySnapshot.deleted_at.is_(None),
            WeaveRepositorySnapshot.indexed_sha.is_not(None),
        )
        if requested:
            query = query.where(WeaveRepositorySnapshot.github_repository_id.in_(requested))
        records = list(
            await session.scalars(query.order_by(WeaveRepositorySnapshot.github_repository_id))
        )
    if not records or (requested and {item.github_repository_id for item in records} != requested):
        raise SurfaceNotFound()
    return records


async def _read_surface(
    organization_id: UUID,
    repository_ids: list[int],
    surface: Literal["export", "visualization"],
) -> SurfaceResponse:
    binding = await get_organization_binding(organization_id)
    if binding is None:
        raise SurfaceNotFound()
    records = await _surface_snapshots(organization_id, repository_ids)
    snapshots = {record.github_repository_id: record for record in records}

    async with scoped_database_context_variables(
        binding.dataset_id,
        binding.service_user_id,
    ):
        graph = await get_graph_engine()
        raw_nodes, raw_edges = await graph.get_filtered_graph_data(
            [{"type": [*CODE_NODE_TYPES, "CodeRepository"]}],
            max_nodes=500,
            max_edges=1000,
        )

    nodes = []
    identities: dict[str, str] = {}
    for node_id, raw_properties in raw_nodes or []:
        properties = _properties(raw_properties)
        snapshot = _matches_snapshot(properties, snapshots)
        if snapshot is None:
            continue
        candidate = _candidate(properties, snapshot, fallback_identity=str(node_id))
        if candidate is None:
            continue
        nodes.append(candidate)
        identities[str(node_id)] = candidate.fact_identity
        if len(nodes) >= 500:
            break

    edges = []
    for edge in raw_edges or []:
        if not isinstance(edge, (tuple, list)) or len(edge) < 3:
            continue
        source_id, target_id, relation_type = map(str, edge[:3])
        if source_id not in identities or target_id not in identities:
            continue
        edges.append(
            SurfaceEdge(
                source_fact_identity=identities[source_id],
                target_fact_identity=identities[target_id],
                relation_type=relation_type,
            )
        )
        if len(edges) >= 1000:
            break

    return SurfaceResponse(
        organization_id=organization_id,
        surface=surface,
        repositories=[_repository_reference(record) for record in records],
        nodes=nodes,
        edges=edges,
    )


async def export_organization(
    organization_id: UUID,
    repository_ids: list[int] | None = None,
) -> SurfaceResponse:
    return await _read_surface(organization_id, repository_ids or [], "export")


async def visualize_organization(
    organization_id: UUID,
    repository_ids: list[int] | None = None,
) -> SurfaceResponse:
    return await _read_surface(organization_id, repository_ids or [], "visualization")


async def _mark_repository_deleted(
    organization_id: UUID,
    github_repository_id: int,
) -> None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        snapshot = await session.scalar(
            select(WeaveRepositorySnapshot)
            .where(
                WeaveRepositorySnapshot.organization_id == organization_id,
                WeaveRepositorySnapshot.github_repository_id == github_repository_id,
            )
            .with_for_update()
        )
        if snapshot is None:
            raise SurfaceNotFound()
        snapshot.deleted_at = _now()
        snapshot.status = "deleted"
        await session.commit()


async def activate_repository(
    organization_id: UUID,
    github_repository_id: int,
) -> DeleteResponse:
    """Reactivate a tombstone only through the verified repository lifecycle API."""

    async with weave_operation_lock(organization_id, github_repository_id):
        binding = await get_organization_binding(organization_id)
        if binding is None or github_repository_id <= 0:
            raise SurfaceNotFound()
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            snapshot = await session.scalar(
                select(WeaveRepositorySnapshot)
                .where(
                    WeaveRepositorySnapshot.organization_id == organization_id,
                    WeaveRepositorySnapshot.github_repository_id == github_repository_id,
                )
                .with_for_update()
            )
            if snapshot is not None and snapshot.deleted_at is not None:
                snapshot.deleted_at = None
                snapshot.requested_sha = None
                snapshot.indexed_sha = None
                snapshot.pipeline_version = None
                snapshot.extraction_version = None
                snapshot.status = "not_indexed"
                snapshot.error_code = None
                await session.commit()
    return DeleteResponse(
        organization_id=organization_id,
        github_repository_id=github_repository_id,
    )


async def delete_repository(
    organization_id: UUID,
    github_repository_id: int,
) -> DeleteResponse:
    async with weave_operation_lock(organization_id, github_repository_id):
        binding = await get_organization_binding(organization_id)
        if binding is None or github_repository_id <= 0:
            raise SurfaceNotFound()

        async with scoped_database_context_variables(
            binding.dataset_id,
            binding.service_user_id,
        ):
            graph = await get_graph_engine()
            vector = await get_vector_engine_async()
            raw_nodes, _raw_edges = await graph.get_filtered_graph_data(
                [{"type": [*CODE_NODE_TYPES, "CodeRepository"]}],
                max_edges=0,
            )
            by_collection: dict[str, list[UUID]] = {}
            node_ids = []
            for node_id, raw_properties in raw_nodes or []:
                properties = _properties(raw_properties)
                if str(properties.get("organization_id")) != str(organization_id):
                    continue
                try:
                    repository_id = int(properties.get("github_repository_id"))
                except (TypeError, ValueError):
                    continue
                if repository_id != github_repository_id:
                    continue
                node_ids.append(str(node_id))
                node_type = str(properties.get("type") or "")
                if node_type:
                    by_collection.setdefault(f"{node_type}_name", []).append(UUID(str(node_id)))

            for collection_name, point_ids in by_collection.items():
                try:
                    await vector.delete_data_points(collection_name, point_ids)
                except CollectionNotFoundError:
                    pass
            await graph.delete_nodes(node_ids)
            invalidate_code_graph_snapshot_cache(dataset_id=binding.dataset_id)

        # Publish the tombstone only after every physical cleanup succeeds. A
        # failed cleanup remains active and the same DELETE can safely retry.
        await _mark_repository_deleted(organization_id, github_repository_id)

    return DeleteResponse(
        organization_id=organization_id,
        github_repository_id=github_repository_id,
    )


async def _load_organization_for_delete(
    organization_id: UUID,
) -> tuple[OrganizationBinding, DatasetDatabase]:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        record = await session.scalar(
            select(WeaveOrganizationBinding)
            .where(
                WeaveOrganizationBinding.organization_id == organization_id,
                WeaveOrganizationBinding.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if record is None:
            raise SurfaceNotFound()
        database = await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == record.primary_dataset_id)
        )
        if database is None:
            raise SurfaceNotFound()
        binding = OrganizationBinding(
            organization_id=record.organization_id,
            tenant_id=record.tenant_id,
            service_user_id=record.service_user_id,
            dataset_id=record.primary_dataset_id,
            graph_schema=database.graph_database_connection_info["graph_database_schema"],
            vector_schema=database.vector_database_connection_info["schema"],
        )
        return binding, database


async def _mark_organization_deleted(organization_id: UUID) -> None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        record = await session.scalar(
            select(WeaveOrganizationBinding)
            .where(
                WeaveOrganizationBinding.organization_id == organization_id,
                WeaveOrganizationBinding.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if record is None:
            raise SurfaceNotFound()
        deleted_at = _now()
        record.deleted_at = deleted_at
        await session.execute(
            update(WeaveRepositorySnapshot)
            .where(
                WeaveRepositorySnapshot.organization_id == organization_id,
                WeaveRepositorySnapshot.deleted_at.is_(None),
            )
            .values(deleted_at=deleted_at, status="deleted")
        )
        await session.commit()


async def _drop_bound_organization_schema(organization_id: UUID) -> None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        await session.execute(
            text(
                "SELECT public.weave_drop_organization_dataset_schema(:organization_id)"
            ),
            {"organization_id": organization_id},
        )
        await session.commit()


async def delete_organization(organization_id: UUID) -> DeleteResponse:
    async with weave_operation_lock(organization_id):
        binding, database = await _load_organization_for_delete(organization_id)
        async with scoped_database_context_variables(
            binding.dataset_id,
            binding.service_user_id,
        ):
            invalidate_code_graph_snapshot_cache(dataset_id=binding.dataset_id)

        from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache
        from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache
        from cognee.infrastructure.databases.vector.pgvector.PGVectorSharedDatasetDatabaseHandler import (
            PGVectorSharedDatasetDatabaseHandler,
        )

        # Graph and vector share one schema. Evict both adapter caches first,
        # then issue one idempotent DROP SCHEMA CASCADE.
        graph_engine_cache.evict_matching(graph_database_schema=binding.graph_schema)
        vector_engine_cache.evict_matching(vector_db_schema=binding.vector_schema)
        if os.getenv("WEAVE_STRICT_MODE") == "true":
            await _drop_bound_organization_schema(organization_id)
        else:
            await PGVectorSharedDatasetDatabaseHandler.delete_dataset(database)
        await _mark_organization_deleted(organization_id)
        return DeleteResponse(organization_id=organization_id)
