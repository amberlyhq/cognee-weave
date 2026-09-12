"""Move native source ownership alongside entity consolidation, without changing matching."""

from uuid import NAMESPACE_OID, UUID, uuid5

from sqlalchemy import or_, select

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import (
    get_dataset_id_from_source_ref_key,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.graph.models import Edge, Node


async def transfer_consolidated_ownership(graph_engine, remap, edges):
    """Attach every original owner before duplicate graph nodes are removed.

    Relational ownership changes commit together. Graph-backed ownership uses the
    adapter's idempotent attach operations; an interrupted transfer can be repeated
    while the original nodes still exist. No new pipeline owns these old facts.
    """
    dataset_id = current_dataset_id.get()
    if dataset_id is None:
        raise ValueError("Entity consolidation requires a dataset ownership scope")
    if await stores_provenance_in_graph(graph_engine):
        await _transfer_graph_ownership(graph_engine, dataset_id, remap, edges)
    else:
        await _transfer_ledger_ownership(dataset_id, remap)


async def _transfer_graph_ownership(graph, dataset_id, remap, edges):
    async def attach(method, target, record):
        if record is None:
            return
        keys = record.source_ref_keys
        if any(get_dataset_id_from_source_ref_key(key) != dataset_id for key in keys):
            raise ValueError("Consolidation ownership belongs to another dataset")
        await method([target], keys, source_run_refs=record.source_run_refs)

    nodes = await graph.get_node_delete_data(list(remap))
    for duplicate, canonical in remap.items():
        await attach(graph.attach_node_source_refs, canonical, nodes.get(duplicate))
    moved = {}
    for source, target, relationship, *_ in edges:
        source, target = str(source), str(target)
        new_source, new_target = remap.get(source, source), remap.get(target, target)
        if (new_source, new_target) != (source, target) and new_source != new_target:
            moved[EdgeIdentity(source, target, relationship)] = EdgeIdentity(
                new_source, new_target, relationship
            )
    ownership = await graph.get_edge_delete_data(list(moved)) if moved else {}
    for old, new in moved.items():
        await attach(graph.attach_edge_source_refs, new, ownership.get(old))


async def _transfer_ledger_ownership(dataset_id, remap):
    ids = {UUID(old): UUID(new) for old, new in remap.items()}
    async with get_relational_engine().get_async_session() as session:
        nodes = list(
            await session.scalars(
                select(Node).where(Node.dataset_id == dataset_id, Node.slug.in_(ids))
            )
        )
        edges = list(
            await session.scalars(
                select(Edge).where(
                    Edge.dataset_id == dataset_id,
                    or_(Edge.source_node_id.in_(ids), Edge.destination_node_id.in_(ids)),
                )
            )
        )
        for row in nodes:
            values = {column.name: getattr(row, column.name) for column in Node.__table__.columns}
            values["slug"] = ids[row.slug]
            values["attributes"] = {**(row.attributes or {}), "id": str(values["slug"])}
            # Include the owning source, not only the canonical graph ID. Reuse
            # existing ownership so repeated consolidation cannot multiply it.
            values["id"] = uuid5(
                NAMESPACE_OID,
                f"consolidated-node:{dataset_id}:{row.user_id}:{row.data_id}:{row.pipeline_run_id}:{values['slug']}",
            )
            existing = await session.scalar(
                select(Node.id)
                .where(
                    Node.dataset_id == dataset_id,
                    Node.user_id == row.user_id,
                    Node.data_id == row.data_id,
                    Node.pipeline_run_id == row.pipeline_run_id,
                    Node.slug == values["slug"],
                )
                .limit(1)
            )
            if existing is None:
                session.add(Node(**values))
            await session.delete(row)
            await session.flush()
        for row in edges:
            source = ids.get(row.source_node_id, row.source_node_id)
            target = ids.get(row.destination_node_id, row.destination_node_id)
            if source != target:
                values = {
                    column.name: getattr(row, column.name) for column in Edge.__table__.columns
                }
                values.update(source_node_id=source, destination_node_id=target)
                values["id"] = uuid5(
                    NAMESPACE_OID,
                    f"consolidated-edge:{dataset_id}:{row.user_id}:{row.data_id}:{row.pipeline_run_id}:{source}:{row.relationship_name}:{target}",
                )
                existing = await session.scalar(
                    select(Edge.id)
                    .where(
                        Edge.dataset_id == dataset_id,
                        Edge.user_id == row.user_id,
                        Edge.data_id == row.data_id,
                        Edge.pipeline_run_id == row.pipeline_run_id,
                        Edge.source_node_id == source,
                        Edge.destination_node_id == target,
                        Edge.relationship_name == row.relationship_name,
                    )
                    .limit(1)
                )
                if existing is None:
                    session.add(Edge(**values))
            await session.delete(row)
            await session.flush()
        await session.commit()
