"""Connect native memory documents to code files without a second fact graph."""

from sqlalchemy import select

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.weave.memory_sources import source_data_id
from cognee.modules.weave.models import WeaveMemoryNote
from cognee.modules.weave.organizations import set_weave_organization_scope


async def sync_note_code_links(binding, repository_id, source_key, paths):
    graph = await get_graph_engine()
    ref = make_source_ref_key(
        binding.dataset_id, source_data_id(binding.organization_id, repository_id, source_key)
    )
    # Native TextDocument identity is Data.id. Older datasets keep provenance
    # in Cognee's relational ledger, so graph source refs may legitimately be empty.
    ids = {
        str(source_data_id(binding.organization_id, repository_id, source_key)),
        *(await graph.find_nodes_by_source_ref(ref)),
    }
    documents = [
        node
        for node in await graph.get_nodes(ids)
        if node.get("type") in {"TextDocument", "Document", "DocumentChunk"}
    ]
    files, _ = await graph.get_filtered_graph_data([{"type": ["CodeFileReference"]}], max_edges=0)
    targets = [
        (str(node_id), props)
        for node_id, props in files
        if str(props.get("organization_id")) == str(binding.organization_id)
        and props.get("github_repository_id") == repository_id
        and props.get("file_path") in paths
        and props.get("deleted_at") is None
    ]
    prior = await graph.find_edges_by_source_ref(ref)
    await graph.delete_edge_triples(
        [
            EdgeIdentity(edge.source_id, edge.target_id, edge.relationship_name)
            for edge in prior
            if edge.relationship_name == "memory_context_for"
        ]
    )
    await graph.add_edges(
        [
            (
                str(document["id"]),
                target,
                "memory_context_for",
                {
                    "organization_id": str(binding.organization_id),
                    "github_repository_id": repository_id,
                    "source_key": source_key,
                    "code_path": props["file_path"],
                    "indexed_sha": props.get("indexed_sha"),
                },
            )
            for document in documents
            for target, props in targets
        ],
        source_ref_key=ref,
    )


async def sync_repository_note_links(binding, repository_id):
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        notes = list(
            await session.scalars(
                select(WeaveMemoryNote).where(
                    WeaveMemoryNote.organization_id == binding.organization_id,
                    WeaveMemoryNote.github_repository_id == repository_id,
                    WeaveMemoryNote.status == "active",
                )
            )
        )
    for note in notes:
        from cognee.modules.weave.legacy_memory_notes import note_source_key

        key = await note_source_key(binding, repository_id, note.note_id)
        await sync_note_code_links(binding, repository_id, key, note.source_paths)
