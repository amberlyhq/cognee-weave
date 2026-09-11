"""Exact file anchors for review context, using Cognee's native code models."""

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache
from cognee.tasks.code_graph.extract_code_graph import KIND_TO_MODEL, map_facts_to_data_points


async def sync_code_files(binding, provenance, paths):
    graph = await get_graph_engine()
    types = [m.__name__ for m in KIND_TO_MODEL.values()] + ["CodeRepository"]
    nodes, _ = await graph.get_filtered_graph_data([{"type": types}], max_edges=0)
    # Only retired code nodes whose ownership was verified during migration.
    legacy = [
        str(i)
        for i, p in nodes
        if str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == provenance.github_repository_id
        and (p.get("name") if p.get("type") == "CodeRepository" else p.get("repo"))
        != provenance.repository_identity
    ]
    await graph.delete_nodes(legacy)
    own = [
        (str(i), p)
        for i, p in nodes
        if str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == provenance.github_repository_id
        and str(i) not in legacy
    ]
    paths = set(paths)
    removed = {
        i
        for i, p in own
        if p.get("type") == "CodeFileReference" and p.get("file_path") not in paths
    }
    await graph.delete_nodes(sorted(removed))
    own = [(i, p) for i, p in own if i not in removed]
    existing_files = {p.get("file_path"): i for i, p in own if p.get("type") == "CodeFileReference"}
    # Enola parses source files. Also anchor actual config/document paths so a
    # review of package.json or workflow YAML can link without LLM processing.
    missing = [
        {"kind": "file_ref", "name": path, "file": path}
        for path in sorted(paths)
        if path not in existing_files
    ]
    points = map_facts_to_data_points(missing, repository_provenance=provenance)
    files = [p for p in points if p.type == "CodeFileReference"]
    await graph.add_nodes(files)
    file_ids = {**existing_files, **{p.file_path: str(p.id) for p in files}}
    repository_id = str(points[0].id)
    await graph.add_edges([(str(p.id), repository_id, "part_of", {}) for p in files])
    await graph.add_edges(
        [
            (i, file_ids[p["file_path"]], "defined_in_file", {})
            for i, p in own
            if p.get("type") == "CodeSymbol" and p.get("file_path") in file_ids
        ]
    )
    # Native snapshot skipping is content based; an exact commit change still
    # needs a fresh provenance stamp even when all parsed code is identical.
    updated = []
    for i, p in own:
        if p.get("indexed_sha") != provenance.indexed_sha:
            p = {
                **p,
                "indexed_sha": provenance.indexed_sha,
                "pipeline_version": provenance.pipeline_version,
            }
            if p.get("type") == "CodeRepository":
                p.update(path=provenance.source_ref, source_path=provenance.source_ref)
            updated.append((i, p))
    await graph.add_nodes(updated)
    invalidate_code_graph_snapshot_cache(dataset_id=binding.dataset_id)
