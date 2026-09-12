"""Read and retire legacy review annotations without creating new fact nodes.

New agent-selected notes link native documents directly via note_code_links.
Existing annotations remain readable until the host updates or retires their source.
"""

import json
from uuid import UUID

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine


async def forget_review_code_links(binding, record):
    if record.qualification is None:
        return
    if record.organization_id != binding.organization_id or record.dataset_id != binding.dataset_id:
        raise ValueError("Foreign review receipt")
    async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_filtered_graph_data([{"type": ["ReviewKnowledge"]}], max_edges=0)
        await graph.delete_nodes(
            [
                str(i)
                for i, p in nodes
                if str(p.get("organization_id")) == str(binding.organization_id)
                and p.get("github_repository_id") == record.github_repository_id
                and p.get("source_key") == record.source_key
                and str(p.get("source_data_id")) == str(record.data_id)
            ]
        )


async def linked_review_context(binding, code_result, records, limit):
    """Follow code -> file <- qualified fact; never guess links from query text."""
    ids = set()

    def collect(value, depth=0):
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                collect(json.loads(value), depth + 1)
            except (ValueError, TypeError):
                pass
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("id", "target_id"):
                    try:
                        ids.add(str(UUID(str(item))))
                    except ValueError:
                        pass
                else:
                    collect(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                collect(item, depth + 1)

    collect(code_result)
    if not ids:
        return []
    completed = {
        (r.github_repository_id, r.source_key): r.artifact_revision or 0
        for r in records
        if r.status == "completed"
        and r.qualification is not None
        and r.organization_id == binding.organization_id
        and r.dataset_id == binding.dataset_id
    }
    async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
        graph = await get_graph_engine()
        selected = await graph.get_nodes(sorted(ids))
        files = {
            str(p["id"]): p
            for p in selected
            if p.get("type") == "CodeFileReference"
            and str(p.get("organization_id")) == str(binding.organization_id)
        }
        if not files:
            return []
        nodes, edges = await graph.get_id_filtered_graph_data(list(files))
        props = dict(nodes)
        output = {}
        for a, b, relation, _ in edges:
            fact = props.get(a, {})
            if (
                relation != "review_context_for"
                or b not in files
                or fact.get("type") != "ReviewKnowledge"
                or str(fact.get("organization_id")) != str(binding.organization_id)
                or fact.get("github_repository_id") != files[b].get("github_repository_id")
                or completed.get((fact.get("github_repository_id"), fact.get("source_key")))
                != fact.get("artifact_revision")
            ):
                continue
            output[a] = {
                key: fact.get(key)
                for key in (
                    "statement",
                    "certainty",
                    "historical",
                    "knowledge_status",
                    "checked_sha",
                    "invalidated_sha",
                    "replacement_source",
                    "github_repository_id",
                    "code_path",
                    "reviewed_head_sha",
                    "source_key",
                    "evidence",
                )
            }
            receipt = next(
                r
                for r in records
                if r.github_repository_id == fact["github_repository_id"]
                and r.source_key == fact["source_key"]
            )
            from cognee.modules.weave.knowledge_lifecycle import fact_digest

            stored_fact = next(
                (
                    f
                    for f in receipt.qualification["facts"]
                    if f["statement"] == fact["statement"] and f["code_path"] == fact["code_path"]
                ),
                None,
            )
            if stored_fact is None:
                continue
            state = receipt.qualification.get("fact_states", {}).get(fact_digest(stored_fact), {})
            output[a].update(
                knowledge_status=state.get("status", "historical"),
                historical=state.get("status") != "current",
                checked_sha=state.get("checked_sha"),
                invalidated_sha=state.get("invalidated_sha"),
                replacement_source=state.get("replacement_source"),
            )
            output[a].update(fact_id=a, code_file_id=b, indexed_sha=files[b].get("indexed_sha"))
        return [output[key] for key in sorted(output)[:limit]]
