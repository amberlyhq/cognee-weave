"""Source-backed review annotations using native graph/provenance APIs.

Semantic extraction remains native Cognee. These deterministic annotations do
not infer program behavior or resolve names: they connect accepted statements
to exact, tenant/repository-scoped file identities, retaining their uncertainty.
Callers hold the organization's operation lock.
"""

import hashlib
import json
import re
from uuid import UUID, uuid5

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models import Data
from cognee.modules.weave.memory_sources import source_data_id, source_records
from cognee.modules.weave.review_qualification import Qualification


class ReviewKnowledge(DataPoint):
    name: str
    statement: str
    certainty: str
    historical: bool = True
    knowledge_status: str = "historical"
    checked_sha: str | None = None
    invalidated_sha: str | None = None
    replacement_source: str | None = None
    organization_id: UUID
    github_repository_id: int
    source_key: str
    source_data_id: UUID
    reviewed_head_sha: str
    artifact_revision: int
    code_path: str
    evidence: list[dict]
    metadata: dict = {"index_fields": []}


async def forget_review_code_links(binding, record):
    if record.qualification is None:
        return
    if (
        record.organization_id != binding.organization_id
        or record.dataset_id != binding.dataset_id
    ):
        raise ValueError("Foreign review receipt")
    async with scoped_database_context_variables(
        binding.dataset_id, binding.service_user_id
    ):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_filtered_graph_data(
            [{"type": ["ReviewKnowledge"]}], max_edges=0
        )
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


def build_review_links(binding, receipt, head_sha, code_nodes):
    if (
        receipt.organization_id != binding.organization_id
        or receipt.dataset_id != binding.dataset_id
    ):
        raise ValueError("Foreign review receipt")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha or ""):
        raise ValueError("Review head is missing")
    facts = Qualification(facts=receipt.qualification["facts"]).facts
    files = {}
    for node_id, props in code_nodes:
        if (
            props.get("type") == "CodeFileReference"
            and str(props.get("organization_id")) == str(binding.organization_id)
            and props.get("github_repository_id") == receipt.github_repository_id
            and props.get("deleted_at") is None
        ):
            files.setdefault(props.get("file_path"), []).append((str(node_id), props))
    nodes, edges = [], []
    for fact in facts:
        digest = hashlib.sha256(
            json.dumps(fact.model_dump(), sort_keys=True).encode()
        ).hexdigest()
        state = receipt.qualification.get("fact_states", {}).get(digest, {})
        node = ReviewKnowledge(
            id=uuid5(
                binding.organization_id,
                f"review-fact.v1:{receipt.github_repository_id}:{receipt.source_key}:{digest}",
            ),
            knowledge_status=state.get("status", "historical"),
            historical=state.get("status") != "current",
            checked_sha=state.get("checked_sha"),
            invalidated_sha=state.get("invalidated_sha"),
            replacement_source=state.get("replacement_source"),
            name=fact.statement,
            statement=fact.statement,
            certainty=fact.certainty,
            organization_id=binding.organization_id,
            github_repository_id=receipt.github_repository_id,
            source_key=receipt.source_key,
            source_data_id=receipt.data_id,
            reviewed_head_sha=head_sha,
            artifact_revision=receipt.artifact_revision or 0,
            code_path=fact.code_path,
            evidence=[e.model_dump() for e in fact.evidence],
        )
        nodes.append(node)
        # Every extra path comes from server-attached merged-source evidence.
        linked_paths = {fact.code_path} | {
            receipt.qualification.get("evidence_paths", {}).get(e.evidence_id)
            for e in fact.evidence
        }
        for path in sorted(p for p in linked_paths if p):
            matches = files.get(path, [])
            if len(matches) == 1:
                target, props = matches[0]
                edges.append(
                    (
                        str(node.id),
                        target,
                        "review_context_for",
                        {
                            "edge_text": fact.statement,
                            "certainty": fact.certainty,
                            "historical": node.historical,
                            "knowledge_status": node.knowledge_status,
                            "checked_sha": node.checked_sha,
                            "reviewed_head_sha": head_sha,
                            "indexed_sha": props.get("indexed_sha"),
                            "organization_id": str(binding.organization_id),
                            "github_repository_id": receipt.github_repository_id,
                            "code_path": path,
                        },
                    )
                )
    return nodes, edges


async def sync_review_code_links(binding, repository_id=None):
    from cognee.modules.weave.native_memory import customer_dataset

    if await customer_dataset(binding) is None:
        raise ValueError("Foreign or missing customer dataset")
    records = await source_records(binding, repository_id)
    records = [
        r for r in records if r.qualification is not None and r.status == "completed"
    ]
    async with scoped_database_context_variables(
        binding.dataset_id, binding.service_user_id
    ):
        graph = await get_graph_engine()
        existing, _ = await graph.get_filtered_graph_data(
            [{"type": ["CodeFileReference", "ReviewKnowledge"]}], max_edges=0
        )
        for record in records:
            if (
                record.organization_id != binding.organization_id
                or record.dataset_id != binding.dataset_id
            ):
                raise ValueError("Foreign review receipt")
            if record.data_id != source_data_id(
                binding.organization_id, record.github_repository_id, record.source_key
            ):
                raise ValueError("Foreign review source identity")
            prior = [
                str(i)
                for i, p in existing
                if p.get("type") == "ReviewKnowledge"
                and p.get("source_key") == record.source_key
                and str(p.get("organization_id")) == str(binding.organization_id)
                and p.get("github_repository_id") == record.github_repository_id
            ]
            if not record.qualification["facts"]:
                await graph.delete_nodes(prior)
                continue
            async with get_relational_engine().get_async_session() as session:
                data = await session.get(Data, record.data_id)
            if data is None or (data.dataset_id, data.owner_id, data.tenant_id) != (
                binding.dataset_id,
                binding.service_user_id,
                binding.tenant_id,
            ):
                raise ValueError("Foreign or missing qualified source data")
            metadata = data.external_metadata or {}
            expected_key = (
                "review:merge:" + str(metadata.get("merge_source"))
                if metadata.get("memory_kind") == "qualified_merge"
                else "review:qualified:" + str(metadata.get("review_id"))
            )
            if (
                metadata.get("github_repository_id") != record.github_repository_id
                or expected_key != record.source_key
            ):
                raise ValueError("Qualified source identity mismatch")
            nodes, edges = build_review_links(
                binding, record, metadata.get("head_sha"), existing
            )
            desired = {str(n.id) for n in nodes}
            await graph.delete_nodes([i for i in prior if i not in desired])
            ref = make_source_ref_key(binding.dataset_id, data.id)
            # Attach native ownership so forget/update cleans up annotations too.
            await graph.add_nodes(nodes, source_ref_key=ref)
            from cognee.infrastructure.databases.provenance import EdgeIdentity

            old_edges = await graph.find_edges_by_source_ref(ref)
            await graph.delete_edge_triples(
                [
                    EdgeIdentity(e.source_id, e.target_id, e.relationship_name)
                    for e in old_edges
                    if e.relationship_name == "review_context_for"
                    and e.source_id in desired
                ]
            )
            await graph.add_edges(edges, source_ref_key=ref)
            source_nodes = await graph.find_nodes_by_source_ref(ref)
            documents = await graph.get_nodes(source_nodes)
            chunks = [p["id"] for p in documents if p.get("type") == "DocumentChunk"]
            await graph.add_edges(
                [
                    (
                        str(n.id),
                        str(chunk),
                        "supported_by",
                        {
                            "historical": n.historical,
                            "certainty": n.certainty,
                            "knowledge_status": n.knowledge_status,
                        },
                    )
                    for n in nodes
                    for chunk in chunks
                ],
                source_ref_key=ref,
            )


async def linked_review_context(binding, code_result, records, limit):
    """Follow code -> file <- qualified fact; never guess links from query text."""
    ids = set()

    def collect(value, depth=0):
        if depth > 12 or len(ids) >= 1000:
            return
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
    async with scoped_database_context_variables(
        binding.dataset_id, binding.service_user_id
    ):
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
                or fact.get("github_repository_id")
                != files[b].get("github_repository_id")
                or completed.get(
                    (fact.get("github_repository_id"), fact.get("source_key"))
                )
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
                    if f["statement"] == fact["statement"]
                    and f["code_path"] == fact["code_path"]
                ),
                None,
            )
            if stored_fact is None:
                continue
            state = receipt.qualification.get("fact_states", {}).get(
                fact_digest(stored_fact), {}
            )
            output[a].update(
                knowledge_status=state.get("status", "historical"),
                historical=state.get("status") != "current",
                checked_sha=state.get("checked_sha"),
                invalidated_sha=state.get("invalidated_sha"),
                replacement_source=state.get("replacement_source"),
            )
            output[a].update(
                fact_id=a, code_file_id=b, indexed_sha=files[b].get("indexed_sha")
            )
        return [output[key] for key in sorted(output)[:limit]]
