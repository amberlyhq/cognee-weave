"""Durable validity receipts around native code and memory operations.

Caller holds the customer operation lock. Graph properties mirror these receipts;
relational receipts remain authoritative if graph refresh fails midway.
"""

import hashlib
import json
from copy import deepcopy

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.weave.memory_sources import source_records, source_data_id
from cognee.modules.weave.models import WeaveMemorySource
from cognee.modules.weave.organizations import set_weave_organization_scope


def fact_digest(fact):
    return hashlib.sha256(json.dumps(fact, sort_keys=True).encode()).hexdigest()


def invalidate_receipt(receipt, paths, head_sha):
    updated = deepcopy(receipt)
    states = updated.setdefault("fact_states", {})
    for fact in updated.get("facts", []):
        key = fact_digest(fact)
        prior = states.get(key, {})
        fact_paths = {fact["code_path"]} | {
            updated.get("evidence_paths", {}).get(e["evidence_id"])
            for e in fact.get("evidence", [])
        }
        if fact_paths.intersection(paths) and prior.get("status") != "superseded":
            states[key] = {
                **prior,
                "status": "needs_recheck",
                "invalidated_sha": head_sha,
            }
    return updated


def affected_paths(binding, repository_id, nodes, edges, changed):
    own = {
        str(i): p
        for i, p in nodes
        if str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == repository_id
        and p.get("file_path")
    }
    affected = {i for i, p in own.items() if p["file_path"] in changed}
    # Follow code dependencies, not repository membership or semantic entities.
    adjacent = {}
    for a, b, relation, _ in edges:
        if (
            str(a) in own
            and str(b) in own
            and relation not in {"part_of", "defined_in_file"}
        ):
            adjacent.setdefault(str(b), set()).add(str(a))
    frontier = set(affected)
    while frontier:
        following = set().union(*(adjacent.get(i, set()) for i in frontier)) - affected
        affected.update(following)
        frontier = following
    return set(changed) | {own[i]["file_path"] for i in affected}


def change_receipt(existing, indexed_shas, head_sha, changed, complete):
    # Once file provenance has advanced to this head, a retry may see identical
    # hashes. Preserve the pre-refresh delta, including its completeness flag.
    # A return from another head is a new transition and replaces this receipt.
    if existing and indexed_shas == {head_sha}:
        return existing
    return {
        "facts": [],
        "changed_paths": sorted(changed),
        "complete": bool(complete),
        "head_sha": head_sha,
    }


async def invalidate_changed_knowledge(binding, repository_id, head_sha, file_hashes):
    graph = await get_graph_engine()
    from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES

    nodes, edges = await graph.get_filtered_graph_data(
        [{"type": [*CODE_NODE_TYPES, "CodeRepository"]}], max_edges=1000000
    )
    previous = {
        p["file_path"]: p.get("weave_content_hash")
        for _, p in nodes
        if p.get("type") == "CodeFileReference"
        and str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == repository_id
        and p.get("file_path")
    }
    indexed_shas = {
        p.get("indexed_sha")
        for _, p in nodes
        if p.get("type") in {"CodeFileReference", "CodeRepository"}
        and str(p.get("organization_id")) == str(binding.organization_id)
        and p.get("github_repository_id") == repository_id
    }
    changed = {
        p
        for p in set(previous) | set(file_hashes)
        if previous.get(p) != file_hashes.get(p)
    }
    paths = affected_paths(binding, repository_id, nodes, edges, changed)
    records = await source_records(binding, repository_id)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        for record in records:
            if record.qualification is None:
                continue
            if record.dataset_id != binding.dataset_id:
                raise ValueError("Foreign knowledge receipt")
            current = await session.get(
                WeaveMemorySource,
                (binding.organization_id, repository_id, record.source_key),
            )
            current.qualification = invalidate_receipt(
                current.qualification, paths, head_sha
            )
        key = f"review:code-change:{head_sha}"
        identity = (binding.organization_id, repository_id, key)
        change = await session.get(WeaveMemorySource, identity)
        if change is None:
            change = WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=repository_id,
                source_key=key,
                data_id=source_data_id(*identity),
                dataset_id=binding.dataset_id,
                content_hash=head_sha.ljust(64, "0"),
                status="completed",
            )
            session.add(change)
        if change.dataset_id != binding.dataset_id or change.data_id != source_data_id(
            *identity
        ):
            raise ValueError("Foreign code change receipt")
        change.qualification = change_receipt(
            change.qualification,
            indexed_shas,
            head_sha,
            changed,
            bool(previous) and all(previous.values()),
        )
        await session.commit()
    return paths


async def stamp_file_hashes(binding, repository_id, file_hashes):
    graph = await get_graph_engine()
    nodes, _ = await graph.get_filtered_graph_data(
        [{"type": ["CodeFileReference"]}], max_edges=0
    )
    await graph.add_nodes(
        [
            (str(i), {**p, "weave_content_hash": file_hashes[p["file_path"]]})
            for i, p in nodes
            if str(p.get("organization_id")) == str(binding.organization_id)
            and p.get("github_repository_id") == repository_id
            and p.get("file_path") in file_hashes
        ]
    )
