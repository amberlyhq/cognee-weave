from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from uuid import UUID

from sqlalchemy import case, select

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES
from cognee.modules.weave.config import get_weave_embedding_config
from cognee.modules.weave.contracts import (
    RecallCandidate,
    RecallDiagnostic,
    RecallRequest,
    RecallResponse,
    RepositoryReference,
)
from cognee.modules.weave.models import WeaveRepositorySnapshot
from cognee.modules.weave.organizations import (
    OrganizationBinding,
    get_organization_binding,
    set_weave_organization_scope,
)

logger = logging.getLogger(__name__)

_VECTOR_COLLECTIONS = tuple(f"{node_type}_name" for node_type in CODE_NODE_TYPES)
_KIND_BY_TYPE = {
    "ApiEndpoint": "route",
    "CodeInsight": "insight",
    "CodeModule": "module",
    "CodeService": "service",
    "CodeSymbol": "symbol",
    "CodeTestReference": "test_ref",
    "CodeFileReference": "file_ref",
    "ExternalDependency": "dependency",
    "StorageResource": "storage",
}


def _unavailable(
    organization_id: UUID,
    request: RecallRequest,
    status: str,
    code: str,
) -> RecallResponse:
    return RecallResponse(
        status=status,
        organization_id=organization_id,
        mode=request.mode,
        diagnostics=[RecallDiagnostic(code=code)],
    )


async def recall(organization_id: UUID, request: RecallRequest) -> RecallResponse:
    """Return bounded evidence candidates, never an Amberly governance decision."""

    try:
        async with asyncio.timeout(request.deadline_ms / 1000):
            binding = await get_organization_binding(organization_id)
            if binding is None:
                return _unavailable(
                    organization_id,
                    request,
                    "unavailable",
                    "organization_not_provisioned",
                )
            return await _recall_scoped(organization_id, binding, request)
    except TimeoutError:
        return _unavailable(organization_id, request, "timed_out", "deadline_exceeded")
    except Exception:
        logger.exception("Weave recall backend failed for organization %s", organization_id)
        return _unavailable(organization_id, request, "unavailable", "backend_unavailable")


async def _load_snapshots(
    organization_id: UUID,
    repository_ids: list[int],
    primary_repository_id: int | None,
) -> list[WeaveRepositorySnapshot]:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        query = select(WeaveRepositorySnapshot).where(
            WeaveRepositorySnapshot.organization_id == organization_id,
            WeaveRepositorySnapshot.deleted_at.is_(None),
            WeaveRepositorySnapshot.indexed_sha.is_not(None),
        )
        if repository_ids:
            query = query.where(WeaveRepositorySnapshot.github_repository_id.in_(repository_ids))
        priority = (
            (
                case(
                    (WeaveRepositorySnapshot.github_repository_id == primary_repository_id, 0),
                    else_=1,
                ),
            )
            if primary_repository_id is not None
            else ()
        )
        records = await session.scalars(
            query.order_by(*priority, WeaveRepositorySnapshot.github_repository_id).limit(20)
        )
        return list(records)


def _age_seconds(updated_at: datetime | None) -> int | None:
    if updated_at is None:
        return None
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - updated_at).total_seconds()))


def _repository_reference(snapshot: WeaveRepositorySnapshot) -> RepositoryReference:
    indexed_sha = snapshot.indexed_sha
    source_ref = None
    if indexed_sha:
        source_ref = (
            f"github://{snapshot.repository_owner}/{snapshot.repository_name}@{indexed_sha}"
        )
    return RepositoryReference(
        github_repository_id=snapshot.github_repository_id,
        repository_owner=snapshot.repository_owner,
        repository_name=snapshot.repository_name,
        indexed_default_sha=indexed_sha,
        requested_default_sha=snapshot.requested_sha,
        source_ref=source_ref,
        age_seconds=_age_seconds(snapshot.updated_at),
    )


def _properties(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        return {}
    result = dict(value)
    nested = result.pop("properties", None)
    if isinstance(nested, Mapping):
        result = {**nested, **result}
    return result


def _matches_snapshot(
    properties: Mapping[str, Any],
    snapshots: Mapping[int, WeaveRepositorySnapshot],
) -> WeaveRepositorySnapshot | None:
    try:
        repository_id = int(properties.get("github_repository_id"))
    except (TypeError, ValueError):
        return None
    snapshot = snapshots.get(repository_id)
    if snapshot is None or not snapshot.indexed_sha:
        return None
    if str(properties.get("organization_id")) != str(snapshot.organization_id):
        return None
    if properties.get("indexed_sha") != snapshot.indexed_sha:
        return None
    if properties.get("deleted_at") not in (None, ""):
        return None
    return snapshot


def _candidate(
    properties: Mapping[str, Any],
    snapshot: WeaveRepositorySnapshot,
    *,
    fallback_identity: str,
    score: float | None = None,
) -> RecallCandidate | None:
    name = str(properties.get("name") or "").strip()
    if not name:
        return None
    node_type = str(properties.get("type") or "")
    kind = str(properties.get("kind") or _KIND_BY_TYPE.get(node_type) or node_type).strip()
    if not kind:
        return None
    source_path = str(properties.get("source_path") or properties.get("file_path") or "")
    fact_identity = str(properties.get("fact_identity") or fallback_identity)
    line = properties.get("line")
    try:
        line = int(line) if line is not None else None
    except (TypeError, ValueError):
        line = None
    return RecallCandidate(
        github_repository_id=snapshot.github_repository_id,
        repository_owner=snapshot.repository_owner,
        repository_name=snapshot.repository_name,
        indexed_sha=snapshot.indexed_sha,
        source_path=source_path,
        fact_identity=fact_identity,
        kind=kind,
        name=name,
        line=line,
        score=score,
    )


def _terms(request: RecallRequest) -> tuple[str, ...]:
    values = [request.query, *request.seeds]
    return tuple(value.casefold() for value in values if value.strip())


def _graph_rank(properties: Mapping[str, Any], terms: tuple[str, ...]) -> tuple[int, str, str]:
    haystack = " ".join(
        str(properties.get(key) or "")
        for key in ("name", "fact_identity", "source_path", "file_path", "description")
    ).casefold()
    match_count = sum(term in haystack for term in terms)
    return (-match_count, str(properties.get("name") or "").casefold(), haystack)


def _traversed_ids(
    seed_ids: list[str],
    edges: Iterable[Any],
    allowed_ids: set[str],
    depth: int,
) -> set[str]:
    if depth <= 0 or not seed_ids:
        return set(seed_ids)
    neighbors: dict[str, set[str]] = {}
    for edge in edges or []:
        if not isinstance(edge, (tuple, list)) or len(edge) < 2:
            continue
        source, target = str(edge[0]), str(edge[1])
        if source not in allowed_ids or target not in allowed_ids:
            continue
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    seen = set(seed_ids)
    queue = deque((node_id, 0) for node_id in seed_ids)
    while queue and len(seen) < 100:
        node_id, distance = queue.popleft()
        if distance >= depth:
            continue
        for neighbor in sorted(neighbors.get(node_id, ())):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return seen


async def _graph_candidates(
    request: RecallRequest,
    snapshots: Mapping[int, WeaveRepositorySnapshot],
    nodes: Iterable[Any],
    edges: Iterable[Any],
) -> list[RecallCandidate]:
    eligible: list[tuple[str, dict[str, Any], WeaveRepositorySnapshot]] = []
    for raw_id, raw_properties in nodes or []:
        properties = _properties(raw_properties)
        snapshot = _matches_snapshot(properties, snapshots)
        if snapshot is not None:
            eligible.append((str(raw_id), properties, snapshot))

    terms = _terms(request)
    ranked = sorted(eligible, key=lambda item: _graph_rank(item[1], terms))
    matching = [item for item in ranked if _graph_rank(item[1], terms)[0] < 0]
    selected = matching or ranked
    if request.mode == "impact_context" and selected:
        seed_ids = [item[0] for item in selected[: max(1, len(request.seeds))]]
        allowed_ids = {item[0] for item in eligible}
        traversed = _traversed_ids(seed_ids, edges, allowed_ids, request.depth)
        selected = [item for item in ranked if item[0] in traversed]

    candidates = []
    for node_id, properties, snapshot in selected:
        candidate = _candidate(properties, snapshot, fallback_identity=node_id)
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= request.top_k:
            break
    return candidates


async def _vector_candidates(
    request: RecallRequest,
    snapshots: Mapping[int, WeaveRepositorySnapshot],
    graph,
) -> list[RecallCandidate]:
    vector = await get_vector_engine_async()
    # Vector payloads intentionally contain only index metadata. Hydrate each
    # semantic hit from the same dataset-scoped graph so provenance cannot be
    # accepted from an untrusted or stale vector payload.
    candidates: dict[tuple[int, str], RecallCandidate] = {}

    available = await asyncio.gather(*(vector.has_collection(name) for name in _VECTOR_COLLECTIONS))
    collection_names = [name for name, exists in zip(_VECTOR_COLLECTIONS, available) if exists]
    if not collection_names:
        return []
    query_vector = (await vector.embedding_engine.embed_text([request.query]))[0]

    async def search_collection(collection_name: str):
        try:
            return await vector.search(
                collection_name,
                query_vector=query_vector,
                limit=min(request.top_k * 2, 50),
                include_payload=True,
            )
        except CollectionNotFoundError:
            return []

    result_groups = await asyncio.gather(
        *(search_collection(collection_name) for collection_name in collection_names)
    )
    scores: dict[str, float] = {}
    for results in result_groups:
        for result in results:
            node_id = str(result.id)
            score = float(result.score)
            scores[node_id] = min(scores.get(node_id, score), score)
    hydrated = await graph.get_nodes(list(scores))
    graph_properties = {
        str(properties.get("id")): _properties(properties) for properties in hydrated
    }
    for results in result_groups:
        for result in results:
            properties = graph_properties.get(str(result.id), {})
            snapshot = _matches_snapshot(properties, snapshots)
            if snapshot is None:
                continue
            candidate = _candidate(
                properties,
                snapshot,
                fallback_identity=str(result.id),
                score=float(result.score),
            )
            if candidate is None:
                continue
            key = (candidate.github_repository_id, candidate.fact_identity)
            previous = candidates.get(key)
            if previous is None or (candidate.score or 0) < (previous.score or 0):
                candidates[key] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (item.score if item.score is not None else float("inf"), item.name),
    )[: request.top_k]


async def _recall_scoped(
    organization_id: UUID,
    binding: OrganizationBinding,
    request: RecallRequest,
) -> RecallResponse:
    records = await _load_snapshots(
        organization_id,
        request.github_repository_ids,
        request.primary_github_repository_id,
    )
    if not records:
        return _unavailable(
            organization_id,
            request,
            "unavailable",
            "no_indexed_repository",
        )

    snapshots = {record.github_repository_id: record for record in records}
    embedding_config = get_weave_embedding_config()
    async with scoped_database_context_variables(
        binding.dataset_id,
        binding.service_user_id,
        embedding_config=embedding_config,
    ):
        graph = await get_graph_engine()
        graph_nodes, graph_edges = await graph.get_filtered_graph_data(
            [{"type": list(CODE_NODE_TYPES)}],
            max_nodes=500,
            max_edges=1000,
        )
        graph_candidates, vector_candidates = await asyncio.gather(
            _graph_candidates(request, snapshots, graph_nodes, graph_edges),
            _vector_candidates(request, snapshots, graph),
        )

    repositories = [_repository_reference(record) for record in records]
    stale = any(
        record.status != "indexed" or record.requested_sha != record.indexed_sha
        for record in records
    )
    if request.github_repository_ids:
        stale = stale or set(request.github_repository_ids) != set(snapshots)
    if request.primary_github_repository_id is not None:
        stale = stale or request.primary_github_repository_id not in snapshots
    ages = [item.age_seconds for item in repositories if item.age_seconds is not None]
    indexed_shas = {item.indexed_default_sha for item in repositories}
    indexed_sha = next(iter(indexed_shas)) if len(indexed_shas) == 1 else None
    return RecallResponse(
        status="stale" if stale else "available",
        organization_id=organization_id,
        mode=request.mode,
        indexed_default_sha=indexed_sha,
        age_seconds=max(ages) if ages else None,
        repositories=repositories,
        graph_candidates=graph_candidates,
        vector_candidates=vector_candidates,
    )
