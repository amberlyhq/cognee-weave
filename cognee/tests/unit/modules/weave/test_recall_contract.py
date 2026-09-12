import asyncio
from uuid import UUID

import pytest
from pydantic import ValidationError


def test_recall_request_is_allow_listed_and_bounded():
    from cognee.modules.weave.contracts import RecallRequest

    request = RecallRequest(
        mode="repository_context",
        query="Message",
        github_repository_ids=[920001, 920002],
        primary_github_repository_id=920002,
        seeds=["symbol:Message"],
        top_k=25,
        depth=4,
    )

    assert request.top_k == 25
    assert request.depth == 4
    assert request.github_repository_ids == [920001, 920002]
    assert request.primary_github_repository_id == 920002
    assert set(RecallRequest.model_json_schema()["properties"]) == {
        "mode",
        "query",
        "github_repository_ids",
        "primary_github_repository_id",
        "seeds",
        "top_k",
        "depth",
    }

    rejected = (
        {"mode": "cypher", "query": "Message"},
        {"mode": "repository_context", "query": "Message", "dataset_id": "anything"},
        {"mode": "repository_context", "query": "Message", "schema": "public"},
        {"mode": "repository_context", "query": "Message", "primary_github_repository_id": 0},
        {"mode": "repository_context", "query": "Message", "deadline_ms": 100},
    )
    for payload in rejected:
        with pytest.raises(ValidationError):
            RecallRequest.model_validate(payload)


def test_recall_response_is_provenance_first_and_has_no_governance_decision_fields():
    from cognee.modules.weave.contracts import (
        RecallCandidate,
        RecallResponse,
        RepositoryReference,
    )

    organization_id = UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01")
    sha = "a" * 40
    repository = RepositoryReference(
        github_repository_id=920001,
        repository_owner="amberlyhq",
        repository_name="weave-alpha",
        indexed_default_sha=sha,
        requested_default_sha=sha,
        source_ref=f"github://amberlyhq/weave-alpha@{sha}",
        age_seconds=12,
    )
    candidate = RecallCandidate(
        github_repository_id=920001,
        repository_owner="amberlyhq",
        repository_name="weave-alpha",
        indexed_sha=sha,
        source_path="main.go",
        fact_identity="symbol:Message",
        kind="symbol",
        name="Message",
        line=3,
        score=0.1,
    )
    response = RecallResponse(
        status="available",
        organization_id=organization_id,
        mode="repository_context",
        indexed_default_sha=sha,
        age_seconds=12,
        repositories=[repository],
        graph_candidates=[candidate],
        vector_candidates=[candidate],
        diagnostics=[],
    )

    dumped = response.model_dump(mode="json")
    assert dumped["graph_candidates"][0]["indexed_sha"] == sha
    assert dumped["repositories"][0]["source_ref"].endswith("@" + sha)

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    forbidden = {"risk", "confidence", "attention", "action"}
    assert forbidden.isdisjoint({key.lower() for key in keys(dumped)})


@pytest.mark.asyncio
async def test_cross_repository_snapshot_lookup_preserves_all_selected_repositories(monkeypatch):
    from cognee.modules.weave import recall as recall_module

    captured = {}

    class Session:
        async def scalars(self, query):
            captured["query"] = query
            return []

    class SessionContext:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *_args):
            return None

    class Engine:
        def get_async_session(self):
            return SessionContext()

    async def scope(_session, _organization_id):
        return None

    monkeypatch.setattr(recall_module, "get_relational_engine", lambda: Engine())
    monkeypatch.setattr(recall_module, "set_weave_organization_scope", scope)

    await recall_module._load_snapshots(
        UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01"),
        [],
        920021,
    )

    assert captured["query"]._limit_clause is None
    assert "CASE WHEN" in str(captured["query"])


@pytest.mark.parametrize("status", ["available", "stale", "unavailable", "timed_out"])
def test_recall_statuses_are_explicit(status):
    from cognee.modules.weave.contracts import RecallResponse

    response = RecallResponse(
        status=status,
        organization_id=UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01"),
        mode="repository_context",
    )
    assert response.status == status


@pytest.mark.asyncio
async def test_recall_returns_safe_unavailable_when_organization_is_unknown(monkeypatch):
    from contextlib import asynccontextmanager

    from cognee.modules.weave import indexing, organizations
    from cognee.modules.weave import recall as recall_module
    from cognee.modules.weave.contracts import RecallRequest

    async def missing(_organization_id):
        return None

    @asynccontextmanager
    async def unlocked(*args, **kwargs):
        yield True

    monkeypatch.setattr(organizations, "get_organization_binding", missing)
    monkeypatch.setattr(indexing, "weave_operation_lock", unlocked)
    response = await recall_module.recall(
        UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01"),
        RecallRequest(query="Message"),
    )

    assert response.status == "unavailable"
    assert [item.code for item in response.diagnostics] == ["organization_not_provisioned"]


@pytest.mark.asyncio
async def test_recall_has_no_added_deadline_and_backend_errors_do_not_escape(monkeypatch):
    from cognee.modules.weave import native_memory
    from cognee.modules.weave import recall as recall_module
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.organizations import OrganizationBinding

    organization_id = UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01")
    binding = OrganizationBinding(
        organization_id=organization_id,
        tenant_id=UUID("10000000-0000-0000-0000-000000000001"),
        service_user_id=UUID("10000000-0000-0000-0000-000000000002"),
        dataset_id=UUID("10000000-0000-0000-0000-000000000003"),
        graph_schema="dataset_10000000000000000000000000000003",
        vector_schema="dataset_10000000000000000000000000000003",
    )

    async def found(_organization_id):
        return binding

    sentinel = object()

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(0)
        return sentinel

    def forbidden_timeout(*_args, **_kwargs):
        raise AssertionError("Weave must not wrap native recall in a timeout")

    monkeypatch.setattr(asyncio, "timeout", forbidden_timeout)

    async def broken(*_args, **_kwargs):
        raise RuntimeError("secret database details")

    monkeypatch.setattr(recall_module, "get_organization_binding", found)
    monkeypatch.setattr(native_memory, "recall_repository_memory", slow)
    timed_out = await recall_module.recall(
        organization_id,
        RecallRequest(query="Message"),
    )
    monkeypatch.setattr(native_memory, "recall_repository_memory", broken)
    unavailable = await recall_module.recall(
        organization_id,
        RecallRequest(query="Message"),
    )

    assert timed_out is sentinel
    assert unavailable.status == "unavailable"
    assert unavailable.diagnostics[0].code == "backend_unavailable"
    assert "secret" not in unavailable.model_dump_json()


def test_native_payloads_and_requested_search_sizes_are_not_artificially_capped():
    from cognee.modules.weave.contracts import (
        RecallRequest,
        RecallResponse,
        SurfaceResponse,
        ReviewMemoryRequest,
    )

    org = UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01")
    request = RecallRequest(query="x" * 3000, top_k=100, depth=8, seeds=["symbol" * 100])
    assert request.top_k == 100
    payload = "x" * 1_100_000
    assert (
        RecallResponse(
            status="available",
            organization_id=org,
            mode="repository_context",
            native_memory=payload,
        ).native_memory
        == payload
    )
    assert (
        SurfaceResponse(organization_id=org, surface="export", native_graph=payload).native_graph
        == payload
    )
    assert (
        ReviewMemoryRequest(
            github_repository_id=1,
            review_id=org,
            head_sha="a" * 40,
            lifecycle_generation=1,
            artifact_revision=0,
            content=payload,
        ).content
        == payload
    )
