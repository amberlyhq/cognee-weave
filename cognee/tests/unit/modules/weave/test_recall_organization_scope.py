"""The service establishes native tenant scope, independent of its caller."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.tests.unit.modules.weave.test_memory_retrieval import source


@pytest.fixture
def recall_backend(monkeypatch):
    import cognee
    from cognee.modules.weave import indexing, memory_sources, native_memory, organizations
    from cognee.modules.weave import review_code_links
    from cognee.modules.weave.scope import native_organization

    owner = SimpleNamespace(
        organization_id=uuid4(), dataset_id=uuid4(), service_user_id=uuid4(), tenant_id=uuid4()
    )
    current = source(owner, source_key="review:current")
    historical = source(owner, source_key="review:old", qualification={})
    foreign = source(owner, source_key="review:foreign", organization_id=uuid4())
    other_repo = source(owner, source_key="review:other-repo", github_repository_id=11)
    records = [current, historical, foreign, other_repo]
    snapshot = SimpleNamespace(
        github_repository_id=10,
        repository_owner="acme",
        repository_name="service",
        status="indexed",
        pipeline_version=native_memory.NATIVE_PIPELINE_VERSION,
        indexed_sha="b" * 40,
        requested_sha="b" * 40,
        updated_at=None,
    )

    @asynccontextmanager
    async def lock(organization_id, **kwargs):
        assert organization_id == owner.organization_id
        yield True

    @asynccontextmanager
    async def database_scope(dataset_id, user_id, **kwargs):
        assert dataset_id == owner.dataset_id and user_id == owner.service_user_id
        yield

    async def lookup(organization_id):
        assert organization_id == owner.organization_id
        return owner

    def chunk(record, **changes):
        return {
            "kind": "chunk",
            "text": record.source_key,
            "dataset_id": str(owner.dataset_id),
            "metadata": {"data_id": str(record.data_id)},
            **changes,
        }

    backend = SimpleNamespace(owner=owner, current=current, fail=None, scopes=[])

    async def native_recall(query, **kwargs):
        backend.scopes.append(native_organization.get())
        assert native_organization.get() == owner.organization_id
        assert kwargs["dataset_ids"] == [owner.dataset_id]
        assert kwargs["user"].id == owner.service_user_id
        if backend.fail == kwargs["scope"]:
            raise RuntimeError("native backend failed")
        if kwargs["scope"] == "code":
            return []
        assert kwargs["node_name"] == [current.qualification["native_source_tag"]]
        return [
            chunk(current),
            chunk(historical),
            chunk(foreign),
            chunk(other_repo),
            chunk(current, dataset_id=str(uuid4())),
        ]

    monkeypatch.setattr(indexing, "weave_operation_lock", lock)
    monkeypatch.setattr(organizations, "get_organization_binding", lookup)
    monkeypatch.setattr(native_memory, "customer_snapshots", AsyncMock(return_value=[snapshot]))
    # Keep real memory_datasets and source allowlist checks at the service boundary.
    backend.dataset_lookup = AsyncMock(return_value=SimpleNamespace(id=owner.dataset_id))
    monkeypatch.setattr(native_memory, "customer_dataset", backend.dataset_lookup)
    monkeypatch.setattr(
        native_memory, "get_user", AsyncMock(return_value=SimpleNamespace(id=owner.service_user_id))
    )
    monkeypatch.setattr(native_memory, "scoped_database_context_variables", database_scope)
    monkeypatch.setattr(native_memory, "get_weave_llm_config", lambda: None)
    monkeypatch.setattr(native_memory, "get_weave_embedding_config", lambda: None)
    monkeypatch.setattr(memory_sources, "source_records", AsyncMock(return_value=records))
    monkeypatch.setattr(review_code_links, "linked_review_context", AsyncMock(return_value=[]))
    monkeypatch.setattr(cognee, "recall", native_recall)
    return backend


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming", [None, "foreign"])
@pytest.mark.parametrize("failure", [None, "code", "graph", "ownership"])
async def test_service_sets_owned_scope_filters_sources_and_restores_caller(
    recall_backend, incoming, failure
):
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.recall import recall
    from cognee.modules.weave.scope import native_organization

    backend = recall_backend
    backend.fail = failure
    if failure == "ownership":
        backend.dataset_lookup.return_value = None
    caller = uuid4() if incoming == "foreign" else None
    token = native_organization.set(caller)
    try:
        response = await recall(
            backend.owner.organization_id,
            RecallRequest(query="authentication", primary_github_repository_id=10),
        )
        assert native_organization.get() == caller
        assert response.organization_id == backend.owner.organization_id
        if failure == "ownership":
            assert response.status == "unavailable" and backend.scopes == []
        else:
            assert backend.scopes and all(
                scope == backend.owner.organization_id for scope in backend.scopes
            )
            if failure == "code":
                assert response.status == "unavailable"
            else:
                assert response.status == "available"
                chunks = json.loads(response.native_memory)
                if failure == "graph":
                    assert chunks == [] and response.diagnostics[0].code == "backend_unavailable"
                else:
                    assert len(chunks) == 1 and chunks[0]["text"] == "review:current"
                    assert chunks[0]["metadata"]["data_id"] == str(backend.current.data_id)
                    assert not response.diagnostics
        backend.dataset_lookup.assert_awaited_once_with(backend.owner)
    finally:
        native_organization.reset(token)


@pytest.mark.asyncio
async def test_large_native_result_reaches_caller_intact(recall_backend):
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.recall import recall

    recall_backend.current.source_key = "review:" + "x" * 70000
    response = await recall(
        recall_backend.owner.organization_id,
        RecallRequest(query="large", github_repository_ids=[10]),
    )
    assert response.status == "available"
    assert json.loads(response.native_memory)[0]["text"] == recall_backend.current.source_key
