"""Current memory recall must not admit historical or foreign source chunks."""

import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


def binding():
    return SimpleNamespace(organization_id=uuid4(), dataset_id=uuid4())


def source(owner, **changes):
    fact = {"kind": "symbol", "name": "authenticate", "path": "auth.py"}
    digest = hashlib.sha256(json.dumps(fact, sort_keys=True).encode()).hexdigest()
    data_id = uuid4()
    values = dict(
        organization_id=owner.organization_id,
        dataset_id=owner.dataset_id,
        github_repository_id=10,
        data_id=data_id,
        content_hash="a" * 64,
        session_id=None,
        status="completed",
        qualification={
            "facts": [fact],
            "fact_states": {digest: {"status": "current", "checked_sha": "b" * 40}},
            "native_source_tag": f"weave-current-source:{data_id}:{'a' * 64}",
        },
    )
    return SimpleNamespace(**{**values, **changes})


def test_source_allowlist_accepts_only_current_completed_owned_documents():
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources

    owner = binding()
    current = source(owner)
    legacy = source(owner, qualification={"facts": current.qualification["facts"]})
    pending = source(owner, status="processing")
    foreign_org = source(owner, organization_id=uuid4())
    foreign_dataset = source(owner, dataset_id=uuid4())
    session = source(owner, session_id="historical-session")
    other_repo = source(owner, github_repository_id=11)
    all_sources = [current, legacy, pending, foreign_org, foreign_dataset, session, other_repo]
    assert eligible_memory_sources(owner, all_sources, {10}) == [current]
    assert eligible_memory_sources(owner, all_sources) == [current, other_repo]


@pytest.mark.parametrize("status", ["needs_recheck", "superseded", "unverified", None])
def test_one_ineligible_fact_excludes_its_entire_source(status):
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources

    owner = binding()
    record = source(owner)
    second_fact = {"kind": "symbol", "name": "authorize", "path": "auth.py"}
    digest = hashlib.sha256(json.dumps(second_fact, sort_keys=True).encode()).hexdigest()
    record.qualification["facts"].append(second_fact)
    record.qualification["fact_states"][digest] = {"status": status, "checked_sha": "b" * 40}
    assert eligible_memory_sources(owner, [record]) == []


@pytest.mark.parametrize("broken", ["facts", "state", "checked_sha", "tag", "old_hash"])
def test_incomplete_or_stale_source_qualification_is_not_recallable(broken):
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources

    owner = binding()
    record = source(owner)
    if broken == "facts":
        record.qualification["facts"] = []
    elif broken == "state":
        record.qualification["fact_states"] = {}
    elif broken == "checked_sha":
        next(iter(record.qualification["fact_states"].values())).pop("checked_sha")
    elif broken == "tag":
        record.qualification["native_source_tag"] = "other-source"
    else:
        record.content_hash = "c" * 64
    assert eligible_memory_sources(owner, [record]) == []


@pytest.mark.asyncio
async def test_native_chunk_recall_filters_source_ids_in_returned_payloads(monkeypatch):
    import cognee
    from cognee.modules.search.types import SearchType
    from cognee.modules.weave.memory_retrieval import recall_current_memory

    owner = binding()
    current = source(owner)
    historical = source(owner, qualification=None)
    good = {"dataset_id": str(owner.dataset_id), "metadata": {"data_id": str(current.data_id)}}
    foreign = {"dataset_id": str(uuid4()), "metadata": {"data_id": str(current.data_id)}}
    stale = {"dataset_id": str(owner.dataset_id), "metadata": {"data_id": str(historical.data_id)}}
    missing = {"dataset_id": str(owner.dataset_id), "metadata": {}}
    native = AsyncMock(return_value=[good, foreign, stale, missing])
    monkeypatch.setattr(cognee, "recall", native)
    result = await recall_current_memory(
        owner,
        "user",
        "authentication",
        [current, historical],
        top_k=5,
        repository_ids={10},
        llm="llm",
        embedding="embedding",
    )
    assert result == [good]
    args = native.call_args.kwargs
    assert args["query_type"] is SearchType.CHUNKS
    assert args["scope"] == "graph"
    assert args["auto_route"] is False
    assert args["node_name"] == [current.qualification["native_source_tag"]]
    assert args["dataset_ids"] == [owner.dataset_id]
    assert "session_id" not in args


@pytest.mark.asyncio
async def test_native_chunk_recall_accepts_pydantic_result_metadata(monkeypatch):
    import cognee
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
    from cognee.modules.weave.memory_retrieval import recall_current_memory

    owner = binding()
    record = source(owner)
    chunk = ResponseGraphEntry(
        source="graph",
        kind="chunk",
        search_type="CHUNKS",
        text="A current fact.",
        dataset_id=str(owner.dataset_id),
        metadata={"data_id": str(record.data_id)},
    )
    monkeypatch.setattr(cognee, "recall", AsyncMock(return_value=[chunk]))
    assert await recall_current_memory(
        owner, "user", "fact", [record], top_k=2, llm=None, embedding=None
    ) == [chunk]


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [[], ["legacy"]])
async def test_empty_eligible_source_list_never_runs_unfiltered_recall(monkeypatch, sources):
    import cognee
    from cognee.modules.weave.memory_retrieval import recall_current_memory

    owner = binding()
    native = AsyncMock(side_effect=AssertionError("Unfiltered recall was attempted"))
    monkeypatch.setattr(cognee, "recall", native)
    records = [source(owner, qualification=None) for _ in sources]
    assert (
        await recall_current_memory(
            owner, "user", "fact", records, top_k=2, llm=None, embedding=None
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("node_set", [None, ["weave-current-source:source:hash"]])
async def test_sync_source_preserves_optional_native_tags(monkeypatch, existing, node_set):
    import cognee
    from cognee.modules.weave import memory_sources

    owner = binding()
    owner.tenant_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    record = source(owner, status="processing")
    record.artifact_revision = None
    data = SimpleNamespace(dataset_id=owner.dataset_id, owner_id=user.id, tenant_id=owner.tenant_id)
    session = SimpleNamespace(
        get=AsyncMock(side_effect=[record, data if existing else None, record]), commit=AsyncMock()
    )

    @asynccontextmanager
    async def get_session():
        yield session

    monkeypatch.setattr(
        memory_sources,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=get_session),
    )
    monkeypatch.setattr(memory_sources, "set_weave_organization_scope", AsyncMock())
    remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
    update = AsyncMock(return_value=SimpleNamespace(status="completed"))
    monkeypatch.setattr(cognee, "remember", remember)
    monkeypatch.setattr(cognee, "update", update)
    assert await memory_sources.sync_source(
        owner,
        user,
        10,
        "review:test",
        SimpleNamespace(),
        "new-hash",
        llm=None,
        embedding=None,
        self_improvement=False,
        node_set=node_set,
    ) == ("update" if existing else "remember")
    call = (update if existing else remember).call_args.kwargs
    assert call.get("node_set") == node_set
    assert ("node_set" in call) == (node_set is not None)
    assert record.status == "completed"
