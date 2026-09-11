"""Durable merge selection and retries around native memory failures."""

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.weave.knowledge_lifecycle import fact_digest
from cognee.modules.weave.memory_sources import source_data_id
from cognee.modules.weave.models import WeaveMemorySource


class ReceiptStore:
    """Detached reads and explicit commits model the relational receipt boundary."""

    def __init__(self):
        self.sources = {}
        self.data = {}

    @asynccontextmanager
    async def get_async_session(self):
        store = self

        class Session:
            def __init__(self):
                self.sources = deepcopy(store.sources)
                self.data = deepcopy(store.data)

            async def get(self, model, key):
                return (self.sources if model is WeaveMemorySource else self.data).get(key)

            def add(self, record):
                key = (record.organization_id, record.github_repository_id, record.source_key)
                self.sources[key] = record

            async def commit(self):
                store.sources = deepcopy(self.sources)
                store.data = deepcopy(self.data)

        yield Session()


@pytest.fixture
def merge_case(monkeypatch, tmp_path):
    import cognee
    from cognee.modules.weave import memory_sources, merge_memory, merge_qualification
    from cognee.modules.weave import review_code_links

    owner = SimpleNamespace(
        organization_id=uuid4(), dataset_id=uuid4(), service_user_id=uuid4(), tenant_id=uuid4()
    )
    user = SimpleNamespace(id=owner.service_user_id)
    request = SimpleNamespace(github_repository_id=42)
    change = SimpleNamespace(
        base_sha="a" * 40, head_sha="b" * 40, changed_paths=["auth.py"], review_context="review"
    )
    snapshot = SimpleNamespace(updated_at=datetime(2026, 9, 11, tzinfo=timezone.utc))
    (tmp_path / "auth.py").write_text("def authenticate(user):\n    return user.is_active\n")
    store = ReceiptStore()
    previous_key = "review:qualified:" + str(uuid4())
    fact = {
        "statement": "Authentication returns whether the user is active.",
        "code_path": "auth.py",
        "certainty": "observed",
        "evidence": [{"evidence_id": "historical", "quote": "return user.is_active"}],
    }
    previous = WeaveMemorySource(
        organization_id=owner.organization_id,
        github_repository_id=42,
        source_key=previous_key,
        data_id=source_data_id(owner.organization_id, 42, previous_key),
        dataset_id=owner.dataset_id,
        content_hash="a" * 64,
        status="completed",
        qualification={
            "facts": [fact],
            "fact_states": {
                fact_digest(fact): {"status": "needs_recheck", "invalidated_sha": change.head_sha}
            },
        },
    )
    store.sources[(owner.organization_id, 42, previous_key)] = previous
    case = SimpleNamespace(
        binding=owner,
        user=user,
        request=request,
        change=change,
        snapshot=snapshot,
        repository=tmp_path,
        store=store,
        previous_key=previous_key,
        fact=fact,
        native_calls=[],
        qualification_calls=[],
        failures=0,
    )

    async def records(binding, repository_id=None):
        return deepcopy(
            [
                record
                for record in store.sources.values()
                if record.organization_id == binding.organization_id
                and (repository_id is None or record.github_repository_id == repository_id)
            ]
        )

    async def qualify(**kwargs):
        case.qualification_calls.append(deepcopy(kwargs))
        fresh = deepcopy(fact)
        fresh["evidence"] = [
            {"evidence_id": "merge:" + kwargs["head_sha"], "quote": "return user.is_active"}
        ]
        return {
            "facts": [fresh],
            "decisions": [
                {
                    "fact_id": prior["fact_id"],
                    "status": "keep",
                    "reason": "Still present in merged code",
                }
                for prior in kwargs["prior_facts"]
            ],
            "coverage_complete": True,
        }

    async def native(item=None, **kwargs):
        item = item or kwargs["data"]
        # Observe committed qualification at the exact provider boundary.
        case.native_calls.append((deepcopy(item), deepcopy(kwargs), deepcopy(store.sources)))
        if case.failures:
            case.failures -= 1
            raise RuntimeError("native provider unavailable")
        store.data[item.data_id] = SimpleNamespace(
            id=item.data_id,
            dataset_id=owner.dataset_id,
            owner_id=user.id,
            tenant_id=owner.tenant_id,
            external_metadata=item.external_metadata,
        )
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(merge_memory, "get_relational_engine", lambda: store)
    monkeypatch.setattr(memory_sources, "get_relational_engine", lambda: store)
    monkeypatch.setattr(merge_memory, "set_weave_organization_scope", AsyncMock())
    monkeypatch.setattr(memory_sources, "set_weave_organization_scope", AsyncMock())
    monkeypatch.setattr(merge_memory, "source_records", records)
    monkeypatch.setattr(merge_qualification, "qualify_merge", qualify)
    monkeypatch.setattr(cognee, "remember", native)
    monkeypatch.setattr(cognee, "update", native)
    case.sync_links = AsyncMock()
    monkeypatch.setattr(review_code_links, "sync_review_code_links", case.sync_links)
    return case


async def run_merge(case):
    from cognee.modules.weave.merge_memory import process_merge

    return await process_merge(
        case.binding,
        case.user,
        case.request,
        case.change,
        case.repository,
        case.snapshot,
        llm=None,
        embedding=None,
    )


def merge_records(case, *, plans=False):
    return [
        r
        for r in case.store.sources.values()
        if r.source_key.startswith("review:merge:") and r.source_key.endswith(":plan") == plans
    ]


def previous_state(case):
    record = case.store.sources[(case.binding.organization_id, 42, case.previous_key)]
    return record.qualification["fact_states"][fact_digest(case.fact)]


@pytest.mark.asyncio
async def test_selection_is_committed_before_native_failure(merge_case):
    case = merge_case
    case.failures = 1
    with pytest.raises(RuntimeError, match="native provider"):
        await run_merge(case)
    plans, batches = merge_records(case, plans=True), merge_records(case)
    assert len(plans) == len(batches) == 1
    assert plans[0].qualification["batches"][0]["prior_facts"][0]["source_key"] == case.previous_key
    assert batches[0].qualification["facts"]
    assert batches[0].qualification["decisions"]
    assert batches[0].status != "completed"
    at_native = case.native_calls[0][2]
    saved = at_native[(case.binding.organization_id, 42, batches[0].source_key)]
    assert saved.qualification == batches[0].qualification
    assert case.native_calls[0][1]["node_set"] == [saved.qualification["native_source_tag"]]


@pytest.mark.asyncio
async def test_failed_native_write_never_applies_fact_decisions(merge_case):
    case = merge_case
    case.failures = 1
    before = deepcopy(previous_state(case))
    with pytest.raises(RuntimeError, match="native provider"):
        await run_merge(case)
    assert previous_state(case) == before
    assert merge_records(case, plans=True)[0].status != "completed"
    case.sync_links.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_reuses_exact_saved_selection_without_requalification(merge_case):
    case = merge_case
    case.failures = 1
    with pytest.raises(RuntimeError):
        await run_merge(case)
    selected = deepcopy(merge_records(case)[0].qualification)
    # Changed optional context must not rewrite an already selected retry payload.
    case.change.review_context = "different review context"
    result = await run_merge(case)
    assert result["status"] == "completed"
    assert len(case.qualification_calls) == 1
    assert len(case.native_calls) == 2
    assert case.native_calls[0][0].data == case.native_calls[1][0].data
    assert merge_records(case)[0].qualification == selected
    assert previous_state(case)["status"] == "superseded"
    assert previous_state(case)["replacement_source"] == merge_records(case)[0].source_key
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources

    assert eligible_memory_sources(
        case.binding, list(case.store.sources.values())
    ) == merge_records(case)


@pytest.mark.asyncio
async def test_completed_manifest_replay_has_no_more_native_or_qualification_work(merge_case):
    case = merge_case
    first = await run_merge(case)
    receipt_snapshot = deepcopy(
        {key: value.qualification for key, value in case.store.sources.items()}
    )
    replay = await run_merge(case)
    assert replay == {**first, "status": "unchanged"}
    assert len(case.qualification_calls) == len(case.native_calls) == 1
    assert {
        key: value.qualification for key, value in case.store.sources.items()
    } == receipt_snapshot
    assert case.sync_links.await_count == 1


@pytest.mark.asyncio
async def test_return_to_same_head_in_a_new_snapshot_epoch_gets_new_manifest(merge_case):
    case = merge_case
    await run_merge(case)
    original = merge_records(case, plans=True)[0].source_key
    case.snapshot.updated_at += timedelta(seconds=1)
    case.change.base_sha, case.change.head_sha = "b" * 40, "c" * 40
    await run_merge(case)
    case.snapshot.updated_at += timedelta(seconds=1)
    case.change.base_sha, case.change.head_sha = "a" * 40, "b" * 40
    result = await run_merge(case)
    assert result["status"] == "completed"
    plans = merge_records(case, plans=True)
    assert len(plans) == 3
    same_head = [
        record.source_key for record in plans if record.qualification["head_sha"] == "b" * 40
    ]
    assert len(same_head) == 2 and original in same_head
    assert len(case.qualification_calls) == len(case.native_calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["dataset_id", "data_id"])
async def test_save_receipt_rejects_foreign_dataset_or_wrong_source_identity(merge_case, field):
    from cognee.modules.weave.merge_memory import save_receipt

    case = merge_case
    key = "review:merge:existing:plan"
    identity = (case.binding.organization_id, 42, key)
    record = WeaveMemorySource(
        organization_id=case.binding.organization_id,
        github_repository_id=42,
        source_key=key,
        data_id=source_data_id(*identity),
        dataset_id=case.binding.dataset_id,
        qualification={"facts": []},
        content_hash="a" * 64,
        status="qualified",
    )
    setattr(record, field, uuid4())
    case.store.sources[identity] = record
    with pytest.raises(ValueError, match="[Ff]oreign|identity"):
        await save_receipt(case.binding, 42, key, {"facts": [], "modified": True})
    assert case.store.sources[identity].qualification == {"facts": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["dataset_id", "data_id"])
async def test_completed_manifest_replay_rejects_foreign_receipt(merge_case, field):
    case = merge_case
    await run_merge(case)
    manifest = merge_records(case, plans=True)[0]
    setattr(manifest, field, uuid4())
    with pytest.raises(ValueError, match="[Ff]oreign|identity"):
        await run_merge(case)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["dataset_id", "owner_id", "tenant_id"])
async def test_foreign_existing_native_data_stops_retry_before_provider_or_decisions(
    merge_case, field
):
    case = merge_case
    case.failures = 1
    with pytest.raises(RuntimeError):
        await run_merge(case)
    batch = merge_records(case)[0]
    data = SimpleNamespace(
        id=batch.data_id,
        dataset_id=case.binding.dataset_id,
        owner_id=case.user.id,
        tenant_id=case.binding.tenant_id,
    )
    setattr(data, field, uuid4())
    case.store.data[batch.data_id] = data
    before = deepcopy(previous_state(case))
    with pytest.raises(ValueError, match="does not belong|[Ff]oreign"):
        await run_merge(case)
    assert len(case.native_calls) == 1
    assert previous_state(case) == before
    assert merge_records(case, plans=True)[0].status != "completed"
