from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.tests.unit.modules.weave.test_review_qualification import request, fact


@pytest.mark.asyncio
async def test_qualification_is_saved_before_native_ingestion_and_reused_after_failure(monkeypatch):
    from cognee.modules.weave import qualified_memory as module
    from cognee.modules.weave.review_qualification import Qualification
    from cognee.modules.weave.models import WeaveMemorySource

    records = {}

    class Session:
        async def get(self, model, identity):
            return records.get(identity) if model is WeaveMemorySource else None

        def add(self, record):
            records[(record.organization_id, record.github_repository_id, record.source_key)] = (
                record
            )

        async def commit(self):
            pass

    @asynccontextmanager
    async def session():
        yield Session()

    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=session)
    )

    async def scope(*args):
        pass

    monkeypatch.setattr(module, "set_weave_organization_scope", scope)
    calls = []

    async def qualify(req):
        calls.append("qualify")
        return Qualification(facts=[fact()])

    async def sync(*args, **kwargs):
        assert next(iter(records.values())).qualification["facts"]
        assert "TodoWrite" not in args[4].data
        assert kwargs["self_improvement"] is False
        calls.append("remember")
        if calls.count("remember") == 1:
            raise RuntimeError("provider unavailable")
        return "remember"

    monkeypatch.setattr(module, "qualify_review", qualify)
    monkeypatch.setattr(module, "sync_source", sync)
    binding = SimpleNamespace(organization_id=uuid4(), dataset_id=uuid4())
    req = request()
    with pytest.raises(RuntimeError):
        await module.sync_qualified_review(binding, None, req, llm=None, embedding=None)
    assert (
        await module.sync_qualified_review(binding, None, req, llm=None, embedding=None)
        == "remember"
    )
    assert calls == ["qualify", "remember", "remember"]
    assert (
        await module.sync_qualified_review(
            binding, None, req.model_copy(update={"artifact_revision": 0}), llm=None, embedding=None
        )
        == "stale_ignored"
    )
    with pytest.raises(ValueError, match="revision conflict"):
        await module.sync_qualified_review(
            binding, None, req.model_copy(update={"content": "changed"}), llm=None, embedding=None
        )


@pytest.mark.asyncio
async def test_empty_qualification_is_receipted_without_native_memory(monkeypatch):
    from cognee.modules.weave import qualified_memory as module
    from cognee.modules.weave.review_qualification import Qualification
    from cognee.modules.weave.models import WeaveMemorySource

    records = {}

    class Session:
        async def get(self, model, identity):
            return records.get(identity) if model is WeaveMemorySource else None

        def add(self, record):
            records[(record.organization_id, record.github_repository_id, record.source_key)] = (
                record
            )

        async def commit(self):
            pass

    @asynccontextmanager
    async def session():
        yield Session()

    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=session)
    )

    async def scope(*args):
        pass

    monkeypatch.setattr(module, "set_weave_organization_scope", scope)
    calls = []

    async def qualify(req):
        calls.append("qualify")
        return Qualification(facts=[])

    async def forbidden(*args, **kwargs):
        raise AssertionError("empty knowledge must not be ingested")

    monkeypatch.setattr(module, "qualify_review", qualify)
    monkeypatch.setattr(module, "sync_source", forbidden)
    binding = SimpleNamespace(organization_id=uuid4(), dataset_id=uuid4())
    req = request()
    for _ in range(2):
        assert (
            await module.sync_qualified_review(binding, None, req, llm=None, embedding=None)
            == "unchanged"
        )
    assert calls == ["qualify"]
    assert next(iter(records.values())).status == "completed"
