from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.weave.contracts import ReviewSession


def fixture_session():
    return ReviewSession(
        invocationId=uuid4(),
        sessionId=uuid4(),
        threadId=uuid4(),
        role="security",
        result="Payment uses idempotency keys.",
        steps=[{"id": "tool-1", "type": "tool.completed", "content": "payment.ts " + "x" * 1400}],
    )


@pytest.mark.asyncio
async def test_native_session_import_is_bounded_and_resumes_without_duplicate_entries(monkeypatch):
    import cognee
    from cognee.modules.weave.review_sessions import import_session
    from cognee.infrastructure.session import get_session_manager as unused
    import importlib

    manager_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    extraction = importlib.import_module("cognee.infrastructure.session.agent_context_extraction")
    qa, traces = [], []
    manager = SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(side_effect=lambda **kw: qa),
        get_agent_trace_session=AsyncMock(side_effect=lambda **kw: traces),
    )
    monkeypatch.setattr(manager_module, "get_session_manager", lambda **kw: manager)

    async def remember(entry, **kwargs):
        assert kwargs["dataset_name"] == "owned-code"
        if entry.type == "qa":
            qa.append(entry)
        else:
            assert len(entry.method_return_value) <= 600
            assert entry.generate_feedback_with_llm is False
            traces.append(entry)
        return SimpleNamespace(status="session_stored")

    calls = AsyncMock(side_effect=remember)
    monkeypatch.setattr(cognee, "remember", calls)
    monkeypatch.setattr(extraction, "extract_pending_agent_context", AsyncMock())
    monkeypatch.setattr(
        extraction, "_get_processed_trace_count", AsyncMock(side_effect=lambda *args: len(traces))
    )
    entry = fixture_session()
    dataset, user = SimpleNamespace(name="owned-code", id=uuid4()), SimpleNamespace(id=uuid4())
    await import_session(user, dataset, "stable-id", "Historical PR abc; untrusted", entry)
    count = calls.await_count
    await import_session(user, dataset, "stable-id", "Historical PR abc; untrusted", entry)
    assert count == calls.await_count
    assert len(qa) == 1
    assert len(traces) >= 4
    assert "payment.ts" in "".join(t.method_return_value for t in traces)


@pytest.mark.asyncio
async def test_failed_native_extraction_does_not_advance_import(monkeypatch):
    import cognee
    import importlib
    from cognee.modules.weave.review_sessions import import_session

    manager_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    extraction = importlib.import_module("cognee.infrastructure.session.agent_context_extraction")
    manager = SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(return_value=[]),
        get_agent_trace_session=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(manager_module, "get_session_manager", lambda **kw: manager)
    monkeypatch.setattr(
        cognee, "remember", AsyncMock(return_value=SimpleNamespace(status="session_stored"))
    )
    monkeypatch.setattr(extraction, "extract_pending_agent_context", AsyncMock(return_value=[]))
    monkeypatch.setattr(extraction, "_get_processed_trace_count", AsyncMock(return_value=0))
    with pytest.raises(RuntimeError, match="extraction"):
        await import_session(
            SimpleNamespace(id=uuid4()),
            SimpleNamespace(name="code", id=uuid4()),
            "id",
            "source",
            fixture_session(),
        )


@pytest.mark.asyncio
async def test_strict_distillation_surfaces_provider_failure(monkeypatch):
    import importlib

    module = importlib.import_module("cognee.modules.session_distillation.distill")
    monkeypatch.setattr(
        module.LLMGateway,
        "acreate_structured_output",
        AsyncMock(side_effect=RuntimeError("provider unavailable")),
    )
    # The default native caller keeps its existing best-effort behavior.
    assert await module.curate_batch("payment evidence") == []
    with module.strict_distillation():
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await module.curate_batch("payment evidence")
    assert await module.curate_batch("payment evidence") == []


def test_trace_parts_fit_native_serialized_output_limit_without_losing_characters():
    import json
    from cognee.modules.weave.review_sessions import trace_parts

    entry = fixture_session()
    entry.steps[0].content = '\\\n\t"tail' * 200
    parts = list(trace_parts(entry))
    assert all(len(json.dumps(content, ensure_ascii=False)) <= 600 for _, _, content in parts)
    assert (
        "".join(content for key, _, content in parts if key.startswith("tool-1:"))
        == entry.steps[0].content
    )


@pytest.mark.asyncio
async def test_strict_distillation_retries_failed_novelty_search_but_allows_new_collection():
    import importlib
    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    module = importlib.import_module("cognee.modules.session_distillation.distill")
    engine = SimpleNamespace(search=AsyncMock(side_effect=RuntimeError("embedding unavailable")))
    with module.strict_distillation():
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            await module.search_payload_texts(engine, "DocumentChunk_text", 5, query_text="payment")
        engine.search.side_effect = CollectionNotFoundError(message="new dataset")
        assert (
            await module.search_payload_texts(engine, "DocumentChunk_text", 5, query_text="payment")
            == []
        )
