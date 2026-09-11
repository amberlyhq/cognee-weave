"""Native source selection wiring, included in the maintained Weave unit gate."""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_native_update_cognifies_only_the_resolved_source(monkeypatch):
    module = importlib.import_module("cognee.api.v1.update.update")
    pinned_id, legacy_id, dataset_id = uuid4(), uuid4(), uuid4()
    from cognee.modules.data import methods
    from cognee.infrastructure.databases import relational

    @asynccontextmanager
    async def session():
        yield SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(legacy_id=None)))

    monkeypatch.setattr(methods, "resolve_data_id", AsyncMock(return_value=pinned_id))
    monkeypatch.setattr(
        relational, "get_relational_engine", lambda: SimpleNamespace(get_async_session=session)
    )
    monkeypatch.setattr(module.datasets, "delete_data", AsyncMock())
    add, cognify = AsyncMock(), AsyncMock(return_value={"completed": True})
    monkeypatch.setattr(module, "add", add)
    monkeypatch.setattr(module, "cognify", cognify)
    result = await module.update(
        legacy_id, "A source replacement.", dataset_id, user=SimpleNamespace(id=uuid4())
    )
    assert result == {"completed": True}
    assert add.call_args.kwargs["data"].data_id == pinned_id
    assert cognify.call_args.kwargs["data_ids"] == [pinned_id]
    assert cognify.call_args.kwargs["datasets"] == [dataset_id]


@pytest.mark.asyncio
async def test_native_remember_routes_source_selection_to_cognify_only(monkeypatch):
    module = importlib.import_module("cognee.api.v1.remember.remember")
    add_pkg = importlib.import_module("cognee.api.v1.add")
    cognify_pkg = importlib.import_module("cognee.api.v1.cognify")
    ids, dataset = [uuid4()], uuid4()
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", AsyncMock())
    monkeypatch.setattr("cognee.modules.migrations.startup.run_migrations_and_block", AsyncMock())
    add, cognify = AsyncMock(), AsyncMock(return_value={})
    monkeypatch.setattr(add_pkg, "add", add)
    monkeypatch.setattr(cognify_pkg, "cognify", cognify)
    await module.remember(
        "A single source.",
        dataset_id=dataset,
        data_ids=ids,
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
    )
    assert "data_ids" not in add.call_args.kwargs
    assert cognify.call_args.kwargs["data_ids"] == ids


@pytest.mark.asyncio
async def test_scoped_remote_remember_cannot_silently_drop_source_selection(monkeypatch):
    module = importlib.import_module("cognee.api.v1.remember.remember")
    client = SimpleNamespace(remember=AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: client
    )
    with pytest.raises(ValueError, match="data_ids"):
        await module._remember_inner(
            "note",
            "dataset",
            dataset_id=uuid4(),
            session_id=None,
            chunk_size=None,
            chunker=None,
            custom_prompt=None,
            run_in_background=False,
            self_improvement=False,
            session_ids=None,
            span=SimpleNamespace(set_attribute=Mock()),
            data_ids=[uuid4()],
        )
    client.remember.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], (), "not-a-list"])
async def test_empty_or_malformed_scope_never_loads_a_dataset(ids):
    from cognee.api.v1.cognify.cognify import _resolve_data_scope

    with pytest.raises(ValueError, match="nonempty list"):
        await _resolve_data_scope([uuid4()], SimpleNamespace(id=uuid4()), ids)
