import asyncio
from uuid import uuid4
import pytest
from cognee.tests.e2e.postgres.test_pgvector_hnsw_plan import (
    _EmbeddingEngine,
    _database_url,
    _create_schema,
    _drop_schema,
)
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


@pytest.mark.asyncio
async def test_concurrent_reflection_does_not_publish_partial_tables():
    schema = "race_" + uuid4().hex
    await _create_schema(schema)
    adapter = PGVectorAdapter(_database_url(), "", _EmbeddingEngine(), schema=schema)
    try:

        async def lookup():
            try:
                table = await adapter.get_table("MissingCollection")
                return list(table.c.keys())
            except CollectionNotFoundError:
                return "missing"

        results = await asyncio.gather(*(lookup() for _ in range(16)))
        assert results == ["missing"] * 16
        await adapter.create_collection("PresentCollection")
        adapter._metadata.clear()
        tables = await asyncio.gather(*(adapter.get_table("PresentCollection") for _ in range(16)))
        assert all(set(t.c.keys()) == {"id", "payload", "vector"} for t in tables)
    finally:
        await adapter.close()
        await _drop_schema(schema)


@pytest.mark.asyncio
async def test_cancelled_reflection_does_not_poison_waiting_reader(monkeypatch):
    from sqlalchemy import Table
    from sqlalchemy.util.concurrency import await_only

    schema = "cancel_" + uuid4().hex
    await _create_schema(schema)
    adapter = PGVectorAdapter(_database_url(), "", _EmbeddingEngine(), schema=schema)
    started = asyncio.Event()
    blocker = asyncio.Event()
    original_autoload = Table._autoload
    pause_next = True

    def paused_autoload(table, *args, **kwargs):
        nonlocal pause_next
        if table.name == "PresentCollection" and pause_next:
            pause_next = False
            started.set()
            await_only(blocker.wait())
        return original_autoload(table, *args, **kwargs)

    first = second = None
    try:
        await adapter.create_collection("PresentCollection")
        adapter._metadata.clear()
        monkeypatch.setattr(Table, "_autoload", paused_autoload)
        first = asyncio.create_task(adapter.get_table("PresentCollection"))
        await asyncio.wait_for(started.wait(), timeout=10)
        second = asyncio.create_task(adapter.get_table("PresentCollection"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        table = await asyncio.wait_for(second, timeout=10)
        assert set(table.c.keys()) == {"id", "payload", "vector"}
    finally:
        for task in (first, second):
            if task and not task.done():
                task.cancel()
        await adapter.close()
        await _drop_schema(schema)
