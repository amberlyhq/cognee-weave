"""Busy organization requests must not consume the active indexer's pool."""

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_waiters_release_database_connections_until_the_holder_finishes():
    from cognee.modules.weave.indexing import weave_operation_lock
    from cognee.infrastructure.databases.relational import get_relational_engine

    org = uuid4()
    entered = []

    async def waiting(index):
        async with weave_operation_lock(org):
            entered.append(index)

    # Reserve an observer connection before saturating the pool, so the test
    # can diagnose blocking waiters without itself waiting for a connection.
    async with get_relational_engine().get_async_session() as observer:
        baseline = await observer.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND usename=current_user AND wait_event='advisory'"
            )
        )
        tasks = []
        try:
            async with weave_operation_lock(org):
                tasks = [asyncio.create_task(waiting(i)) for i in range(45)]
                await asyncio.sleep(2)
                await observer.execute(text("SELECT pg_stat_clear_snapshot()"))
                blocked = await observer.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND usename=current_user AND wait_event='advisory'"
                    )
                )
                assert blocked == baseline, (
                    "Busy requests are holding pooled connections in pg_advisory_lock"
                )
                assert not entered
                async with get_relational_engine().get_async_session() as active_job:
                    assert await asyncio.wait_for(active_job.scalar(text("SELECT 1")), 2) == 1
            await asyncio.wait_for(asyncio.gather(*tasks), 20)
            assert len(entered) == 45
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
