import asyncio

import pytest


@pytest.fixture
def offline_native_recall(monkeypatch):
    """Stub the paid public operation, not Weave's dataset routing or storage.

    Native answer quality is covered by the separately authorized live pilot.
    These offline tests prove the arguments crossing the tenant boundary.
    """
    import cognee

    calls = []

    async def recall(query, *, dataset_ids, user, **kwargs):
        calls.append((list(dataset_ids), user.id))
        return [{"offline_dataset_ids": [str(item) for item in dataset_ids]}]

    monkeypatch.setattr(cognee, "recall", recall)
    return calls


@pytest.fixture(scope="session")
def event_loop():
    """Keep cached async Postgres engines on one loop for the E2E session."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
