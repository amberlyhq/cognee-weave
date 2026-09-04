import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Keep cached async Postgres engines on one loop for the E2E session."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
