import re

import pytest

from cognee.infrastructure.databases.vector.pgvector.indexes import (
    ensure_hnsw_cosine_index,
    hnsw_index_name,
)


def test_hnsw_index_name_is_deterministic_bounded_and_identifier_safe():
    collection = 'Tenant Notes/with spaces and "quotes"' * 4

    first = hnsw_index_name(collection)
    second = hnsw_index_name(collection)

    assert first == second
    assert len(first) <= 63
    assert re.fullmatch(r"[a-z_][a-z0-9_]*", first)


def test_hnsw_index_names_do_not_collide_for_similar_long_collections():
    prefix = "a" * 100

    assert hnsw_index_name(prefix + "first") != hnsw_index_name(prefix + "second")


@pytest.mark.asyncio
async def test_hnsw_index_is_skipped_above_pgvector_dimension_limit():
    class EngineThatMustNotBeUsed:
        def begin(self):
            raise AssertionError("an unsupported index must not open a database connection")

    assert (
        await ensure_hnsw_cosine_index(
            EngineThatMustNotBeUsed(),
            "large_embeddings",
            vector_size=2001,
        )
        is None
    )
