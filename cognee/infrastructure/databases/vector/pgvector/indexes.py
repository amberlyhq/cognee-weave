"""Schema-safe pgvector index lifecycle helpers."""

import hashlib
import re

from sqlalchemy import Index, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.schema import CreateIndex

MAX_HNSW_VECTOR_DIMENSIONS = 2000


def hnsw_index_name(collection_name: str) -> str:
    """Return a stable Postgres-safe HNSW index name for a collection."""
    digest = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9_]+", "_", collection_name.lower()).strip("_") or "collection"
    suffix = f"_vector_hnsw_{digest}"
    prefix_length = 63 - len("ix_") - len(suffix)
    return f"ix_{slug[:prefix_length]}{suffix}"


async def ensure_hnsw_cosine_index(
    engine: AsyncEngine,
    collection_name: str,
    *,
    vector_size: int,
    schema: str | None = None,
) -> str | None:
    """Create the collection's cosine HNSW index if it is missing.

    SQLAlchemy owns all identifier quoting. ``IF NOT EXISTS`` also makes two
    workers repairing the same collection safe.
    """
    if vector_size > MAX_HNSW_VECTOR_DIMENSIONS:
        return None

    index_name = hnsw_index_name(collection_name)

    async with engine.begin() as connection:

        def create_index(sync_connection) -> None:
            table = Table(
                collection_name,
                MetaData(),
                schema=schema or None,
                autoload_with=sync_connection,
            )
            index = Index(
                index_name,
                table.c.vector,
                postgresql_using="hnsw",
                postgresql_ops={"vector": "vector_cosine_ops"},
            )
            sync_connection.execute(CreateIndex(index, if_not_exists=True))

        await connection.run_sync(create_index)

    return index_name
