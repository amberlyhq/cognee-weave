"""Real Postgres catalog and query-plan checks for schema-local HNSW indexes."""

import json
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    dataset_schema_name,
    drop_pg_schema_if_exists,
)
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    IndexSchema,
    PGVectorAdapter,
)
from cognee.infrastructure.databases.vector.pgvector.indexes import hnsw_index_name


def _database_config() -> dict[str, str]:
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
        "username": os.environ.get("DB_USERNAME", "cognee"),
        "password": os.environ.get("DB_PASSWORD", "cognee"),
        "name": os.environ.get("DB_NAME", "cognee_db"),
    }


def _database_url() -> str:
    config = _database_config()
    return (
        "postgresql+asyncpg://"
        f"{config['username']}:{config['password']}@{config['host']}:"
        f"{config['port']}/{config['name']}"
    )


class _EmbeddingEngine:
    def get_vector_size(self) -> int:
        return 8

    def get_batch_size(self) -> int:
        return 1000

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((sum(text.encode()) * (offset + 1)) % 997) / 997 for offset in range(8)]
            for text in texts
        ]


async def _postgres_reachable() -> bool:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


async def _create_schema(schema: str) -> None:
    config = _database_config()
    await create_pg_schema_if_not_exists(
        config["name"],
        schema,
        host=config["host"],
        port=config["port"],
        username=config["username"],
        password=config["password"],
        with_vector_extension=True,
    )


async def _drop_schema(schema: str) -> None:
    config = _database_config()
    await drop_pg_schema_if_exists(
        config["name"],
        schema,
        host=config["host"],
        port=config["port"],
        username=config["username"],
        password=config["password"],
    )


def _plan_nodes(plan: dict):
    yield plan
    for child in plan.get("Plans", []):
        yield from _plan_nodes(child)


@pytest.mark.asyncio
async def test_same_collection_gets_schema_local_hnsw_index_used_by_cosine_plan():
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable for HNSW query-plan test")

    collection = "PlanVectors"
    schemas = [dataset_schema_name(uuid4()), dataset_schema_name(uuid4())]
    for schema in schemas:
        await _create_schema(schema)

    embedding_engine = _EmbeddingEngine()
    adapters = [
        PGVectorAdapter(_database_url(), "", embedding_engine, schema=schema)
        for schema in schemas
    ]
    try:
        for tenant, adapter in enumerate(adapters):
            await adapter.create_data_points(
                collection,
                [
                    IndexSchema(id=uuid4(), text=f"tenant-{tenant}-vector-{number}")
                    for number in range(256)
                ],
            )

        engine = create_async_engine(_database_url())
        try:
            async with engine.connect() as connection:
                catalog_rows = await connection.execute(
                    text(
                        """
                        SELECT namespace.nspname, index_class.relname, access_method.amname,
                               pg_get_indexdef(index_class.oid)
                        FROM pg_class AS index_class
                        JOIN pg_namespace AS namespace
                          ON namespace.oid = index_class.relnamespace
                        JOIN pg_am AS access_method
                          ON access_method.oid = index_class.relam
                        WHERE index_class.relkind = 'i'
                          AND namespace.nspname = ANY(:schemas)
                          AND index_class.relname = :index_name
                        ORDER BY namespace.nspname
                        """
                    ),
                    {"schemas": schemas, "index_name": hnsw_index_name(collection)},
                )
                catalog = catalog_rows.fetchall()

            assert [row[0] for row in catalog] == sorted(schemas)
            assert all(row[2] == "hnsw" for row in catalog)
            assert all("vector_cosine_ops" in row[3] for row in catalog)

            query_vector = (await embedding_engine.embed_text(["tenant-0-vector-9"]))[0]
            for schema in schemas:
                async with engine.begin() as connection:
                    await connection.execute(text(f'ANALYZE "{schema}"."{collection}"'))
                    await connection.execute(text("SET LOCAL enable_seqscan = off"))
                    result = await connection.execute(
                        text(
                            f'EXPLAIN (FORMAT JSON, VERBOSE) SELECT id FROM "{schema}".'
                            f'"{collection}" ORDER BY vector <=> CAST(:query AS vector) LIMIT 10'
                        ),
                        {"query": str(query_vector)},
                    )
                    raw_plan = result.scalar_one()

                document = json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
                nodes = list(_plan_nodes(document[0]["Plan"]))
                index_scans = [node for node in nodes if node.get("Node Type") == "Index Scan"]
                assert len(index_scans) == 1
                assert index_scans[0]["Index Name"] == hnsw_index_name(collection)
                assert index_scans[0]["Schema"] == schema
        finally:
            await engine.dispose()
    finally:
        for adapter in adapters:
            await adapter.close()
        for schema in schemas:
            await _drop_schema(schema)
