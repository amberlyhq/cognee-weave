"""Concurrent acceptance tests for schema-per-dataset Postgres isolation."""

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    dataset_schema_name,
    drop_pg_schema_if_exists,
)
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    IndexSchema,
    PGVectorAdapter,
)


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
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((sum(text.encode()) + offset) % 101) / 101 for offset in range(8)]
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


async def _create_schema(schema: str, *, vector: bool = False) -> None:
    config = _database_config()
    await create_pg_schema_if_not_exists(
        config["name"],
        schema,
        host=config["host"],
        port=config["port"],
        username=config["username"],
        password=config["password"],
        with_vector_extension=vector,
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


@pytest.mark.asyncio
async def test_concurrent_vector_and_graph_operations_remain_in_tenant_schema():
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable for shared-schema concurrency test")

    schemas = [dataset_schema_name(uuid4()), dataset_schema_name(uuid4())]
    for schema in schemas:
        await _create_schema(schema, vector=True)

    vectors = [PGVectorAdapter(_database_url(), "", _EmbeddingEngine(), schema=s) for s in schemas]
    graphs = [PostgresDemoAdapter(_database_url(), schema=s) for s in schemas]

    try:
        await asyncio.gather(*(graph.initialize() for graph in graphs))

        async def write_tenant(tenant: int) -> None:
            for number in range(12):
                label = f"tenant-{tenant}-item-{number}"
                await asyncio.gather(
                    vectors[tenant].create_data_points(
                        "shared_collection", [IndexSchema(id=uuid4(), text=label)]
                    ),
                    graphs[tenant].add_node(
                        label, properties={"name": label, "type": "TenantCanary"}
                    ),
                )

        await asyncio.gather(write_tenant(0), write_tenant(1))

        async def assert_tenant(tenant: int) -> None:
            expected_prefix = f"tenant-{tenant}-"
            for _ in range(8):
                vector_rows, graph_data = await asyncio.gather(
                    vectors[tenant].search(
                        "shared_collection", query_text=expected_prefix, limit=100
                    ),
                    graphs[tenant].get_graph_data(),
                )
                nodes, edges = graph_data
                assert len(vector_rows) == 12
                assert len(nodes) == 12
                assert not edges
                assert all(node[0].startswith(expected_prefix) for node in nodes)

        await asyncio.gather(assert_tenant(0), assert_tenant(1))

        for tenant, schema in enumerate(schemas):
            async with vectors[tenant].engine.connect() as connection:
                assert await connection.scalar(text("SHOW search_path")) == f"{schema}, public"
            async with graphs[tenant].engine.connect() as connection:
                assert await connection.scalar(text("SHOW search_path")) == schema

        await _drop_schema(schemas[0])
        remaining_rows = await vectors[1].search(
            "shared_collection", query_text="tenant-1", limit=100
        )
        remaining_nodes, _ = await graphs[1].get_graph_data()
        assert len(remaining_rows) == 12
        assert len(remaining_nodes) == 12
    finally:
        await asyncio.gather(
            *(adapter.close() for adapter in [*vectors, *graphs]), return_exceptions=True
        )
        await asyncio.gather(*(_drop_schema(schema) for schema in schemas), return_exceptions=True)
