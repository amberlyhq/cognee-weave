"""Real Postgres checks for bounded and tenant-routed graph retrieval."""

import asyncio
from contextvars import ContextVar
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    dataset_schema_name,
    drop_pg_schema_if_exists,
)
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


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


class _EmbeddingEngine:
    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _SeedVectorEngine:
    def __init__(self, seed_id: UUID):
        self.seed_id = seed_id
        self.embedding_engine = _EmbeddingEngine()

    async def search(self, *, collection_name, **_kwargs):
        if collection_name != "Entity_name":
            return []
        return [ScoredResult(id=self.seed_id, score=0.01, payload={"text": "seed"})]


@pytest.mark.asyncio
async def test_postgres_neighborhood_is_deterministic_and_hard_bounded():
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable for graph traversal test")

    schema = dataset_schema_name(uuid4())
    await _create_schema(schema)
    adapter = PostgresDemoAdapter(_database_url(), schema=schema)
    try:
        await adapter.initialize()
        await adapter.add_nodes(
            [
                ("root", {"name": "root", "type": "Entity"}),
                *[
                    (f"child-{number}", {"name": f"child-{number}", "type": "Entity"})
                    for number in range(10)
                ],
            ]
        )
        await adapter.add_edges(
            [
                ("root", f"child-{number}", "ALLOWED", {"order": number})
                for number in range(10)
            ]
            + [("child-0", "child-1", "CYCLE", {}), ("child-1", "root", "CYCLE", {})]
        )

        kwargs = {
            "depth": 3,
            "fan_out": 3,
            "max_nodes": 5,
            "max_edges": 4,
            "statement_timeout_ms": 1000,
        }
        first = await adapter.get_neighborhood(["root"], **kwargs)
        second = await adapter.get_neighborhood(["root"], **kwargs)

        assert first == second
        nodes, edges = first
        assert len(nodes) <= 5
        assert len(edges) <= 4
        assert "root" in {node_id for node_id, _ in nodes}

        filtered_nodes, filtered_edges = await adapter.get_neighborhood(
            ["root"], edge_types=["ALLOWED"], **kwargs
        )
        assert len(filtered_nodes) <= 5
        assert all(edge[2] == "ALLOWED" for edge in filtered_edges)

        with pytest.raises(NotImplementedError, match="does not support raw Cypher"):
            await adapter.query("MATCH (n) RETURN n")
    finally:
        await adapter.close()
        await _drop_schema(schema)


@pytest.mark.asyncio
async def test_concurrent_graph_completion_uses_only_active_dataset_graph(monkeypatch):
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable for tenant graph retrieval test")

    schemas = [dataset_schema_name(uuid4()), dataset_schema_name(uuid4())]
    for schema in schemas:
        await _create_schema(schema)
    adapters = [PostgresDemoAdapter(_database_url(), schema=schema) for schema in schemas]
    seeds = [uuid4(), uuid4()]
    canaries = ["tenant-zero-canary", "tenant-one-canary"]

    try:
        for tenant, adapter in enumerate(adapters):
            await adapter.initialize()
            await adapter.add_nodes(
                [
                    (str(seeds[tenant]), {"name": f"tenant-{tenant}-seed", "type": "Entity"}),
                    (
                        f"canary-{tenant}",
                        {"name": canaries[tenant], "type": "Entity"},
                    ),
                ]
            )
            await adapter.add_edge(str(seeds[tenant]), f"canary-{tenant}", "KNOWS", {})

        active_tenant: ContextVar[int] = ContextVar("active_test_tenant")
        unified_engines = [
            SimpleNamespace(graph=adapters[index], vector=_SeedVectorEngine(seeds[index]))
            for index in range(2)
        ]

        async def resolve_unified_engine():
            return unified_engines[active_tenant.get()]

        async def reject_global_graph():
            raise AssertionError("tenant retrieval attempted to resolve the global graph")

        monkeypatch.setattr(
            "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine",
            resolve_unified_engine,
        )
        monkeypatch.setattr(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
            reject_global_graph,
        )

        retriever = GraphCompletionRetriever(top_k=5, wide_search_top_k=5)

        async def retrieve(tenant: int):
            token = active_tenant.set(tenant)
            try:
                return await retriever.get_retrieved_objects(query=f"tenant {tenant}")
            finally:
                active_tenant.reset(token)

        results = await asyncio.gather(retrieve(0), retrieve(1))
        rendered = [
            " ".join(
                str(value)
                for edge in tenant_edges
                for value in (edge.node1.attributes, edge.attributes, edge.node2.attributes)
            )
            for tenant_edges in results
        ]

        assert canaries[0] in rendered[0]
        assert canaries[1] not in rendered[0]
        assert canaries[1] in rendered[1]
        assert canaries[0] not in rendered[1]
    finally:
        await asyncio.gather(*(adapter.close() for adapter in adapters), return_exceptions=True)
        await asyncio.gather(*(_drop_schema(schema) for schema in schemas), return_exceptions=True)
