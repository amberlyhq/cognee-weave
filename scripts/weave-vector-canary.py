"""Verify real runtime-role vector storage in a disposable parity customer."""

import asyncio
import sys
from uuid import UUID, uuid5

from sqlalchemy import text

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import IndexSchema
from cognee.infrastructure.databases.vector.pgvector.indexes import hnsw_index_name
from cognee.modules.weave.config import get_weave_embedding_config
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.organizations import get_organization_binding


async def verify_vector_storage(organization_id, *, read_only=False):
    async with weave_operation_lock(organization_id):
        binding = await get_organization_binding(organization_id)
        if binding is None:
            raise RuntimeError("Parity customer is not provisioned")
        async with scoped_database_context_variables(
            binding.dataset_id,
            binding.service_user_id,
            embedding_config=get_weave_embedding_config(),
        ):
            vector = await get_vector_engine_async()
            assert vector.embedding_engine.get_vector_size() == 1536
            collection = "WeaveParityVector"
            point_id = uuid5(organization_id, "weave-parity-vector-canary")
            if not read_only:
                await vector.create_data_points(
                    collection, [IndexSchema(id=point_id, text="Weave vector storage canary")]
                )
            found = await vector.search(
                collection, query_text="Weave vector storage canary", limit=1
            )
            assert found and str(found[0].id) == str(point_id)
            async with vector.get_async_session() as session:
                role = (
                    await session.execute(
                        text(
                            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
                        )
                    )
                ).one()
                assert not role.rolsuper and not role.rolbypassrls
                count = await session.scalar(
                    text(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname = :schema "
                        "AND indexname = :index AND indexdef ILIKE '%using hnsw%'"
                    ),
                    {"schema": binding.vector_schema, "index": hnsw_index_name(collection)},
                )
                assert count == 1
    print("runtime vector search and HNSW catalog verified")


if __name__ == "__main__":
    asyncio.run(verify_vector_storage(UUID(sys.argv[1]), read_only="--read-only" in sys.argv[2:]))
