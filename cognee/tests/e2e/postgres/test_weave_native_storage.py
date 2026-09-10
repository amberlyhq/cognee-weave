"""Native forget must retain the strict tenant boundary on shared Postgres."""

from uuid import uuid4
import json
import os

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_native_forget_removes_only_the_bound_repository_dataset():
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.native_memory import (
        forget_repository,
        repository_dataset,
        repository_dataset_name,
    )
    from cognee.modules.weave.organizations import (
        provision_organization,
        set_weave_organization_scope,
    )
    from cognee.modules.weave.models import WeaveRepositorySnapshot
    from cognee.modules.weave.native_memory import NATIVE_PIPELINE_VERSION
    from cognee.modules.weave.deletion import visualize_organization, delete_organization

    a, b = await provision_organization(uuid4()), await provision_organization(uuid4())
    datasets = []
    for binding in (a, b):
        user = await get_user(binding.service_user_id)
        dataset = await create_authorized_dataset(
            repository_dataset_name(binding.organization_id, 42), user
        )
        async with scoped_database_context_variables(dataset.id, user.id):
            graph = await get_graph_engine()
            await graph.add_nodes(
                [
                    (str(uuid4()), {"name": "tenant canary", "type": "Entity"}),
                    (str(uuid4()), {"name": "native summary", "type": "TextSummary"}),
                ]
            )
            bounded, _ = await graph.get_filtered_graph_data([], max_nodes=1, max_edges=0)
            assert len(bounded) == 1
        datasets.append(dataset)

    engine = get_relational_engine()
    # Native inspection includes repository datasets and the separate review dataset.
    async with scoped_database_context_variables(a.dataset_id, a.service_user_id):
        graph = await get_graph_engine()
        await graph.add_nodes([(str(uuid4()), {"name": "primary memory", "type": "Entity"})])
    async with engine.get_async_session() as session:
        await set_weave_organization_scope(session, a.organization_id)
        session.add(
            WeaveRepositorySnapshot(
                organization_id=a.organization_id,
                github_repository_id=42,
                repository_owner="test",
                repository_name="native",
                default_branch="main",
                requested_sha="a" * 40,
                indexed_sha="a" * 40,
                pipeline_version=NATIVE_PIPELINE_VERSION,
                extraction_version="native",
                status="indexed",
            )
        )
        await session.commit()
    surface = await visualize_organization(a.organization_id, [42])
    assert not surface.nodes  # Do not pretend native entities are verified symbols.
    native = json.loads(surface.native_graph)
    assert {properties["name"] for _, properties in native[0]["nodes"]} == {
        "tenant canary",
        "native summary",
    }
    assert {properties["name"] for _, properties in native[1]["nodes"]} == {"primary memory"}
    assert native[0]["dataset_id"] == str(datasets[0].id)
    assert native[1]["dataset_id"] == str(a.dataset_id)
    assert surface.repositories[0].indexed_default_sha == "a" * 40
    with pytest.raises(Exception, match="Bound native Weave dataset not found"):
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, b.organization_id)
            await session.execute(
                text("SELECT public.weave_drop_native_dataset_schema(:org, :dataset)"),
                {"org": b.organization_id, "dataset": datasets[0].id},
            )

    await forget_repository(a, 42)
    assert await repository_dataset(a, 42) is None
    assert (await repository_dataset(b, 42)).id == datasets[1].id
    async with scoped_database_context_variables(datasets[1].id, b.service_user_id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_graph_data()
        assert len(nodes) == 2
    await forget_repository(b, 42)
    await delete_organization(a.organization_id, 1)
    await delete_organization(b.organization_id, 1)
