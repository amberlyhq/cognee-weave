import os
import random
import zipfile
from uuid import uuid4

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    os.getenv("DB_PROVIDER") != "postgres",
    reason="requires the real Postgres Weave control plane",
)


def _archive(tmp_path, name: str, marker: str):
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/go.mod", f"module github.com/amberlyhq/{name}\n\ngo 1.24\n")
        archive.writestr(
            f"{name}/main.go",
            "package main\n\n"
            f'func Message() string {{ return "{marker}" }}\n\n'
            "func main() { println(Message()) }\n",
        )
    return path


def _request(organization_id, repository_id, name, sha):
    from cognee.modules.weave.indexing import IndexRequest

    return IndexRequest(
        organization_id=organization_id,
        github_repository_id=repository_id,
        repository_owner="amberlyhq",
        repository_name=name,
        default_branch="main",
        requested_sha=sha,
        pipeline_version="weave-surface-e2e-v1",
        extraction_version="enola-0.3.13",
    )


def _repository_node_id(request):
    from cognee.tasks.code_graph.extract_code_graph import fact_node_id
    from cognee.tasks.code_graph.models import RepositoryProvenance

    provenance = RepositoryProvenance(
        organization_id=request.organization_id,
        github_repository_id=request.github_repository_id,
        repository_owner=request.repository_owner,
        repository_name=request.repository_name,
        indexed_sha=request.requested_sha,
        pipeline_version=request.pipeline_version,
        extraction_version=request.extraction_version,
    )
    return str(
        fact_node_id(
            provenance.repository_identity,
            "repository",
            provenance.repository_identity,
        )
    )


@pytest.mark.asyncio
async def test_every_surface_stays_scoped_through_repository_and_organization_deletion(
    tmp_path, offline_native_recall, monkeypatch
):
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.graph.config import get_graph_context_config
    from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
    from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.deletion import (
        SurfaceNotFound,
        activate_repository,
        delete_organization,
        delete_repository,
        export_organization,
        visualize_organization,
    )
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.native_memory import (
        customer_dataset,
        repository_dataset,
        repository_dataset_name,
    )
    from cognee.modules.weave.organizations import (
        get_organization_binding,
        provision_organization,
    )
    from cognee.modules.weave.recall import recall

    organization_a = uuid4()
    organization_b = uuid4()
    binding_a = await provision_organization(organization_a)
    binding_b = await provision_organization(organization_b)
    inputs = (
        (_request(organization_a, 940001, "surface-alpha", "1" * 40), "SURFACE_ALPHA"),
        (_request(organization_a, 940002, "surface-beta", "2" * 40), "SURFACE_BETA"),
        (_request(organization_b, 940003, "surface-canary", "3" * 40), "SURFACE_CANARY"),
    )
    for request, marker in inputs:
        await index_repository_archive(
            request,
            _archive(tmp_path, request.repository_name, marker),
        )

    foreign_recall = await recall(
        organization_a,
        RecallRequest(query="Message", github_repository_ids=[940003]),
    )
    absent_recall = await recall(
        organization_a,
        RecallRequest(query="Message", github_repository_ids=[949999]),
    )
    assert foreign_recall == absent_recall

    operations = [export_organization, visualize_organization] * 3
    random.Random(20260831).shuffle(operations)
    for operation in operations:
        surface = await operation(organization_a)
        serialized = surface.model_dump_json()
        assert {item.github_repository_id for item in surface.repositories} == {940001, 940002}
        assert "SURFACE_CANARY" not in serialized
        assert "940003" not in serialized
        assert surface.native_graph

    for repository_id in (940003, 949999):
        with pytest.raises(SurfaceNotFound, match="Resource not found"):
            await export_organization(organization_a, [repository_id])
        with pytest.raises(SurfaceNotFound, match="Resource not found"):
            await delete_repository(organization_a, repository_id, 1)

    # Explicit cleanup removes the dedicated native repository dataset.
    from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
    from cognee.modules.users.methods import get_user

    await create_authorized_dataset(
        repository_dataset_name(organization_a, 940001), await get_user(binding_a.service_user_id)
    )
    assert await repository_dataset(binding_a, 940001) is not None
    from cognee.modules.weave import deletion
    from cognee.modules.weave.indexing import RepositoryDeletedError

    async def fail_publication(*args, **kwargs):
        raise RuntimeError("Injected failure after physical cleanup")

    with monkeypatch.context() as patch:
        patch.setattr(deletion, "_publish_repository_transition", fail_publication)
        with pytest.raises(RuntimeError, match="after physical cleanup"):
            await delete_repository(organization_a, 940001, 2)
    assert not await source_records(binding_a, 940001)
    assert (await recall(organization_a, RecallRequest(query="Message"))).status == "unavailable"
    with pytest.raises(SurfaceNotFound):
        await export_organization(organization_a, [940002])
    with pytest.raises(RuntimeError, match="cleanup is still pending"):
        await activate_repository(organization_a, 940001, 3)
    with pytest.raises(RepositoryDeletedError):
        await index_repository_archive(inputs[0][0], _archive(tmp_path, "surface-alpha", "retry"))
    await delete_repository(organization_a, 940001, 2)
    remaining_a = await recall(organization_a, RecallRequest(query="Message"))
    untouched_b = await recall(organization_b, RecallRequest(query="Message"))
    assert {item.github_repository_id for item in remaining_a.repositories} == {940002}
    assert {item.github_repository_id for item in untouched_b.repositories} == {940003}
    assert await repository_dataset(binding_a, 940001) is None
    assert await customer_dataset(binding_a) is not None
    assert await customer_dataset(binding_b) is not None
    assert not await source_records(binding_a, 940001)
    assert await customer_dataset(binding_a)
    assert await customer_dataset(binding_b)

    # Index delivery alone cannot revive a removed repository. Only the
    # separately verified GitHub installation lifecycle may reactivate it.
    alpha_request, alpha_marker = inputs[0]
    from cognee.modules.weave.indexing import RepositoryDeletedError

    with pytest.raises(RepositoryDeletedError):
        await index_repository_archive(
            alpha_request,
            _archive(tmp_path, alpha_request.repository_name, alpha_marker),
        )
    await activate_repository(organization_a, alpha_request.github_repository_id, 1)
    still_removed = await recall(organization_a, RecallRequest(query="Message"))
    assert {item.github_repository_id for item in still_removed.repositories} == {940002}
    await activate_repository(organization_a, alpha_request.github_repository_id, 3)
    object.__setattr__(alpha_request, "lifecycle_generation", 3)
    await index_repository_archive(
        alpha_request,
        _archive(tmp_path, alpha_request.repository_name, alpha_marker),
    )
    restored_a = await recall(organization_a, RecallRequest(query="Message"))
    assert {item.github_repository_id for item in restored_a.repositories} == {940001, 940002}
    assert (await recall(organization_b, RecallRequest(query="Message"))).status == "available"

    async with scoped_database_context_variables(
        binding_a.dataset_id,
        binding_a.service_user_id,
    ):
        await get_graph_engine()
        (await get_vector_engine_async())._engine()
        graph_config_a = get_graph_context_config()
        vector_config_a = get_vectordb_context_config()
    async with scoped_database_context_variables(
        binding_b.dataset_id,
        binding_b.service_user_id,
    ):
        await get_graph_engine()
        (await get_vector_engine_async())._engine()
        graph_config_b = get_graph_context_config()
        vector_config_b = get_vectordb_context_config()
    assert graph_engine_cache.is_cached(**graph_config_a)
    assert vector_engine_cache.is_cached(**vector_config_a)
    assert graph_engine_cache.is_cached(**graph_config_b)
    assert vector_engine_cache.is_cached(**vector_config_b)

    # An installation deletion governs organization existence even when a
    # newer repository event was observed first.
    from cognee.modules.weave.organizations import (
        OrganizationDeletedError,
        reactivate_organization,
    )

    async def fail_organization_publication(*args, **kwargs):
        raise RuntimeError("Injected organization tombstone failure")

    with monkeypatch.context() as fault:
        fault.setattr(deletion, "_mark_organization_deleted", fail_organization_publication)
        with pytest.raises(RuntimeError, match="Injected organization tombstone failure"):
            await delete_organization(organization_a, 2)
    assert await get_organization_binding(organization_a) is None
    assert (await recall(organization_a, RecallRequest(query="Message"))).status == "unavailable"
    with pytest.raises(OrganizationDeletedError):
        await provision_organization(organization_a)
    with pytest.raises(OrganizationDeletedError):
        await reactivate_organization(organization_a, 4)

    await delete_organization(organization_a, 2)
    assert not graph_engine_cache.is_cached(**graph_config_a)
    assert not vector_engine_cache.is_cached(**vector_config_a)
    assert graph_engine_cache.is_cached(**graph_config_b)
    assert vector_engine_cache.is_cached(**vector_config_b)
    assert await get_organization_binding(organization_a) is None
    assert (await recall(organization_a, RecallRequest(query="Message"))).status == "unavailable"
    assert (await recall(organization_b, RecallRequest(query="Message"))).status == "available"

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        schema_exists = await session.scalar(
            text("SELECT to_regnamespace(:schema) IS NOT NULL"),
            {"schema": binding_a.graph_schema},
        )
    assert schema_exists is False

    # Ordinary traffic cannot recreate a tombstoned organization. Only a newer
    # verified installation.created lifecycle generation can do that.
    with pytest.raises(OrganizationDeletedError):
        await provision_organization(organization_a)
    assert await reactivate_organization(organization_a, 1) is None
    reprovisioned_a = await reactivate_organization(organization_a, 4)
    assert reprovisioned_a is not None
    assert reprovisioned_a.dataset_id == binding_a.dataset_id
    # A repository event older than the organization reactivation cannot cross
    # the organization epoch barrier, even if it is newer than repository state.
    await activate_repository(organization_a, alpha_request.github_repository_id, 3)
    object.__setattr__(alpha_request, "lifecycle_generation", 3)
    with pytest.raises(RepositoryDeletedError):
        await index_repository_archive(
            alpha_request,
            _archive(tmp_path, alpha_request.repository_name, alpha_marker),
        )
    await activate_repository(organization_a, alpha_request.github_repository_id, 4)
    object.__setattr__(alpha_request, "lifecycle_generation", 4)
    await index_repository_archive(
        alpha_request,
        _archive(tmp_path, alpha_request.repository_name, alpha_marker),
    )
    async with engine.get_async_session() as session:
        recreated_schema_exists = await session.scalar(
            text("SELECT to_regnamespace(:schema) IS NOT NULL"),
            {"schema": binding_a.graph_schema},
        )
    assert recreated_schema_exists is True
