import os
import zipfile
from uuid import UUID

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("DB_PROVIDER") != "postgres",
    reason="requires the real Postgres Weave control plane",
)


def _repository_archive(tmp_path, name: str, message: str):
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{name}/go.mod",
            f"module github.com/amberlyhq/{name}\n\ngo 1.24\n",
        )
        archive.writestr(
            f"{name}/main.go",
            "package main\n\n"
            f'func Message() string {{ return "{message}" }}\n\n'
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
        pipeline_version="weave-e2e-v3",
        extraction_version="enola-0.3.13",
    )


@pytest.mark.asyncio
async def test_exact_sha_indexing_keeps_two_repositories_and_a_tenant_canary_isolated(tmp_path):
    from uuid import uuid4

    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.weave.deletion import delete_organization, export_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.native_memory import NATIVE_PIPELINE_VERSION, customer_dataset
    from cognee.modules.weave.organizations import provision_organization

    a, b = await provision_organization(uuid4()), await provision_organization(uuid4())
    seen = []
    for binding, repo, name, sha in (
        (a, 920001, "alpha", "a" * 40),
        (a, 920002, "beta", "b" * 40),
        (b, 920003, "canary", "c" * 40),
    ):
        request = _request(binding.organization_id, repo, name, sha)
        archive = _repository_archive(tmp_path, name, name)
        first = await index_repository_archive(request, archive)
        duplicate = await index_repository_archive(request, archive)
        assert first.id == duplicate.id
        dataset = await customer_dataset(binding)
        assert dataset.id == binding.dataset_id
        seen.append(dataset.id)
        async with scoped_database_context_variables(dataset.id, binding.service_user_id):
            graph = await get_graph_engine()
            nodes, edges = await graph.get_graph_data()
            assert nodes and edges
            assert any("Message" in properties.get("name", "") for _, properties in nodes)
        surface = await export_organization(binding.organization_id, [repo])
        assert (
            next(
                item for item in surface.repositories if item.github_repository_id == repo
            ).indexed_default_sha
            == sha
        )
        assert first.request.pipeline_version == NATIVE_PIPELINE_VERSION
        assert surface.native_graph
    assert seen[0] == seen[1] and seen[2] != seen[0]
    assert not await source_records(a, 920003)
    assert not await source_records(b, 920001)
    await delete_organization(a.organization_id, 1)
    await delete_organization(b.organization_id, 1)


@pytest.mark.asyncio
async def test_remember_improvement_failure_is_recorded_and_retried(tmp_path, monkeypatch):
    from uuid import uuid4

    from cognee.memify_pipelines import memify_default_tasks
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.native_memory import customer_snapshots, snapshots_ready
    from cognee.modules.weave.organizations import provision_organization

    binding = await provision_organization(uuid4())
    request = _request(binding.organization_id, 920010, "improvement-retry", "d" * 40)
    archive = _repository_archive(tmp_path, "improvement-retry", "retry")

    async def failing_improvement(data_points, **kwargs):
        raise RuntimeError("Injected improvement embedding failure")

    with monkeypatch.context() as patch:
        # The default native pipeline logs an error and then rethrows it.
        # Do not mock remember or alter its error-handling behavior.
        patch.setattr(memify_default_tasks, "index_data_points", failing_improvement)
        with pytest.raises(RuntimeError, match="Injected improvement embedding failure"):
            await index_repository_archive(request, archive)
    assert not snapshots_ready(await customer_snapshots(binding.organization_id))
    assert {r.status for r in await source_records(binding)} == {"processing_remember"}
    import cognee

    improved = []
    native_improve = cognee.improve

    async def observed_improve(*args, **kwargs):
        improved.append(kwargs["dataset"])
        return await native_improve(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cognee, "improve", observed_improve)
        await index_repository_archive(request, archive)
    assert improved == [binding.dataset_id]  # retry finishes the interrupted native improvement
    assert snapshots_ready(await customer_snapshots(binding.organization_id))
    assert {r.status for r in await source_records(binding)} == {"completed"}
    await delete_organization(binding.organization_id, 1)
