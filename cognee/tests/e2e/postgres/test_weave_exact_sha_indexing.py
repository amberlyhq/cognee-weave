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
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.native_memory import repository_dataset, NATIVE_PIPELINE_VERSION
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.deletion import export_organization, delete_organization

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
        dataset = await repository_dataset(binding, repo)
        assert dataset.id not in seen
        seen.append(dataset.id)
        async with scoped_database_context_variables(dataset.id, binding.service_user_id):
            graph = await get_graph_engine()
            nodes, edges = await graph.get_graph_data()
            assert nodes and edges
            assert any("Message" in properties.get("name", "") for _, properties in nodes)
        surface = await export_organization(binding.organization_id, [repo])
        assert surface.repositories[0].indexed_default_sha == sha
        assert first.request.pipeline_version == NATIVE_PIPELINE_VERSION
        assert surface.native_graph
    assert await repository_dataset(a, 920003) is None
    assert await repository_dataset(b, 920001) is None
    await delete_organization(a.organization_id, 1)
    await delete_organization(b.organization_id, 1)
