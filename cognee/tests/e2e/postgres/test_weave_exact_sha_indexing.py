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
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.organizations import provision_organization
    from cognee.tasks.code_graph.extract_code_graph import fact_node_id
    from cognee.tasks.code_graph.models import RepositoryProvenance

    organization_a = UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a01")
    organization_b = UUID("7e1a7b9d-08c2-4f57-9884-623e01b68a02")
    binding_a = await provision_organization(organization_a)
    binding_b = await provision_organization(organization_b)

    requests = (
        _request(organization_a, 920001, "weave-alpha", "a" * 40),
        _request(organization_a, 920002, "weave-beta", "b" * 40),
        _request(organization_b, 920003, "weave-canary", "c" * 40),
    )
    archives = (
        _repository_archive(tmp_path, "weave-alpha", "alpha"),
        _repository_archive(tmp_path, "weave-beta", "beta"),
        _repository_archive(tmp_path, "weave-canary", "canary"),
    )

    for request, archive in zip(requests, archives):
        await index_repository_archive(request, archive)

    def repository_id(request):
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

    alpha_id, beta_id, canary_id = [repository_id(request) for request in requests]

    async with scoped_database_context_variables(binding_a.dataset_id, binding_a.service_user_id):
        graph = await get_graph_engine()
        vector = await get_vector_engine_async()
        alpha = await graph.get_node(alpha_id)
        beta = await graph.get_node(beta_id)
        canary = await graph.get_node(canary_id)
        nodes, _edges = await graph.get_graph_data()
        alpha_vectors = await vector.retrieve("CodeRepository_name", [alpha_id])

    assert alpha is not None
    assert beta is not None
    assert canary is None
    assert alpha["indexed_sha"] == "a" * 40
    assert alpha["source_path"] == "github://amberlyhq/weave-alpha@" + "a" * 40
    assert any(
        properties.get("github_repository_id") in (920001, 920002)
        and properties.get("type") != "CodeRepository"
        for _node_id, properties in nodes
    )
    assert len(alpha_vectors) == 1

    async with scoped_database_context_variables(binding_b.dataset_id, binding_b.service_user_id):
        graph = await get_graph_engine()
        assert await graph.get_node(alpha_id) is None
        assert await graph.get_node(beta_id) is None
        assert await graph.get_node(canary_id) is not None
