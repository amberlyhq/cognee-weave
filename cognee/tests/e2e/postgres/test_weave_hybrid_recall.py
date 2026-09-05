import os
import zipfile
from uuid import UUID

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("DB_PROVIDER") != "postgres",
    reason="requires the real Postgres Weave control plane",
)


def _archive(tmp_path, name: str, value: str):
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/go.mod", f"module github.com/amberlyhq/{name}\n\ngo 1.24\n")
        archive.writestr(
            f"{name}/main.go",
            "package main\n\n"
            f'func Message() string {{ return "{value}" }}\n\n'
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
        pipeline_version="weave-recall-e2e-v1",
        extraction_version="enola-0.3.13",
    )


@pytest.mark.asyncio
async def test_native_recall_routes_repository_datasets_but_never_organizations(
    tmp_path, offline_native_recall
):
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.recall import recall
    from cognee.modules.weave.native_memory import repository_dataset

    organization_a = UUID("8e1a7b9d-08c2-4f57-9884-623e01b68a01")
    organization_b = UUID("8e1a7b9d-08c2-4f57-9884-623e01b68a02")
    binding_a = await provision_organization(organization_a)
    binding_b = await provision_organization(organization_b)

    inputs = (
        (_request(organization_a, 930001, "recall-alpha", "d" * 40), "alpha"),
        (_request(organization_a, 930002, "recall-beta", "e" * 40), "beta"),
        (_request(organization_b, 930003, "recall-canary", "f" * 40), "canary"),
    )
    for request, value in inputs:
        await index_repository_archive(
            request,
            _archive(tmp_path, request.repository_name, value),
        )

    response_a = await recall(
        organization_a,
        RecallRequest(query="Message", top_k=25, deadline_ms=15000),
    )
    response_b = await recall(
        organization_b,
        RecallRequest(query="Message", top_k=25, deadline_ms=15000),
    )

    assert response_a.status == "available"
    assert {item.github_repository_id for item in response_a.repositories} == {930001, 930002}
    assert not response_a.graph_candidates and not response_a.vector_candidates
    assert response_a.native_memory
    a_ids = {(await repository_dataset(binding_a, item)).id for item in (930001, 930002)}
    b_id = (await repository_dataset(binding_b, 930003)).id
    assert set(offline_native_recall[0][0]) == a_ids
    assert offline_native_recall[0][1] == binding_a.service_user_id
    assert b_id not in a_ids
    assert response_b.status == "available"
    assert {item.github_repository_id for item in response_b.repositories} == {930003}
    assert offline_native_recall[1] == ([b_id], binding_b.service_user_id)
