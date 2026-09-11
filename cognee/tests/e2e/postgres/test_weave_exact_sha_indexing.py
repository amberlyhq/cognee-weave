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
            f'func Message{message.title()}() string {{ return "{message}" }}\n\n'
            f"func main() {{ println(Message{message.title()}()) }}\n",
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
            assert any(
                p.get("type") == "CodeRepository"
                and str(p.get("organization_id")) == str(binding.organization_id)
                and p.get("github_repository_id") == repo
                for _, p in nodes
            )
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
async def test_code_import_never_calls_memory_or_embedding_models(tmp_path, monkeypatch):
    from uuid import uuid4
    import cognee
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.native_memory import (
        customer_snapshots,
        snapshots_ready,
        recall_repository_memory,
    )
    from cognee.modules.weave.contracts import RecallRequest
    from cognee.modules.weave.organizations import provision_organization

    async def forbidden(*args, **kwargs):
        raise AssertionError("Code-only import/recall must not call a model")

    binding = await provision_organization(uuid4())
    archive = _repository_archive(tmp_path, "code-only", "payment")
    with zipfile.ZipFile(archive, "a") as contents:
        contents.writestr("code-only/README.md", "Do not enrich this document")
        contents.writestr("code-only/screenshot.png", b"invalid-image-should-never-be-loaded")
    monkeypatch.setattr(cognee, "improve", forbidden)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", forbidden)
    await index_repository_archive(
        _request(binding.organization_id, 920010, "code-only", "d" * 40), archive
    )
    assert snapshots_ready(await customer_snapshots(binding.organization_id))
    result = await recall_repository_memory(
        binding.organization_id, RecallRequest(query="MessagePayment")
    )
    assert result.status == "available"
    assert "MessagePayment" in result.native_memory
    await delete_organization(binding.organization_id, 1)


@pytest.mark.asyncio
async def test_return_to_previous_commit_restores_native_sources(tmp_path):
    from uuid import uuid4

    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.native_memory import customer_snapshots
    from cognee.modules.weave.organizations import provision_organization

    binding = await provision_organization(uuid4())
    repo = 920020
    requests = [_request(binding.organization_id, repo, "returning", sha * 40) for sha in "ab"]
    paths = [tmp_path / label for label in "ab"]
    for path in paths:
        path.mkdir()
    archives = [_repository_archive(path, "returning", label) for path, label in zip(paths, "ab")]

    async def receipt():
        from cognee.context_global_variables import scoped_database_context_variables
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.modules.weave.native_memory import customer_dataset

        dataset = await customer_dataset(binding)
        async with scoped_database_context_variables(dataset.id, binding.service_user_id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        return {p["name"].split(".")[-1] for _, p in nodes if p.get("type") == "CodeSymbol"}

    try:
        first = await index_repository_archive(requests[0], archives[0])
        sources_a = await receipt()
        assert sources_a
        await index_repository_archive(requests[1], archives[1])
        sources_b = await receipt()
        assert sources_a != sources_b
        assert "MessageA" in sources_a and "MessageA" not in sources_b
        assert "MessageB" in sources_b and "MessageB" not in sources_a
        restored = await index_repository_archive(requests[0], archives[0])
        assert await receipt() == sources_a
        snapshot = (await customer_snapshots(binding.organization_id))[0]
        assert snapshot.requested_sha == requests[0].requested_sha
        assert snapshot.indexed_sha == requests[0].requested_sha
        assert snapshot.status == "indexed"
        assert restored.id == first.id
        assert restored.attempt_count == first.attempt_count + 1
        duplicate = await index_repository_archive(requests[0], archives[0])
        assert duplicate.attempt_count == restored.attempt_count
        assert duplicate.status == "succeeded"
        assert await receipt() == sources_a
    finally:
        await delete_organization(binding.organization_id, 1)


@pytest.mark.asyncio
async def test_directory_migration_preserves_review_and_sibling_source_data(tmp_path):
    import cognee
    from uuid import uuid4
    from sqlalchemy import select
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.models import WeaveMemorySource
    from cognee.modules.weave.memory_sources import source_data_id, source_records
    from cognee.modules.weave.organizations import (
        provision_organization,
        set_weave_organization_scope,
    )
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.tasks.ingestion.data_item import DataItem

    binding = await provision_organization(uuid4())
    user = await get_user(binding.service_user_id)
    records = [
        (920030, "doc:legacy", "old repository source"),
        (920031, "doc:sibling", "sibling source"),
        (920030, "review:" + str(uuid4()), "saved historical review"),
    ]
    ids = [source_data_id(binding.organization_id, repo, key) for repo, key, _ in records]
    try:
        # Seed through native remember so the fixture has the same graph
        # provenance marker as the already-indexed V2 staging dataset.
        baseline = tmp_path / "legacy-repository"
        baseline.mkdir()
        (baseline / "go.mod").write_text("module example.com/legacy\n\ngo 1.24\n")
        (baseline / "main.go").write_text("package main\nfunc Legacy() {}\n")
        async with scoped_database_context_variables(binding.dataset_id, user.id):
            await cognee.remember(str(baseline), dataset_id=binding.dataset_id, user=user)
        async with get_relational_engine().get_async_session() as session:
            legacy_code_id = await session.scalar(
                select(Data.id).where(Data.dataset_id == binding.dataset_id)
            )
        async with scoped_database_context_variables(binding.dataset_id, user.id):
            await cognee.add(
                [
                    DataItem(data=content, data_id=data_id)
                    for (_, _, content), data_id in zip(records, ids)
                ],
                dataset_id=binding.dataset_id,
                user=user,
            )
        # Stamp graph facts with the same native ownership used by cognify.
        # Migration must remove only the old repository's facts.
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.infrastructure.databases.provenance.source_refs import make_source_ref_key

        canaries = [str(uuid4()) for _ in ids]
        async with scoped_database_context_variables(binding.dataset_id, user.id):
            graph = await get_graph_engine()
            for canary, data_id, (_, _, content) in zip(canaries, ids, records):
                await graph.add_nodes(
                    [(canary, {"name": content, "type": "Entity"})],
                    source_ref_key=make_source_ref_key(binding.dataset_id, data_id),
                )
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, binding.organization_id)
            session.add(
                WeaveMemorySource(
                    organization_id=binding.organization_id,
                    github_repository_id=920030,
                    source_key="code",
                    data_id=legacy_code_id,
                    content_hash="fixture",
                    status="completed",
                )
            )
            for (repo, key, _), data_id in zip(records, ids):
                session.add(
                    WeaveMemorySource(
                        organization_id=binding.organization_id,
                        github_repository_id=repo,
                        source_key=key,
                        data_id=data_id,
                        content_hash="fixture",
                        status="completed",
                    )
                )
            await session.commit()
        await index_repository_archive(
            _request(binding.organization_id, 920030, "migration", "a" * 40),
            _repository_archive(tmp_path, "migration", "migration"),
        )
        async with get_relational_engine().get_async_session() as session:
            remaining = set(await session.scalars(select(Data.id).where(Data.id.in_(ids))))
        assert remaining == {ids[1], ids[2]}
        assert {r.data_id for r in await source_records(binding)} == remaining
        async with scoped_database_context_variables(binding.dataset_id, user.id):
            graph = await get_graph_engine()
            nodes, _ = await graph.get_graph_data()
            surviving_nodes = {str(node_id) for node_id, _ in nodes}
        assert canaries[0] not in surviving_nodes
        assert set(canaries[1:]) <= surviving_nodes
    finally:
        await delete_organization(binding.organization_id, 1)


@pytest.mark.asyncio
async def test_config_file_anchor_follows_snapshot_even_when_parsed_code_is_unchanged(tmp_path):
    from uuid import uuid4
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.deletion import delete_organization

    binding = await provision_organization(uuid4())
    try:
        first = _repository_archive(tmp_path, "config", "payment")
        with zipfile.ZipFile(first, "a") as archive:
            archive.writestr("config/settings.yaml", "retries: 2")
        await index_repository_archive(
            _request(binding.organization_id, 920050, "config", "a" * 40), first
        )
        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert any(p.get("file_path") == "settings.yaml" for _, p in nodes)
        second = _repository_archive(tmp_path, "config", "payment")
        await index_repository_archive(
            _request(binding.organization_id, 920050, "config", "b" * 40), second
        )
        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert not any(p.get("file_path") == "settings.yaml" for _, p in nodes)
        assert all(p.get("indexed_sha") == "b" * 40 for _, p in nodes)
    finally:
        await delete_organization(binding.organization_id, 1)
