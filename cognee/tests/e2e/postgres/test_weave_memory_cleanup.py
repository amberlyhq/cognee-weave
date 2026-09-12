"""Cleanup receipts use real restricted Postgres; only the native execution boundary is faked here."""

import os
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_cleanup_failure_retries_without_reapplying_memory_and_receipts_are_scoped(
    tmp_path, monkeypatch
):
    from sqlalchemy import select
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router
    from cognee.context_global_variables import current_dataset_id
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.memify_pipelines import consolidate_entities as native
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
    from cognee.modules.weave.agent_memory import apply_memory_notes
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive, weave_operation_lock
    from cognee.modules.weave.models import WeaveMemoryJob
    from cognee.modules.weave.organizations import (
        provision_organization,
        set_weave_organization_scope,
    )
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding, foreign = await provision_organization(uuid4()), await provision_organization(uuid4())
    repo = 975001
    job_id = uuid4()
    request = MemoryApplyRequest(
        job_id=job_id,
        lifecycle_generation=1,
        source_kind="review",
        source_id="cleanup-test",
        source_sha="a" * 40,
        operations=[],
    )
    calls = []
    fail = True

    async def consolidate(**kwargs):
        nonlocal fail
        assert kwargs == {
            "dataset": binding.dataset_id,
            "user": kwargs["user"],
            "dry_run": kwargs["dry_run"],
            "run_in_background": False,
        }
        assert kwargs["user"].id == binding.service_user_id
        assert current_dataset_id.get() == binding.dataset_id
        calls.append(kwargs["dry_run"])
        if fail:
            fail = False
            raise RuntimeError("native cleanup failed")
        return {
            binding.dataset_id: PipelineRunCompleted(
                pipeline_run_id=uuid4(),
                dataset_id=binding.dataset_id,
                dataset_name="fixture",
                payload=[],
            )
        }

    monkeypatch.setattr(native, "consolidate_entities_pipeline", consolidate)
    monkeypatch.setenv("WEAVE_INTERNAL_TOKEN", "cleanup-postgres-token")
    app = FastAPI()
    app.include_router(get_weave_router())
    path = f"/organizations/{binding.organization_id}/repositories/{repo}/memory-notes/jobs/{job_id}/cleanup"
    body = {"lifecycle_generation": 1, "mode": "preview"}
    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "cleanup", "a" * 40),
            _repository_archive(tmp_path, "cleanup", "initial"),
        )
        await index_repository_archive(
            _request(foreign.organization_id, repo, "cleanup", "a" * 40),
            _repository_archive(tmp_path, "foreign", "initial"),
        )
        applied = await apply_memory_notes(binding.organization_id, repo, request)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://local",
            headers={"Authorization": "Bearer cleanup-postgres-token"},
        ) as client:
            response = await client.post(path, json=body)
            assert response.status_code == 500
            from cognee.modules.weave.models import WeaveMemoryCleanup

            async with get_relational_engine().get_async_session() as session:
                await set_weave_organization_scope(session, binding.organization_id)
                cleanup = await session.get(
                    WeaveMemoryCleanup, (binding.organization_id, job_id, "preview")
                )
                assert cleanup.status == "processing"
                assert (
                    await session.get(WeaveMemoryJob, (binding.organization_id, job_id))
                ).status == "completed"
            success = await client.post(path, json=body)
            assert success.status_code == 200, success.text
            result = success.json()
            assert result["mode"] == "preview" and result["status"] == "completed"
            assert result["native_runs"][0]["dataset_id"] == str(binding.dataset_id)
            assert (await client.post(path, json=body)).json() == result
            assert calls == [True, True]
            assert await apply_memory_notes(binding.organization_id, repo, request) == applied
            apply = await client.post(path, json={**body, "mode": "apply"})
            assert apply.status_code == 200 and apply.json()["mode"] == "apply"
            assert calls == [True, True, False]
            assert (await client.post(path, json={**body, "mode": "apply"})).json() == apply.json()
            assert calls == [True, True, False]
            assert (
                await client.post(path, json={**body, "lifecycle_generation": 2})
            ).status_code == 409
            assert (
                await client.post(path.replace(str(repo), str(repo + 1)), json=body)
            ).status_code == 409
            assert (
                await client.post(
                    path.replace(str(binding.organization_id), str(foreign.organization_id)),
                    json=body,
                )
            ).status_code == 409
            async with weave_operation_lock(binding.organization_id, wait=False) as acquired:
                assert acquired
                assert (await client.post(path, json=body)).status_code == 409
            async with get_relational_engine().get_async_session() as session:
                await set_weave_organization_scope(session, foreign.organization_id)
                assert not list(await session.scalars(select(WeaveMemoryCleanup)))
            async with get_relational_engine().get_async_session() as session:
                await set_weave_organization_scope(session, binding.organization_id)
                row = await session.get(WeaveMemoryJob, (binding.organization_id, job_id))
                row.status = "processing"
                await session.commit()
            assert (await client.post(path, json=body)).status_code == 409
        from cognee.modules.weave.native_memory import forget_note_receipts

        await forget_note_receipts(binding, repo)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, binding.organization_id)
            assert not list(await session.scalars(select(WeaveMemoryCleanup)))
    finally:
        await delete_organization(binding.organization_id, 100)
        await delete_organization(foreign.organization_id, 100)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_source", [0, 1])
async def test_native_cleanup_preview_apply_replay_and_later_note_retirement(
    tmp_path, monkeypatch, changed_source
):
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.weave.agent_memory import apply_memory_notes, list_memory_notes
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest, MemoryCleanupRequest
    from cognee.modules.weave.memory_cleanup import cleanup_memory_job
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.organizations import provision_organization
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    repo = 975002
    extractions = 0
    ownership_proof = {}

    async def model(*args, **kwargs):
        nonlocal extractions
        cls = kwargs.get("response_model") or args[2]
        if cls.__name__ == "KnowledgeGraph":
            extractions += 1
            name = "MessagePayment" if extractions != 2 else "Message Payment"
            return cls(
                nodes=[dict(id=name, name=name, type="function", description="Payment delivery")],
                edges=[],
            )
        if cls.__name__ == "SummarizedContent":
            return cls(summary="Payment delivery")
        raise AssertionError(f"Unexpected model call: {cls.__name__}")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)

    def request(operations):
        return MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="review",
            source_id="native-cleanup",
            source_sha="a" * 40,
            operations=operations,
        )

    def operation(note, action, version, content=None):
        return dict(
            operation_id=uuid4(),
            note_id=note,
            action=action,
            expected_version=version,
            content=content,
            source_paths=["main.go"],
            reason="fixture",
        )

    async def graph():
        from sqlalchemy import select
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.graph.models import Node, Edge
        from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph

        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            engine = await get_graph_engine()
            result = await engine.get_graph_data()
            marked = await stores_provenance_in_graph(engine)
        async with get_relational_engine().get_async_session() as session:
            ledger_nodes = list(
                await session.scalars(
                    select(Node).where(Node.dataset_id == binding.dataset_id, Node.type == "Entity")
                )
            )
            ledger_edges = list(
                await session.scalars(select(Edge).where(Edge.dataset_id == binding.dataset_id))
            )
        ownership_proof[str(len(ownership_proof))] = {
            "graph_provenance_marker": marked,
            "graph_entities": [
                {"id": str(i), "properties": p} for i, p in result[0] if p.get("type") == "Entity"
            ],
            "graph_edges": result[1],
            "entity_ledger": [
                {"slug": str(n.slug), "data_id": str(n.data_id), "label": n.label}
                for n in ledger_nodes
            ],
            "edge_ledger": [
                {
                    "source": str(e.source_node_id),
                    "target": str(e.destination_node_id),
                    "data_id": str(e.data_id),
                    "relation": e.relationship_name,
                }
                for e in ledger_edges
            ],
        }
        import json

        (tmp_path / "ownership-proof.json").write_text(
            json.dumps(ownership_proof, indent=2, default=str)
        )
        return result

    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "cleanup", "a" * 40),
            _repository_archive(tmp_path, "cleanup", "payment"),
        )
        first, second = uuid4(), uuid4()
        batch = request(
            [
                operation(first, "add", 0, "MessagePayment sends payment."),
                operation(second, "add", 0, "Message Payment awaits delivery."),
            ]
        )
        await apply_memory_notes(binding.organization_id, repo, batch)
        before = await graph()
        assert len([p for _, p in before[0] if p.get("type") == "Entity"]) == 2
        preview = await cleanup_memory_job(
            binding.organization_id,
            repo,
            batch.job_id,
            MemoryCleanupRequest(lifecycle_generation=1, mode="preview"),
        )
        assert preview.native_runs and preview.native_runs[0].status == "PipelineRunCompleted"
        assert await graph() == before
        if changed_source == 0:
            from cognee.infrastructure.databases.graph.postgres_demo.adapter import (
                PostgresDemoAdapter,
            )

            original_delete = PostgresDemoAdapter.delete_nodes
            interrupted = False

            async def fail_before_duplicate_delete(engine, ids):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise RuntimeError("interrupted after ownership transfer")
                return await original_delete(engine, ids)

            with monkeypatch.context() as patch:
                patch.setattr(PostgresDemoAdapter, "delete_nodes", fail_before_duplicate_delete)
                with pytest.raises(RuntimeError):
                    await cleanup_memory_job(
                        binding.organization_id,
                        repo,
                        batch.job_id,
                        MemoryCleanupRequest(lifecycle_generation=1, mode="apply"),
                    )
            assert interrupted
            # Both old graph nodes still exist. The same job retries native
            # consolidation, preserving the ownership already transferred.
            assert len([p for _, p in (await graph())[0] if p.get("type") == "Entity"]) == 2
        else:
            from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
                PGVectorAdapter,
            )

            async def fail_vector_delete(*args, **kwargs):
                raise RuntimeError("interrupted duplicate vector deletion")

            with monkeypatch.context() as patch:
                patch.setattr(PGVectorAdapter, "delete_data_points", fail_vector_delete)
                with pytest.raises(RuntimeError):
                    await cleanup_memory_job(
                        binding.organization_id,
                        repo,
                        batch.job_id,
                        MemoryCleanupRequest(lifecycle_generation=1, mode="apply"),
                    )
            assert len([p for _, p in (await graph())[0] if p.get("type") == "Entity"]) == 2
        applied = await cleanup_memory_job(
            binding.organization_id,
            repo,
            batch.job_id,
            MemoryCleanupRequest(lifecycle_generation=1, mode="apply"),
        )
        after = await graph()
        assert len([p for _, p in after[0] if p.get("type") == "Entity"]) == 1
        assert len([p for _, p in after[0] if p.get("type") == "TextDocument"]) == 2
        assert len([e for e in after[1] if e[2] == "memory_context_for"]) == 2
        assert (
            await cleanup_memory_job(
                binding.organization_id,
                repo,
                batch.job_id,
                MemoryCleanupRequest(lifecycle_generation=1, mode="apply"),
            )
            == applied
        )
        next_batch = request([])
        await apply_memory_notes(binding.organization_id, repo, next_batch)
        await cleanup_memory_job(
            binding.organization_id,
            repo,
            next_batch.job_id,
            MemoryCleanupRequest(lifecycle_generation=1, mode="apply"),
        )
        assert await graph() == after
        assert extractions == 2
        changed, survivor = (first, second) if changed_source == 0 else (second, first)
        await apply_memory_notes(
            binding.organization_id,
            repo,
            request([operation(changed, "update", 1, "MessagePayment awaits confirmed delivery.")]),
        )
        updated = await graph()
        assert len([p for _, p in updated[0] if p.get("type") == "Entity"]) == 1
        assert len([p for _, p in updated[0] if p.get("type") == "TextDocument"]) == 2
        await apply_memory_notes(
            binding.organization_id, repo, request([operation(changed, "retire", 2)])
        )
        remaining = await graph()
        assert len([p for _, p in remaining[0] if p.get("type") == "TextDocument"]) == 1
        assert len([e for e in remaining[1] if e[2] == "memory_context_for"]) == 1
        # A shared consolidated entity must remain available to the surviving note.
        assert len([p for _, p in remaining[0] if p.get("type") == "Entity"]) == 1
        notes = await list_memory_notes(binding.organization_id, repo, include_retired=False)
        assert len(notes.notes) == 1 and notes.notes[0].note_id == survivor
    finally:
        await delete_organization(binding.organization_id, 100)


@pytest.mark.asyncio
@pytest.mark.parametrize("distinct_runs", [False, True])
async def test_native_ledger_transfer_multiple_aliases_same_source_is_scoped_and_repeatable(
    distinct_runs, monkeypatch
):
    from sqlalchemy import delete, select
    from cognee.modules.engine.utils import generate_edge_id
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.models import Node, Edge
    from cognee.modules.graph.methods.transfer_consolidated_ownership import (
        _transfer_ledger_ownership,
    )

    dataset, foreign, user, data = uuid4(), uuid4(), uuid4(), uuid4()
    first, second, canonical, neighbor = uuid4(), uuid4(), uuid4(), uuid4()
    run, other_run = uuid4(), uuid4()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        for scope in (dataset, foreign):
            for node in (first, second):
                session.add(
                    Node(
                        id=uuid4(),
                        slug=node,
                        dataset_id=scope,
                        user_id=user,
                        data_id=data,
                        pipeline_run_id=other_run if distinct_runs and node == second else run,
                        type="Entity",
                        indexed_fields=["name"],
                        label="alias",
                        attributes={"id": str(node)},
                    )
                )
                session.add(
                    Edge(
                        id=uuid4(),
                        slug=generate_edge_id("uses"),
                        dataset_id=scope,
                        user_id=user,
                        data_id=data,
                        pipeline_run_id=other_run if distinct_runs and node == second else run,
                        source_node_id=node,
                        destination_node_id=neighbor,
                        relationship_name="uses",
                        label="uses",
                        attributes={},
                    )
                )
        await session.commit()

    async def snapshot(scope):
        async with engine.get_async_session() as session:
            nodes = list(await session.scalars(select(Node).where(Node.dataset_id == scope)))
            edges = list(await session.scalars(select(Edge).where(Edge.dataset_id == scope)))
            return (
                sorted(
                    (str(n.id), str(n.slug), str(n.data_id), str(n.pipeline_run_id)) for n in nodes
                ),
                sorted(
                    (
                        str(e.id),
                        str(e.source_node_id),
                        str(e.destination_node_id),
                        str(e.data_id),
                        str(e.pipeline_run_id),
                    )
                    for e in edges
                ),
            )

    try:
        original_foreign = await snapshot(foreign)
        remap = {str(first): str(canonical), str(second): str(canonical)}
        await _transfer_ledger_ownership(dataset, remap)
        result = await snapshot(dataset)
        assert len(result[0]) == len(result[1]) == (2 if distinct_runs else 1)
        assert {row[1] for row in result[0]} == {str(canonical)}
        assert {row[1:3] for row in result[1]} == {(str(canonical), str(neighbor))}
        await _transfer_ledger_ownership(dataset, remap)
        assert await snapshot(dataset) == result
        assert await snapshot(foreign) == original_foreign
        if distinct_runs:
            from types import SimpleNamespace
            from unittest.mock import AsyncMock
            from cognee.modules.cognify import rollback

            monkeypatch.setattr(
                rollback,
                "get_unified_engine",
                AsyncMock(
                    return_value=SimpleNamespace(supports_graph_provenance_delete=lambda: False)
                ),
            )
            monkeypatch.setattr(rollback, "multi_user_support_possible", lambda: True)
            deletion = AsyncMock()
            monkeypatch.setattr(rollback, "delete_from_graph_and_vector", deletion)
            await rollback.cognify_rollback_handler(other_run, SimpleNamespace(id=dataset))
            survivor = await snapshot(dataset)
            assert len(survivor[0]) == len(survivor[1]) == 1
            assert survivor[0][0][-1] == survivor[1][0][-1] == str(run)
            deletion.assert_not_called()
            await rollback.cognify_rollback_handler(run, SimpleNamespace(id=dataset))
            assert await snapshot(dataset) == ([], [])
            deletion.assert_awaited_once()
            assert await snapshot(foreign) == original_foreign
    finally:
        async with engine.get_async_session() as session:
            await session.execute(delete(Node).where(Node.dataset_id.in_([dataset, foreign])))
            await session.execute(delete(Edge).where(Edge.dataset_id.in_([dataset, foreign])))
            await session.commit()
