"""Actual code imports, merge receipts, native tagged chunks, and tenant cleanup.

Only paid LLM responses are fixtures. Postgres, native ingestion, vectors, graph
annotations, and native CHUNKS recall execute their real implementations.
"""

import json
import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_index_retry_preserves_changed_paths_after_hash_stamp_crash(tmp_path, monkeypatch):
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.weave import merge_qualification, review_code_links
    from cognee.modules.weave.contracts import MergeKnowledgeChange
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.merge_memory import remember_merge
    from cognee.modules.weave.native_memory import customer_snapshots
    from cognee.modules.weave.organizations import provision_organization
    from cognee.tests.e2e.postgres.test_weave_hybrid_recall import _archive, _request

    binding = await provision_organization(uuid4())
    repo, name = 960002, "merge-index-retry"
    directories = [tmp_path / "a", tmp_path / "b"]
    for directory in directories:
        directory.mkdir()
    alpha = _archive(directories[0], name, "alpha")
    beta = _archive(directories[1], name, "beta")
    request_a = _request(binding.organization_id, repo, name, "a" * 40)
    request_b = _request(binding.organization_id, repo, name, "b" * 40)

    async def change_receipt():
        return next(
            r
            for r in await source_records(binding, repo)
            if r.source_key == "review:code-change:" + "b" * 40
        )

    async def snapshot():
        return next(
            s
            for s in await customer_snapshots(binding.organization_id)
            if s.github_repository_id == repo
        )

    async def crash_after_hash_stamp(*args, **kwargs):
        raise RuntimeError("Injected failure after file hash stamp")

    try:
        await index_repository_archive(request_a, alpha)
        with monkeypatch.context() as patch:
            patch.setattr(review_code_links, "sync_review_code_links", crash_after_hash_stamp)
            with pytest.raises(RuntimeError, match="after file hash stamp"):
                await index_repository_archive(request_b, beta)
        assert (await snapshot()).status == "failed"
        original = (await change_receipt()).qualification
        assert original["changed_paths"] == ["main.go"]
        assert original["complete"] is True
        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        stamped = [
            p
            for _, p in nodes
            if p.get("type") == "CodeFileReference" and p.get("file_path") == "main.go"
        ]
        assert stamped and all(
            p.get("indexed_sha") == "b" * 40 and p.get("weave_content_hash") for p in stamped
        )

        await index_repository_archive(request_b, beta)
        assert (await snapshot()).status == "indexed"
        retained = (await change_receipt()).qualification
        assert {key: retained[key] for key in original} == original

        # GitHub's comparison can be incomplete. The durable local delta must
        # still send the changed source to qualification after the retry.
        qualified_files = []

        async def qualify(**kwargs):
            qualified_files.append(kwargs["files"])
            return {"facts": [], "decisions": [], "coverage_complete": True}

        monkeypatch.setattr(merge_qualification, "qualify_merge", qualify)
        merged = await remember_merge(
            request_b,
            MergeKnowledgeChange(
                base_sha="a" * 40, head_sha="b" * 40, changed_paths=[], complete=False
            ),
            beta,
        )
        assert merged.status == "completed"
        assert len(qualified_files) == 1
        assert 'return "beta"' in qualified_files[0]["main.go"]
    finally:
        await delete_organization(binding.organization_id, 2)


@pytest.mark.asyncio
async def test_merged_knowledge_replaces_current_claims_preserves_history_and_isolates_tenants(
    tmp_path, monkeypatch
):
    import cognee
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.data.models import Data
    from cognee.modules.search.types import SearchType
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
    from cognee.modules.weave.contracts import MergeKnowledgeChange, ReviewMemoryRequest
    from cognee.modules.weave.deletion import delete_organization, delete_repository
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.knowledge_lifecycle import fact_digest
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources, recall_current_memory
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.merge_memory import remember_merge
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.review_memory import remember_review
    from cognee.tests.e2e.postgres.test_weave_hybrid_recall import _archive, _request

    binding, canary = await provision_organization(uuid4()), await provision_organization(uuid4())
    repository_id, name = 960001, "merge-payment"
    old_statement = "The Message function in main.go returns the alpha message."
    calls = []

    async def model(*args, **kwargs):
        schema = kwargs.get("response_model") or args[2]
        calls.append(schema.__name__)
        text = kwargs.get("text_input", "")
        if schema.__name__ == "KnowledgeSelection":
            return schema(
                facts=[
                    dict(
                        statement=old_statement,
                        code_path="main.go",
                        certainty="reported",
                        evidence_ids=["E1"],
                    )
                ]
            )
        if schema.__name__ in {"KnowledgeAudit", "MergeAudit"}:
            return schema(issues=[])
        if schema.__name__ == "MergeSelection":
            packet = json.loads(text.split("\nCURRENT SOURCE PASSAGES:", 1)[0])
            value = "beta" if 'return "beta"' in text else "alpha"
            statement = f"The Message function in main.go returns the {value} message."
            return schema(
                facts=[
                    dict(
                        statement=statement,
                        code_path="main.go",
                        certainty="observed",
                        evidence_ids=["E1"],
                    )
                ],
                decisions=[
                    dict(
                        fact_id=f["fact_id"],
                        status="keep" if f["statement"] == statement else "superseded",
                        reason="The exact return value is shown in current source.",
                        evidence_ids=["E1"],
                    )
                    for f in packet["prior_facts"]
                ],
            )
        if schema.__name__ == "KnowledgeGraph":
            # Shared native entities deliberately retain historical wording.
            # Current recall must select source chunks, never synthesize this.
            return schema(
                nodes=[
                    dict(id="Message", name="Message", type="function", description=old_statement)
                ],
                edges=[],
            )
        if schema.__name__ == "SummarizedContent":
            return schema(summary=old_statement)
        raise AssertionError(f"Unexpected model stage: {schema.__name__}")

    async def graph(owner):
        async with scoped_database_context_variables(owner.dataset_id, owner.service_user_id):
            return await (await get_graph_engine()).get_graph_data()

    def archive(label, value):
        directory = tmp_path / label
        directory.mkdir()
        return _archive(directory, name, value)

    alpha, beta = archive("alpha", "alpha"), archive("beta", "beta")
    user = await get_user(binding.service_user_id)
    llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
    try:
        for owner in (binding, canary):
            await index_repository_archive(
                _request(owner.organization_id, repository_id, name, "a" * 40), alpha
            )
        monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
        review = ReviewMemoryRequest(
            github_repository_id=repository_id,
            review_id=uuid4(),
            head_sha="a" * 40,
            lifecycle_generation=1,
            artifact_revision=1,
            content=old_statement,
        )
        for owner in (binding, canary):
            assert (await remember_review(owner.organization_id, review)).status == "remember"
        original = next(
            r for r in await source_records(binding) if r.source_key.startswith("review:qualified:")
        )
        original_digest = fact_digest(original.qualification["facts"][0])
        canary_before = json.dumps(await graph(canary), sort_keys=True, default=str)

        request_b = _request(binding.organization_id, repository_id, name, "b" * 40)
        await index_repository_archive(request_b, beta)
        invalidated = next(
            r for r in await source_records(binding) if r.source_key == original.source_key
        )
        assert (
            invalidated.qualification["fact_states"][original_digest]["status"] == "needs_recheck"
        )
        nodes, _ = await graph(binding)
        old_node = next(
            p
            for _, p in nodes
            if p.get("type") == "ReviewKnowledge" and p.get("source_key") == original.source_key
        )
        assert old_node["knowledge_status"] == "needs_recheck"
        assert not eligible_memory_sources(binding, await source_records(binding))

        change_b = MergeKnowledgeChange(
            base_sha="a" * 40, head_sha="b" * 40, changed_paths=["main.go"], complete=True
        )
        merged = await remember_merge(request_b, change_b, beta)
        assert merged.status == "completed" and merged.facts_saved == 1
        records = await source_records(binding)
        current = eligible_memory_sources(binding, records)
        assert len(current) == 1 and current[0].qualification["facts"][0]["certainty"] == "observed"
        historical = next(r for r in records if r.source_key == original.source_key)
        assert historical.qualification["fact_states"][original_digest]["status"] == "superseded"
        async with get_relational_engine().get_async_session() as session:
            assert await session.get(Data, original.data_id) is not None
        async with scoped_database_context_variables(
            binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
        ):
            chunks = await recall_current_memory(
                binding,
                user,
                "What does Message return?",
                records,
                repository_ids={repository_id},
                top_k=10,
                llm=llm,
                embedding=embedding,
            )
        assert chunks
        assert all("beta message" in r.text and "alpha message" not in r.text for r in chunks)
        assert {r.metadata["data_id"] for r in chunks} == {str(current[0].data_id)}
        assert all(r.dataset_id == str(binding.dataset_id) for r in chunks)
        nodes, edges = await graph(binding)
        current_nodes = [
            (i, p)
            for i, p in nodes
            if p.get("type") == "ReviewKnowledge" and p.get("knowledge_status") == "current"
        ]
        assert len(current_nodes) == 1
        assert current_nodes[0][1]["checked_sha"] == "b" * 40
        assert any(
            a == current_nodes[0][0] and relation == "review_context_for"
            for a, _, relation, _ in edges
        )

        # Native recall on the other real dataset cannot resolve these source tags.
        canary_user = await get_user(canary.service_user_id)
        async with scoped_database_context_variables(
            canary.dataset_id, canary_user.id, llm_config=llm, embedding_config=embedding
        ):
            foreign = await cognee.recall(
                "Message",
                query_type=SearchType.CHUNKS,
                scope="graph",
                auto_route=False,
                dataset_ids=[canary.dataset_id],
                node_name=[current[0].qualification["native_source_tag"]],
                user=canary_user,
                llm_config=llm,
                embedding_config=embedding,
            )
        assert not foreign
        before_replay = list(calls)
        replay = await remember_merge(request_b, change_b, beta)
        assert replay.status == "unchanged" and replay.facts_saved == 1
        assert calls == before_replay
        original_b_plan = next(
            r.source_key
            for r in records
            if r.source_key.startswith("review:merge:") and r.source_key.endswith(":plan")
        )

        # Returning through A to B creates a new B epoch without losing history.
        request_a = _request(binding.organization_id, repository_id, name, "a" * 40)
        await index_repository_archive(request_a, alpha)
        assert not eligible_memory_sources(binding, await source_records(binding))
        await remember_merge(
            request_a,
            MergeKnowledgeChange(
                base_sha="b" * 40, head_sha="a" * 40, changed_paths=["main.go"], complete=True
            ),
            alpha,
        )
        await index_repository_archive(request_b, beta)
        assert (await remember_merge(request_b, change_b, beta)).status == "completed"
        records = await source_records(binding)
        b_plans = [
            r.source_key
            for r in records
            if r.source_key.startswith("review:merge:" + "b" * 40)
            and r.source_key.endswith(":plan")
        ]
        assert len(b_plans) == 2 and original_b_plan in b_plans
        assert len(eligible_memory_sources(binding, records)) == 1
        assert canary_before == json.dumps(await graph(canary), sort_keys=True, default=str)

        data_ids = [r.data_id for r in records if r.qualification and r.qualification.get("facts")]
        await delete_repository(binding.organization_id, repository_id, 2)
        assert not await source_records(binding, repository_id)
        async with get_relational_engine().get_async_session() as session:
            for data_id in data_ids:
                assert await session.get(Data, data_id) is None
        remaining, _ = await graph(binding)
        assert all(p.get("github_repository_id") != repository_id for _, p in remaining)
        assert canary_before == json.dumps(await graph(canary), sort_keys=True, default=str)
    finally:
        await delete_organization(binding.organization_id, 2)
        await delete_organization(canary.organization_id, 2)
