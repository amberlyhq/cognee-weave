"""Real native session/cache/graph pipeline; only paid model responses are fixtures."""

import importlib
import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_native_review_learning_retries_and_survives_code_refresh(tmp_path, monkeypatch):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.weave.contracts import ReviewMemoryRequest
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.native_memory import customer_dataset
    from cognee.modules.weave.indexing import index_repository_archive, weave_operation_lock
    from cognee.modules.weave.review_sessions import sync_review_session, session_identity
    from cognee.modules.weave.config import get_weave_llm_config, get_weave_embedding_config
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.users.methods import get_user
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    canary = await provision_organization(uuid4())
    user = await get_user(binding.service_user_id)
    repo = 940001
    first = _repository_archive(tmp_path, "payment", "payment")
    await index_repository_archive(
        _request(binding.organization_id, repo, "payment", "a" * 40), first
    )
    dataset = await customer_dataset(binding)
    request = ReviewMemoryRequest(
        github_repository_id=repo,
        review_id=uuid4(),
        head_sha="b" * 40,
        lifecycle_generation=1,
        artifact_revision=0,
        content="Final review",
        sessions=[
            dict(
                invocationId=str(uuid4()),
                sessionId=str(uuid4()),
                threadId=str(uuid4()),
                role="production_risk_review",
                result="main.go MessagePayment uses idempotency keys.",
                steps=[
                    dict(
                        id="read-payment",
                        type="tool.completed",
                        toolName="read_file",
                        content="main.go MessagePayment uses idempotency keys to prevent duplicate charges.",
                    )
                ],
            )
        ],
    )
    entry = request.sessions[0]
    calls = []
    fail = True

    async def model(*args, **kwargs):
        nonlocal fail
        cls = kwargs.get("response_model") or args[2]
        name = cls.__name__
        calls.append(name)
        if name == "AgentContextExtraction":
            assert "MessagePayment" in kwargs["text_input"]
            return cls(
                lessons=[
                    dict(
                        section="environment_facts",
                        content="main.go MessagePayment uses idempotency keys to prevent duplicate charges.",
                        confidence=0.95,
                    )
                ]
            )
        if name == "CuratorBatchOutput":
            if fail:
                raise RuntimeError("Injected provider outage")
            return cls(
                lessons=[
                    dict(
                        working_statement="main.go MessagePayment uses idempotency keys.",
                        member_entry_ids=[],
                    )
                ]
            )
        if name == "WrittenLesson":
            return cls(
                accept=True,
                statement="main.go MessagePayment uses idempotency keys to prevent duplicate charges.",
                entities=["MessagePayment"],
                why_learned="Historical PR review inspected main.go",
            )
        if name == "KnowledgeGraph":
            return cls(
                nodes=[
                    dict(
                        id="MessagePayment",
                        name="MessagePayment",
                        type="function",
                        description="main.go payment function",
                    ),
                    dict(
                        id="idempotency keys",
                        name="idempotency keys",
                        type="mechanism",
                        description="prevent duplicate charges",
                    ),
                ],
                edges=[
                    dict(
                        source_node_id="MessagePayment",
                        target_node_id="idempotency keys",
                        relationship_name="uses",
                    )
                ],
            )
        if name == "SummarizedContent":
            return cls(summary="MessagePayment uses idempotency keys.")
        raise AssertionError(f"Unexpected model call: {name}")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)

    async def sync():
        async with weave_operation_lock(binding.organization_id):
            return await sync_review_session(
                binding,
                user,
                dataset,
                request,
                entry,
                llm=get_weave_llm_config(),
                embedding=get_weave_embedding_config(),
            )

    try:
        with pytest.raises(RuntimeError, match="Injected provider outage"):
            await sync()
        assert (await source_records(binding))[0].status != "completed"
        extracted = calls.count("AgentContextExtraction")
        fail = False
        assert await sync() == "remember"
        assert calls.count("AgentContextExtraction") == extracted
        before = len(calls)
        assert await sync() == "unchanged"
        assert len(calls) == before
        native_id = session_identity(binding, request, entry)
        manager = get_session_manager(dataset_id=dataset.id)
        assert (
            len(
                await manager.get_session(
                    user_id=str(user.id), session_id=native_id, formatted=False
                )
            )
            == 1
        )
        assert (
            await manager.get_session(
                user_id=str(canary.service_user_id), session_id=native_id, formatted=False
            )
            == []
        )

        async def graph_state():
            async with scoped_database_context_variables(dataset.id, user.id):
                return await (await get_graph_engine()).get_graph_data()

        nodes, edges = await graph_state()
        lesson_ids = {
            node_id
            for node_id, props in nodes
            if props.get("type") in {"TextDocument", "DocumentChunk", "TextSummary"}
        }
        assert lesson_ids
        assert any("idempotency" in str(props) for _, props in nodes)
        for sha, label in [("b", "newpayment"), ("a", "payment")]:
            directory = tmp_path / sha
            directory.mkdir()
            archive = _repository_archive(directory, "payment", label)
            await index_repository_archive(
                _request(binding.organization_id, repo, "payment", sha * 40), archive
            )
            assert (await customer_dataset(binding)).id == dataset.id
            current, _ = await graph_state()
            assert lesson_ids.issubset({node_id for node_id, _ in current})
        assert len(calls) == before  # code refreshes do not revisit memory models
        import cognee
        from cognee.modules.weave.recall import recall
        from cognee.modules.weave.contracts import RecallRequest

        original_recall = cognee.recall

        async def unavailable_semantic(*args, **kwargs):
            if kwargs.get("scope") == "code":
                return await original_recall(*args, **kwargs)
            raise RuntimeError("Injected exhausted provider credits")

        monkeypatch.setattr(cognee, "recall", unavailable_semantic)
        # Exact wire shape sent by Amberly for initial review context: the query
        # names a PR, not a symbol. Empty seeds request a bounded code overview.
        context = await recall(
            binding.organization_id,
            RecallRequest(
                mode="repository_context",
                query="amberlyhq/payment pull request 17",
                primary_github_repository_id=repo,
                seeds=[],
                top_k=10,
                depth=1,
            ),
        )
        assert context.status == "available"
        assert "MessagePayment" in context.native_memory
        assert [diagnostic.code for diagnostic in context.diagnostics] == ["backend_unavailable"]

        await delete_organization(binding.organization_id, 1)
        assert (
            await manager.get_session(user_id=str(user.id), session_id=native_id, formatted=False)
            == []
        )
    finally:
        await delete_organization(binding.organization_id, 1)
        await delete_organization(canary.organization_id, 1)
