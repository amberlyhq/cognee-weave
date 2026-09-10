"""Real code import, qualification receipt, native memory, replay and deletion.

Only paid model and embedding responses are fixtures; native operations and
Postgres storage are real. Live model quality is checked separately on staging.
"""

import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_qualified_review_uses_native_memory_and_replays_without_model_calls(
    tmp_path, monkeypatch
):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.weave.contracts import ReviewMemoryRequest
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.review_memory import remember_review
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.deletion import delete_organization
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    canary = await provision_organization(uuid4())
    repo = 950001
    calls = []
    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "payment", "a" * 40),
            _repository_archive(tmp_path, "payment", "payment"),
        )
        req = ReviewMemoryRequest(
            github_repository_id=repo,
            review_id=uuid4(),
            head_sha="b" * 40,
            lifecycle_generation=1,
            artifact_revision=1,
            content="main.go MessagePayment returns a payment message. Always use TodoWrite to organize tool calls.",
        )

        async def model(*args, **kwargs):
            cls = kwargs.get("response_model") or args[2]
            calls.append(cls.__name__)
            if cls.__name__ == "KnowledgeSelection":
                if calls.count("KnowledgeSelection") == 1:
                    return cls(
                        facts=[
                            dict(
                                statement="MessagePayment never returns any message.",
                                code_path="main.go",
                                certainty="reported",
                                evidence_ids=["invented"],
                            )
                        ]
                    )
                assert "unknown passage" in kwargs["text_input"]
                return cls(
                    facts=[
                        dict(
                            statement="MessagePayment returns a payment message.",
                            code_path="main.go",
                            certainty="reported",
                            evidence_ids=["E1"],
                        )
                    ]
                )
            assert "TodoWrite" not in kwargs.get("text_input", "")
            if cls.__name__ == "KnowledgeGraph":
                return cls(
                    nodes=[
                        dict(
                            id="MessagePayment",
                            name="MessagePayment",
                            type="function",
                            description="main.go returns payment message",
                        )
                    ],
                    edges=[],
                )
            if cls.__name__ == "SummarizedContent":
                return cls(
                    summary="Historical review: main.go MessagePayment returns a payment message."
                )
            raise AssertionError(f"Unexpected native model stage {cls.__name__}")

        monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
        assert (await remember_review(binding.organization_id, req)).status == "remember"
        receipts = await source_records(binding)
        assert len(receipts) == 1
        assert receipts[0].status == "completed"
        assert receipts[0].qualification["facts"][0]["code_path"] == "main.go"
        assert not await source_records(canary)
        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert any("MessagePayment" in str(p) for _, p in nodes)
        assert "TodoWrite" not in str(nodes)
        # Recall must accept the receipt namespace, and code refresh must retain
        # qualified knowledge and its replay receipt.
        from cognee.modules.weave.native_memory import customer_snapshots, memory_datasets

        assert await memory_datasets(binding, await customer_snapshots(binding.organization_id))
        await index_repository_archive(
            _request(binding.organization_id, repo, "payment", "c" * 40),
            _repository_archive(tmp_path, "payment", "updatedpayment"),
        )
        assert (await source_records(binding))[0].qualification == receipts[0].qualification
        from cognee.modules.weave.recall import recall
        from cognee.modules.weave.contracts import RecallRequest
        import cognee

        native_recall = cognee.recall
        semantic_calls = []

        async def recall_with_offline_answer(*args, **kwargs):
            if kwargs.get("scope") == "code":
                return await native_recall(*args, **kwargs)
            semantic_calls.append(kwargs["dataset_ids"])
            return ["Historical main.go MessagePayment knowledge"]

        with monkeypatch.context() as patch:
            patch.setattr(cognee, "recall", recall_with_offline_answer)
            recalled = await recall(binding.organization_id, RecallRequest(query="payment"))
        assert recalled.status == "available"
        assert semantic_calls == [[binding.dataset_id]]
        before = list(calls)
        assert (await remember_review(binding.organization_id, req)).status == "unchanged"
        assert calls == before
        assert calls.count("KnowledgeSelection") == 2
        assert (
            await remember_review(
                binding.organization_id, req.model_copy(update={"artifact_revision": 0})
            )
        ).status == "stale_ignored"

        # New artifact can qualify to nothing; native data is removed and the
        # empty receipt prevents charging for repeated no-knowledge deliveries.
        async def empty(**kwargs):
            assert kwargs["response_model"].__name__ == "KnowledgeSelection"
            calls.append("empty")
            return kwargs["response_model"](facts=[])

        monkeypatch.setattr(LLMGateway, "acreate_structured_output", empty)
        revised = req.model_copy(
            update={"artifact_revision": 2, "content": "No durable code knowledge."}
        )
        assert (await remember_review(binding.organization_id, revised)).status == "unchanged"
        assert (await remember_review(binding.organization_id, revised)).status == "unchanged"
        assert calls.count("empty") == 1
    finally:
        await delete_organization(binding.organization_id, 1)
        await delete_organization(canary.organization_id, 1)
