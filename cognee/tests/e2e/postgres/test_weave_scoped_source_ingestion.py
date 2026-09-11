"""Saving one qualified source must not revisit a customer's old corpus."""

import os
from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remember", "update", "retry"])
async def test_qualified_source_ignores_disposed_code_archive_and_pending_old_corpus(
    tmp_path, monkeypatch, operation
):
    import cognee
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.data.models import Data
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.archive import validated_archive
    from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
    from cognee.modules.weave.contracts import ReviewMemoryRequest
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.models import WeaveMemorySource
    from cognee.modules.weave.organizations import (
        provision_organization,
        set_weave_organization_scope,
    )
    from cognee.modules.weave.review_memory import remember_review
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tests.e2e.postgres.test_weave_hybrid_recall import _archive, _request

    binding = await provision_organization(uuid4())
    user = await get_user(binding.service_user_id)
    repo = 960003
    archive = _archive(tmp_path, "scoped-memory", "payment")
    calls = []
    statement = "FRESH_REVIEW: Message in main.go returns the payment message."

    async def model(*args, **kwargs):
        schema = kwargs.get("response_model") or args[2]
        text = kwargs.get("text_input", "")
        calls.append((schema.__name__, text))
        if schema.__name__ == "KnowledgeSelection":
            return schema(
                facts=[
                    dict(
                        statement=statement,
                        code_path="main.go",
                        certainty="reported",
                        evidence_ids=["E1"],
                    )
                ]
            )
        if schema.__name__ == "KnowledgeAudit":
            return schema(issues=[])
        if schema.__name__ == "KnowledgeGraph":
            return schema(
                nodes=[dict(id="Message", name="Message", type="function", description=statement)],
                edges=[],
            )
        if schema.__name__ == "SummarizedContent":
            return schema(summary=statement)
        raise AssertionError(schema.__name__)

    async def data_rows():
        async with get_relational_engine().get_async_session() as session:
            return list(
                await session.scalars(select(Data).where(Data.dataset_id == binding.dataset_id))
            )

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    request = ReviewMemoryRequest(
        github_repository_id=repo,
        review_id=uuid4(),
        head_sha="a" * 40,
        lifecycle_generation=1,
        artifact_revision=1,
        content=statement,
    )
    llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "scoped-memory", "a" * 40), archive
        )
        if operation in {"update", "retry"}:
            assert (await remember_review(binding.organization_id, request)).status == "remember"
        if operation == "update":
            statement = "FRESH_REVIEW: Message in main.go still returns the payment message."
            request = request.model_copy(update={"artifact_revision": 2, "content": statement})
        async with scoped_database_context_variables(
            binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
        ):
            # This reproduces an old directory-loader manifest in the same
            # customer dataset. Its source directory is intentionally disposable.
            with validated_archive(archive) as repository:
                await cognee.add(str(repository), dataset_id=binding.dataset_id, user=user)
                disposed_repository = repository
            assert not disposed_repository.exists()
            await cognee.add(
                [
                    DataItem(
                        data=f"OLD_CORPUS_SENTINEL_{i}: unrelated legacy customer document.",
                        label=f"old-{i}.txt",
                    )
                    for i in range(3)
                ],
                dataset_id=binding.dataset_id,
                user=user,
            )
        before = await data_rows()
        legacy = [
            r
            for r in before
            if (r.system_metadata or {}).get("source") == "code_repo"
            or (r.label or "").startswith("old-")
        ]
        assert len(legacy) == 4
        prior_states = {r.id: deepcopy(r.pipeline_status) for r in legacy}
        if operation == "retry":
            async with get_relational_engine().get_async_session() as session:
                await set_weave_organization_scope(session, binding.organization_id)
                record = await session.get(
                    WeaveMemorySource,
                    (binding.organization_id, repo, f"review:qualified:{request.review_id}"),
                )
                # Staging failed after creating Data but before the native
                # memory completion receipt. Redelivery must take update().
                record.status = "processing_remember"
                await session.commit()
        calls.clear()
        result = await remember_review(binding.organization_id, request)
        assert result.status == ("remember" if operation == "remember" else "update")
        if operation == "retry":
            assert all(name != "KnowledgeSelection" for name, _ in calls)
        assert all("OLD_CORPUS_SENTINEL" not in text for _, text in calls)
        assert sum(name == "KnowledgeGraph" for name, _ in calls) == 1
        assert sum(name == "SummarizedContent" for name, _ in calls) == 1
        after = {r.id: r for r in await data_rows()}
        assert {key: after[key].pipeline_status for key in prior_states} == prior_states
        current = next(
            r
            for r in await source_records(binding, repo)
            if r.source_key == f"review:qualified:{request.review_id}"
        )
        assert current.status == "completed"
        assert current.data_id in after
        async with scoped_database_context_variables(binding.dataset_id, user.id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
        assert any(statement in str(p) for _, p in nodes)
        assert "OLD_CORPUS_SENTINEL" not in str(nodes)
    finally:
        await delete_organization(binding.organization_id, 2)


@pytest.mark.asyncio
async def test_cognify_rejects_empty_missing_and_foreign_data_scopes_without_processing(
    monkeypatch,
):
    import cognee
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.weave.organizations import provision_organization
    from cognee.tasks.ingestion.data_item import DataItem

    owner, foreign = await provision_organization(uuid4()), await provision_organization(uuid4())
    user, foreign_user = (
        await get_user(owner.service_user_id),
        await get_user(foreign.service_user_id),
    )
    own_id, foreign_id = uuid4(), uuid4()

    async def unexpected_model(*args, **kwargs):
        pytest.fail("An invalid source selection must never process any document")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", unexpected_model)
    try:
        await cognee.add(
            DataItem(data="Owned pending source content.", data_id=own_id),
            dataset_id=owner.dataset_id,
            user=user,
        )
        await cognee.add(
            DataItem(data="Foreign pending source content.", data_id=foreign_id),
            dataset_id=foreign.dataset_id,
            user=foreign_user,
        )
        for ids in ([], [uuid4()], [foreign_id], [own_id, foreign_id]):
            with pytest.raises(ValueError, match="data_ids"):
                await cognee.cognify(datasets=[owner.dataset_id], data_ids=ids, user=user)
        from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

        with pytest.raises(PermissionDeniedError):
            await cognee.cognify(datasets=[foreign.dataset_id], data_ids=[foreign_id], user=user)
        with pytest.raises(ValueError, match="explicit dataset"):
            await cognee.cognify(data_ids=[own_id], user=user)
    finally:
        await delete_organization(owner.organization_id, 2)
        await delete_organization(foreign.organization_id, 2)
