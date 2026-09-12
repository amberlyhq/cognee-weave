"""Real Postgres/native memory; only paid model responses and embeddings are fixtures."""

import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


@pytest.mark.asyncio
async def test_host_selected_native_notes_update_retire_replay_isolation_and_code_links(
    tmp_path, monkeypatch
):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.weave.agent_memory import apply_memory_notes, list_memory_notes
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.deletion import delete_organization
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding, canary = await provision_organization(uuid4()), await provision_organization(uuid4())
    repo = 960001
    calls = []

    async def model(*args, **kwargs):
        cls = kwargs.get("response_model") or args[2]
        calls.append(cls.__name__)
        if cls.__name__ == "KnowledgeGraph":
            return cls(
                nodes=[
                    dict(
                        id="MessagePayment",
                        name="MessagePayment",
                        type="function",
                        description="Returns a payment message",
                    )
                ],
                edges=[],
            )
        if cls.__name__ == "SummarizedContent":
            return cls(summary="MessagePayment returns a payment message.")
        raise AssertionError(f"Weave must not qualify host results: {cls.__name__}")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)

    async def index(binding, sha, message):
        return await index_repository_archive(
            _request(binding.organization_id, repo, "payment", sha),
            _repository_archive(tmp_path, "payment", message),
        )

    def request(action, note_id, version, content=None, sha="b" * 40, kind="review"):
        return MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind=kind,
            source_id="local-e2e",
            source_sha=sha,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=note_id,
                    action=action,
                    expected_version=version,
                    content=content,
                    source_paths=["main.go"],
                    reason="Agent selected",
                    provenance={"review_id": "e2e"},
                )
            ],
        )

    try:
        await index(binding, "a" * 40, "payment")
        await index(canary, "a" * 40, "payment")
        note_id = uuid4()
        add = request("add", note_id, 0, "MessagePayment returns a payment message.")
        first = await apply_memory_notes(binding.organization_id, repo, add)
        assert first.receipts[0].version == 1
        initial_calls = list(calls)
        assert await apply_memory_notes(binding.organization_id, repo, add) == first
        assert calls == initial_calls
        notes = await list_memory_notes(binding.organization_id, repo)
        assert notes.notes[0].content == add.operations[0].content
        assert not (await list_memory_notes(canary.organization_id, repo)).notes
        from sqlalchemy import select
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.weave.models import (
            WeaveMemoryNote,
            WeaveMemoryJob,
            WeaveMemoryOperation,
        )
        from cognee.modules.weave.organizations import set_weave_organization_scope

        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, canary.organization_id)
            for table in (WeaveMemoryNote, WeaveMemoryJob, WeaveMemoryOperation):
                assert not list(await session.scalars(select(table)))

        with pytest.raises(LookupError, match="missing|version"):
            await apply_memory_notes(
                canary.organization_id, repo, request("update", note_id, 1, "Foreign overwrite")
            )
        with pytest.raises(LookupError, match="payload conflict"):
            await apply_memory_notes(
                binding.organization_id, repo, add.model_copy(update={"source_id": "changed"})
            )
        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            nodes, edges = await (await get_graph_engine()).get_graph_data()
        assert not any(p.get("type") == "ReviewKnowledge" for _, p in nodes)
        assert any(r == "memory_context_for" for a, b, r, p in edges)
        from cognee.modules.weave.memory_retrieval import recall_current_memory
        from cognee.modules.weave.config import get_weave_llm_config, get_weave_embedding_config
        from cognee.modules.users.methods import get_user

        async with scoped_database_context_variables(binding.dataset_id, binding.service_user_id):
            recalled = await recall_current_memory(
                binding,
                await get_user(binding.service_user_id),
                "MessagePayment",
                await source_records(binding, repo),
                repository_ids=[repo],
                top_k=5,
                llm=get_weave_llm_config(),
                embedding=get_weave_embedding_config(),
            )
        assert recalled
        assert "payment message" in str(recalled)
        sources = await source_records(binding, repo)
        data_id = next(s.data_id for s in sources if s.source_key == f"review:note:{note_id}")
        with pytest.raises(LookupError, match="stale"):
            await apply_memory_notes(
                binding.organization_id,
                repo,
                request("update", note_id, 1, "Updated payment", kind="default_branch"),
            )
        await index(binding, "b" * 40, "updated")
        update = request(
            "update",
            note_id,
            1,
            "MessagePayment now returns an updated message.",
            kind="default_branch",
        )
        assert (await apply_memory_notes(binding.organization_id, repo, update)).receipts[
            0
        ].version == 2
        assert (
            next(
                s.data_id
                for s in await source_records(binding, repo)
                if s.source_key == f"review:note:{note_id}"
            )
            == data_id
        )
        await index(binding, "a" * 40, "payment")
        revert = request(
            "update",
            note_id,
            2,
            "MessagePayment returns the original payment message.",
            sha="a" * 40,
            kind="default_branch",
        )
        assert (await apply_memory_notes(binding.organization_id, repo, revert)).receipts[
            0
        ].version == 3
        before = list(calls)
        unchanged = request("unchanged", note_id, 3)
        assert (await apply_memory_notes(binding.organization_id, repo, unchanged)).receipts[
            0
        ].version == 3
        assert calls == before
        retire = request("retire", note_id, 3)
        assert (await apply_memory_notes(binding.organization_id, repo, retire)).receipts[
            0
        ].version == 4
        assert await apply_memory_notes(binding.organization_id, repo, retire)
        assert not [
            s
            for s in await source_records(binding, repo)
            if s.source_key == f"review:note:{note_id}"
        ]
        assert (await list_memory_notes(binding.organization_id, repo)).notes[0].status == "retired"
        assert not (
            await list_memory_notes(binding.organization_id, repo, include_retired=False)
        ).notes
    finally:
        await delete_organization(binding.organization_id, 100)
        await delete_organization(canary.organization_id, 100)


@pytest.mark.asyncio
async def test_failed_native_job_resumes_without_regeneration_and_stale_job_is_superseded(
    tmp_path, monkeypatch
):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.weave import note_code_links
    from cognee.modules.weave.agent_memory import (
        apply_memory_notes,
        list_memory_notes,
        MemoryConflict,
    )
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.deletion import delete_organization
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    repo, note_id, calls = 960002, uuid4(), []

    async def model(*args, **kwargs):
        cls = kwargs.get("response_model") or args[2]
        calls.append(cls.__name__)
        if cls.__name__ == "KnowledgeGraph":
            return cls(nodes=[], edges=[])
        if cls.__name__ == "SummarizedContent":
            return cls(summary="Payment behavior")
        raise AssertionError(cls.__name__)

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)

    async def index(sha):
        await index_repository_archive(
            _request(binding.organization_id, repo, "payment", sha),
            _repository_archive(tmp_path, "payment", "payment"),
        )

    def request(action, version, content, sha="a" * 40):
        return MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="default_branch",
            source_id="local-failure",
            source_sha=sha,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=note_id,
                    action=action,
                    expected_version=version,
                    content=content,
                    source_paths=["main.go"],
                    reason="agent choice",
                )
            ],
        )

    original_links = note_code_links.sync_note_code_links

    async def failed_links(*args, **kwargs):
        raise RuntimeError("simulated process failure after native remember")

    try:
        await index("a" * 40)
        add = request("add", 0, "Payment returns a message.")
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failed_links)
        with pytest.raises(RuntimeError, match="simulated"):
            await apply_memory_notes(binding.organization_id, repo, add)
        assert not (await list_memory_notes(binding.organization_id, repo)).notes
        source = next(
            s for s in await source_records(binding, repo) if s.source_key.endswith(str(note_id))
        )
        assert source.status == "processing_note"
        before = list(calls)
        with pytest.raises(MemoryConflict, match="unfinished"):
            await apply_memory_notes(
                binding.organization_id, repo, request("add", 0, "Competing result.")
            )
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        first = await apply_memory_notes(binding.organization_id, repo, add)
        assert calls == before
        assert first.receipts[0].version == 1
        update = request("update", 1, "Payment returns an enhanced message.")
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failed_links)
        with pytest.raises(RuntimeError):
            await apply_memory_notes(binding.organization_id, repo, update)
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        await index("b" * 40)
        with pytest.raises(MemoryConflict) as stale:
            await apply_memory_notes(binding.organization_id, repo, update)
        assert stale.value.code == "stale_source"
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        from cognee.modules.weave.agent_memory_supersession import supersede_memory_job

        assert (
            await supersede_memory_job(binding.organization_id, repo, update.job_id, 1)
        ).status == "superseded"
        replacement = request("update", 1, "The current source returns a newer message.", "b" * 40)
        assert (await apply_memory_notes(binding.organization_id, repo, replacement)).receipts[
            0
        ].version == 2
        await index("a" * 40)
        with pytest.raises(MemoryConflict, match="superseded"):
            await apply_memory_notes(binding.organization_id, repo, update)
        # Completed receipts remain replayable after source moves, with no rewrite.
        before = list(calls)
        assert await apply_memory_notes(binding.organization_id, repo, add) == first
        assert calls == before
        assert (await list_memory_notes(binding.organization_id, repo)).notes[
            0
        ].content == replacement.operations[0].content
        replacement_note = uuid4()
        replace_job = MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="review",
            source_id="replacement-order",
            source_sha="c" * 40,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=note_id,
                    action="retire",
                    expected_version=2,
                    reason="Replaced by consolidated note",
                ),
                dict(
                    operation_id=uuid4(),
                    note_id=replacement_note,
                    action="add",
                    expected_version=0,
                    content="Consolidated payment behavior.",
                    source_paths=["main.go"],
                    reason="Consolidation",
                ),
            ],
        )
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failed_links)
        with pytest.raises(RuntimeError):
            await apply_memory_notes(binding.organization_id, repo, replace_job)
        assert any(
            note.note_id == note_id and note.status == "active"
            for note in (await list_memory_notes(binding.organization_id, repo)).notes
        )
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        await apply_memory_notes(binding.organization_id, repo, replace_job)
        final_notes = (await list_memory_notes(binding.organization_id, repo)).notes
        assert {note.note_id: note.status for note in final_notes} == {
            note_id: "retired",
            replacement_note: "active",
        }
    finally:
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        await delete_organization(binding.organization_id, 100)


@pytest.mark.asyncio
async def test_legacy_native_source_becomes_a_full_stable_note_without_qualification(
    tmp_path, monkeypatch
):
    import hashlib
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.weave.agent_memory import apply_memory_notes, list_memory_notes
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest
    from cognee.modules.weave.organizations import (
        provision_organization,
        set_weave_organization_scope,
    )
    from cognee.modules.weave.indexing import index_repository_archive, weave_operation_lock
    from cognee.modules.weave.memory_sources import sync_source, source_data_id, source_records
    from cognee.modules.weave.models import WeaveMemorySource
    from cognee.modules.weave.config import get_weave_llm_config, get_weave_embedding_config
    from cognee.modules.weave.deletion import delete_organization
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.scope import native_organization
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    repo, calls = 960003, []

    async def model(*args, **kwargs):
        cls = kwargs.get("response_model") or args[2]
        calls.append(cls.__name__)
        if cls.__name__ == "KnowledgeGraph":
            return cls(nodes=[], edges=[])
        if cls.__name__ == "SummarizedContent":
            return cls(summary="Legacy payment knowledge")
        raise AssertionError(cls.__name__)

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "payment", "a" * 40),
            _repository_archive(tmp_path, "payment", "payment"),
        )
        user = await get_user(binding.service_user_id)
        key = f"review:qualified:{uuid4()}"
        identity = (binding.organization_id, repo, key)
        content = "Legacy full note: payment returns a message.\nSource evidence remains attached."
        item = DataItem(
            data=content,
            label="legacy.txt",
            external_metadata={
                "memory_kind": "qualified_review",
                "github_repository_id": repo,
                "head_sha": "a" * 40,
            },
        )
        llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
        token = native_organization.set(binding.organization_id)
        try:
            async with weave_operation_lock(binding.organization_id):
                async with scoped_database_context_variables(
                    binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
                ):
                    await sync_source(
                        binding,
                        user,
                        repo,
                        key,
                        item,
                        hashlib.sha256(content.encode()).hexdigest(),
                        llm=llm,
                        embedding=embedding,
                        artifact_revision=3,
                        self_improvement=False,
                    )
                    async with get_relational_engine().get_async_session() as session:
                        await set_weave_organization_scope(session, binding.organization_id)
                        source = await session.get(WeaveMemorySource, identity)
                        source.dataset_id = binding.dataset_id
                        source.qualification = {
                            "facts": [
                                {
                                    "statement": "Payment returns a message.",
                                    "code_path": "main.go",
                                    "certainty": "observed",
                                    "evidence": [],
                                }
                            ]
                        }
                        await session.commit()
        finally:
            native_organization.reset(token)
        before = list(calls)
        notes = await list_memory_notes(binding.organization_id, repo)
        assert calls == before
        assert notes.notes[0].content == content
        assert notes.notes[0].note_id == source_data_id(*identity)
        assert notes.notes[0].version == 3
        from cognee.modules.weave import note_code_links
        from cognee.modules.weave.agent_memory_supersession import supersede_memory_job
        from cognee.modules.weave.memory_retrieval import eligible_memory_sources

        original_links = note_code_links.sync_note_code_links

        async def failed_legacy_links(*args, **kwargs):
            raise RuntimeError("interrupted legacy promotion")

        aborted = MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="review",
            source_id="obsolete-legacy-promotion",
            source_sha="b" * 40,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=notes.notes[0].note_id,
                    expected_version=3,
                    action="update",
                    content="Interrupted legacy rewrite.",
                    source_paths=["main.go"],
                    reason="Synthetic failure",
                )
            ],
        )
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failed_legacy_links)
        with pytest.raises(RuntimeError, match="interrupted legacy"):
            await apply_memory_notes(binding.organization_id, repo, aborted)
        monkeypatch.setattr(note_code_links, "sync_note_code_links", original_links)
        await supersede_memory_job(binding.organization_id, repo, aborted.job_id, 1)
        restored_legacy = next(
            source for source in await source_records(binding, repo) if source.source_key == key
        )
        assert not restored_legacy.qualification.get("memory_note")
        assert not eligible_memory_sources(binding, [restored_legacy], [repo])
        assert (await list_memory_notes(binding.organization_id, repo)).notes[0].version == 3
        req = MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="review",
            source_id="legacy-migration-test",
            source_sha="b" * 40,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=notes.notes[0].note_id,
                    expected_version=3,
                    action="update",
                    content="Agent consolidated payment knowledge.",
                    source_paths=["main.go"],
                    reason="Consolidated legacy note",
                )
            ],
        )
        result = await apply_memory_notes(binding.organization_id, repo, req)
        assert result.receipts[0].version == 4
        sources = [
            s
            for s in await source_records(binding, repo)
            if s.source_key.startswith("review:qualified:")
        ]
        assert len(sources) == 1 and sources[0].data_id == notes.notes[0].note_id
        assert sources[0].qualification["memory_note"] is True
        assert (await list_memory_notes(binding.organization_id, repo)).notes[
            0
        ].content == "Agent consolidated payment knowledge."
        from cognee.modules.weave import agent_memory
        from cognee.modules.weave.agent_memory_supersession import supersede_memory_job

        real_forget = agent_memory.forget_source

        async def interrupted_forget(*args, **kwargs):
            await real_forget(*args, **kwargs)
            raise RuntimeError("interrupt after native retirement")

        retire = req.model_copy(
            update={
                "job_id": uuid4(),
                "operations": [
                    req.operations[0].model_copy(
                        update={
                            "operation_id": uuid4(),
                            "action": "retire",
                            "expected_version": 4,
                            "content": None,
                        }
                    )
                ],
            }
        )
        monkeypatch.setattr(agent_memory, "forget_source", interrupted_forget)
        with pytest.raises(RuntimeError, match="interrupt after"):
            await apply_memory_notes(binding.organization_id, repo, retire)
        monkeypatch.setattr(agent_memory, "forget_source", real_forget)
        assert not [
            source for source in await source_records(binding, repo) if source.source_key == key
        ]
        restored = await supersede_memory_job(binding.organization_id, repo, retire.job_id, 1)
        assert restored.status == "superseded"
        assert (
            await supersede_memory_job(binding.organization_id, repo, retire.job_id, 1) == restored
        )
        source = next(
            source for source in await source_records(binding, repo) if source.source_key == key
        )
        assert source.data_id == notes.notes[0].note_id
        committed = (await list_memory_notes(binding.organization_id, repo)).notes[0]
        assert committed.status == "active" and committed.version == 4
        assert committed.content == "Agent consolidated payment knowledge."
    finally:
        await delete_organization(binding.organization_id, 100)


@pytest.mark.asyncio
async def test_superseded_review_restores_committed_note_and_unblocks_new_revision(
    tmp_path, monkeypatch
):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.weave import note_code_links, agent_memory_supersession
    from cognee.modules.weave.agent_memory import (
        apply_memory_notes,
        list_memory_notes,
        MemoryConflict,
    )
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest
    from cognee.modules.weave.agent_memory_supersession import supersede_memory_job
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.indexing import index_repository_archive
    from cognee.modules.weave.memory_sources import source_records
    from cognee.modules.weave.deletion import delete_organization
    from cognee.tests.e2e.postgres.test_weave_exact_sha_indexing import (
        _repository_archive,
        _request,
    )

    binding = await provision_organization(uuid4())
    repo = 960004
    note_id = uuid4()
    calls = []

    async def model(*args, **kwargs):
        cls = kwargs.get("response_model") or args[2]
        calls.append(cls.__name__)
        if cls.__name__ == "KnowledgeGraph":
            return cls(nodes=[], edges=[])
        if cls.__name__ == "SummarizedContent":
            return cls(summary="Payment behavior")
        raise AssertionError(cls.__name__)

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)

    def request(action, version, content, target=note_id):
        return MemoryApplyRequest(
            job_id=uuid4(),
            lifecycle_generation=1,
            source_kind="review",
            source_id="review-revision",
            source_sha="b" * 40,
            operations=[
                dict(
                    operation_id=uuid4(),
                    note_id=target,
                    action=action,
                    expected_version=version,
                    content=content,
                    source_paths=["main.go"],
                    reason="Agent selection",
                )
            ],
        )

    async def failure(*args, **kwargs):
        raise RuntimeError("simulated link failure")

    real_links = note_code_links.sync_note_code_links
    try:
        await index_repository_archive(
            _request(binding.organization_id, repo, "payment", "a" * 40),
            _repository_archive(tmp_path, "payment", "payment"),
        )
        add = request("add", 0, "Committed payment knowledge.")
        await apply_memory_notes(binding.organization_id, repo, add)
        assert (
            await supersede_memory_job(binding.organization_id, repo, add.job_id, 1)
        ).status == "completed"
        original = next(
            s for s in await source_records(binding, repo) if s.source_key.endswith(str(note_id))
        )
        update = request("update", 1, "Obsolete interrupted review knowledge.")
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failure)
        with pytest.raises(RuntimeError):
            await apply_memory_notes(binding.organization_id, repo, update)
        monkeypatch.setattr(note_code_links, "sync_note_code_links", real_links)
        # Rollback can fail too; its durable state must block competing writes.
        monkeypatch.setattr(agent_memory_supersession, "sync_note_code_links", failure)
        with pytest.raises(RuntimeError):
            await supersede_memory_job(binding.organization_id, repo, update.job_id, 1)
        with pytest.raises(MemoryConflict, match="unfinished"):
            await apply_memory_notes(
                binding.organization_id, repo, request("update", 1, "New revision knowledge.")
            )
        before = list(calls)
        monkeypatch.setattr(agent_memory_supersession, "sync_note_code_links", real_links)
        result = await supersede_memory_job(binding.organization_id, repo, update.job_id, 1)
        assert result.status == "superseded" and calls == before
        assert await supersede_memory_job(binding.organization_id, repo, update.job_id, 1) == result
        note = (await list_memory_notes(binding.organization_id, repo)).notes[0]
        assert note.version == 1 and note.content == "Committed payment knowledge."
        restored = next(
            s for s in await source_records(binding, repo) if s.source_key.endswith(str(note_id))
        )
        assert (
            restored.data_id == original.data_id and restored.content_hash == original.content_hash
        )
        assert (
            await apply_memory_notes(
                binding.organization_id, repo, request("update", 1, "New revision knowledge.")
            )
        ).receipts[0].version == 2
        # Provisional new notes are forgotten rather than made permanent by cancellation.
        provisional_id = uuid4()
        provisional = request("add", 0, "Provisional never-committed knowledge.", provisional_id)
        monkeypatch.setattr(note_code_links, "sync_note_code_links", failure)
        with pytest.raises(RuntimeError):
            await apply_memory_notes(binding.organization_id, repo, provisional)
        monkeypatch.setattr(note_code_links, "sync_note_code_links", real_links)
        await supersede_memory_job(binding.organization_id, repo, provisional.job_id, 1)
        assert not [
            s
            for s in await source_records(binding, repo)
            if s.source_key.endswith(str(provisional_id))
        ]
        assert len((await list_memory_notes(binding.organization_id, repo)).notes) == 1
        # Cancellation before first delivery leaves a tombstone against delayed events.
        late = request("add", 0, "Delayed obsolete knowledge.", uuid4())
        await supersede_memory_job(binding.organization_id, repo, late.job_id, 1)
        with pytest.raises(MemoryConflict, match="superseded"):
            await apply_memory_notes(binding.organization_id, repo, late)
    finally:
        await delete_organization(binding.organization_id, 100)
