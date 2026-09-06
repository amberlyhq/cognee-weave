import asyncio
from uuid import uuid4

import pytest


def _request(**overrides):
    from cognee.modules.weave.indexing import IndexRequest

    values = {
        "organization_id": uuid4(),
        "github_repository_id": 1234,
        "repository_owner": "amberlyhq",
        "repository_name": "amberly",
        "default_branch": "main",
        "requested_sha": "a" * 40,
        "pipeline_version": "weave-v1",
        "extraction_version": "enola-v1",
    }
    values.update(overrides)
    return IndexRequest(**values)


@pytest.mark.parametrize("sha", ["", "a" * 39, "a" * 41, "g" * 40, "A" * 40])
def test_index_request_requires_lowercase_full_sha(sha):
    with pytest.raises(ValueError, match="40-character"):
        _request(requested_sha=sha)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("github_repository_id", 0),
        ("github_repository_id", 2**63),
        ("repository_owner", "owner/name"),
        ("repository_name", "../repo"),
        ("default_branch", "../main"),
    ],
)
def test_index_request_rejects_invalid_repository_identity(field, value):
    with pytest.raises(ValueError, match="invalid|positive"):
        _request(**{field: value})


@pytest.mark.asyncio
async def test_duplicate_delivery_creates_one_job():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    request = _request()

    jobs = await asyncio.gather(*(store.accept(request) for _ in range(12)))

    assert len({job.id for job in jobs}) == 1
    assert len(store.jobs) == 1


@pytest.mark.asyncio
async def test_duplicate_workers_can_claim_a_job_only_once():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    job = await store.accept(_request())

    claims = await asyncio.gather(*(store.claim(job.id) for _ in range(12)))

    assert claims.count(True) == 1
    assert claims.count(False) == 11


@pytest.mark.asyncio
async def test_failed_delivery_can_retry_same_job():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    request = _request()
    first = await store.accept(request)
    await store.fail(first.id, "archive_invalid")

    retry = await store.accept(request)

    assert retry.id == first.id
    assert retry.status == "queued"
    assert retry.attempt_count == 2
    assert retry.error_code is None


@pytest.mark.asyncio
async def test_crashed_running_delivery_can_be_reclaimed_under_the_operation_lock():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    request = _request()
    first = await store.accept(request)
    assert await store.claim(first.id)

    retry = await store.accept(request, reclaim_running=True)

    assert retry.id == first.id
    assert retry.status == "queued"
    assert retry.attempt_count == 2


@pytest.mark.asyncio
async def test_same_sha_with_new_pipeline_version_is_new_work():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    request_v1 = _request()
    request_v2 = _request(
        organization_id=request_v1.organization_id,
        pipeline_version="weave-v2",
    )

    job_v1 = await store.accept(request_v1)
    job_v2 = await store.accept(request_v2)

    assert job_v1.id != job_v2.id
    assert len(store.jobs) == 2
    assert store.current(request_v1).pipeline_version == "weave-v2"


@pytest.mark.asyncio
async def test_older_job_finishing_cannot_replace_newer_request():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    older = _request(requested_sha="1" * 40)
    newer = _request(organization_id=older.organization_id, requested_sha="2" * 40)

    older_job = await store.accept(older)
    newer_job = await store.accept(newer)
    await store.succeed(newer_job.id, newer.requested_sha)
    await store.succeed(older_job.id, older.requested_sha)

    current = store.current(newer)
    assert current.requested_sha == newer.requested_sha
    assert current.indexed_sha == newer.requested_sha
    assert current.status == "indexed"
    assert store.jobs[older_job.id].status == "succeeded"


@pytest.mark.asyncio
async def test_repository_identity_is_scoped_by_organization():
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    first = _request(organization_id=uuid4(), github_repository_id=99)
    second = _request(organization_id=uuid4(), github_repository_id=99)

    await store.accept(first)
    await store.accept(second)

    assert len(store.snapshots) == 2


def test_code_graph_provenance_never_persists_the_temporary_checkout_path():
    from cognee.tasks.code_graph.extract_code_graph import map_facts_to_data_points
    from cognee.tasks.code_graph.models import CodeRepository, RepositoryProvenance

    organization_id = uuid4()
    provenance = RepositoryProvenance(
        organization_id=organization_id,
        github_repository_id=1234,
        repository_owner="amberlyhq",
        repository_name="amberly",
        indexed_sha="a" * 40,
        pipeline_version="weave-v1",
        extraction_version="enola-v1",
    )
    points = map_facts_to_data_points(
        [{"kind": "symbol", "name": "review", "file": "src/review.ts"}],
        repo_path="/private/tmp/untrusted-checkout",
        repository_provenance=provenance,
    )

    repository = next(point for point in points if isinstance(point, CodeRepository))
    symbol = next(point for point in points if not isinstance(point, CodeRepository))
    expected_source = "github://amberlyhq/amberly@" + "a" * 40

    assert repository.path == expected_source
    assert repository.source_path == expected_source
    assert repository.organization_id == organization_id
    assert repository.github_repository_id == 1234
    assert repository.indexed_sha == "a" * 40
    assert symbol.source_path == "src/review.ts"
    assert symbol.fact_identity == "symbol:review"
    assert "/private/tmp" not in str(repository.model_dump())
    assert "/private/tmp" not in str(symbol.model_dump())


def test_weave_uses_openai_small_embeddings_through_openrouter_by_default(monkeypatch):
    from cognee.modules.weave.config import get_weave_embedding_config

    for name in (
        "WEAVE_EMBEDDING_PROVIDER",
        "WEAVE_EMBEDDING_MODEL",
        "WEAVE_EMBEDDING_DIMENSIONS",
        "WEAVE_EMBEDDING_ENDPOINT",
        "WEAVE_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    config = get_weave_embedding_config()

    assert config.embedding_provider == "openrouter"
    assert config.embedding_model == "openrouter/openai/text-embedding-3-small"
    assert config.embedding_dimensions == 1536
    assert config.embedding_endpoint == "https://openrouter.ai/api/v1"
    assert config.embedding_api_key == "test-openrouter-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_status", ["succeeded", "superseded", "failed", "queued"])
async def test_return_to_previous_commit_requeues_and_promotes_current_snapshot(previous_status):
    from cognee.modules.weave.indexing import InMemoryIndexStateStore

    store = InMemoryIndexStateStore()
    first = _request()
    second = _request(organization_id=first.organization_id, requested_sha="b" * 40)
    job_a = await store.accept(first)
    if previous_status == "succeeded":
        assert await store.claim(job_a.id)
        await store.succeed(job_a.id, first.requested_sha)
    elif previous_status == "failed":
        await store.fail(job_a.id, "indexing_failed")
    job_b = await store.accept(second)
    if previous_status == "superseded":
        assert not await store.claim(job_a.id)
    assert await store.claim(job_b.id)
    await store.succeed(job_b.id, second.requested_sha)

    returned = await store.accept(first)
    assert returned.status == "queued"
    assert store.current(first).requested_sha == first.requested_sha
    assert store.current(first).status == "queued"
    assert await store.claim(returned.id)
    await store.succeed(returned.id, first.requested_sha)
    assert store.current(first).indexed_sha == first.requested_sha
    assert store.current(first).status == "indexed"
    duplicate = await store.accept(first)
    assert duplicate.status == "succeeded"
    assert duplicate.attempt_count == returned.attempt_count
