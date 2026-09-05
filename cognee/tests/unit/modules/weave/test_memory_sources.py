"""Retry/source ownership policy around native memory, not extraction semantics."""

from types import SimpleNamespace
from uuid import uuid4

import pytest


def test_completed_unchanged_source_is_a_noop():
    from cognee.modules.weave.memory_sources import source_action

    record = SimpleNamespace(status="completed", content_hash="abc")
    assert source_action(record, "abc", exists=True) == "unchanged"
    assert source_action(record, "changed", exists=True) == "update"


def test_partial_updates_are_replaced_and_missing_items_are_recreated():
    from cognee.modules.weave.memory_sources import source_action

    record = SimpleNamespace(status="processing", content_hash="abc")
    assert source_action(record, "abc", exists=True) == "update"
    assert source_action(record, "abc", exists=False) == "remember"
    assert source_action(None, "abc", exists=False) == "remember"


def test_native_source_ids_separate_customers_repositories_and_paths():
    from cognee.modules.weave.memory_sources import source_data_id

    a, b = uuid4(), uuid4()
    identities = [
        source_data_id(a, 1, "code"),
        source_data_id(a, 2, "code"),
        source_data_id(b, 1, "code"),
        source_data_id(a, 1, "doc:README.md"),
    ]
    assert len(set(identities)) == 4
    assert source_data_id(a, 1, "code") == identities[0]


@pytest.mark.parametrize("status", ["PipelineRunErrored", "PipelineRunStarted", "failed"])
def test_failed_or_partial_native_results_cannot_publish_a_source_receipt(status):
    from cognee.modules.weave.memory_sources import assert_native_completed

    with pytest.raises(RuntimeError):
        assert_native_completed({uuid4(): SimpleNamespace(status=status)})


def test_nested_native_pipeline_failures_are_not_hidden_by_parent_completion():
    from cognee.modules.weave.memory_sources import assert_native_completed

    with pytest.raises(RuntimeError):
        assert_native_completed(
            {
                uuid4(): SimpleNamespace(
                    status="PipelineRunCompleted",
                    data_ingestion_info=[
                        {"run_info": SimpleNamespace(status="PipelineRunErrored")}
                    ],
                )
            }
        )


def test_native_completed_and_cached_results_are_accepted():
    from cognee.modules.weave.memory_sources import assert_native_completed

    assert_native_completed(SimpleNamespace(status="completed"))
    assert_native_completed({uuid4(): SimpleNamespace(status="PipelineRunAlreadyCompleted")})


def test_shared_memory_requires_all_repository_receipts_even_when_primary_is_ready():
    from cognee.modules.weave.native_memory import NATIVE_PIPELINE_VERSION, snapshots_ready

    ready = SimpleNamespace(
        status="indexed",
        pipeline_version=NATIVE_PIPELINE_VERSION,
        requested_sha="a" * 40,
        indexed_sha="a" * 40,
    )
    pending = SimpleNamespace(
        status="running",
        pipeline_version=NATIVE_PIPELINE_VERSION,
        requested_sha="b" * 40,
        indexed_sha=None,
    )
    assert not snapshots_ready([ready, pending])


@pytest.mark.asyncio
async def test_native_repo_sources_keep_manifest_granularity_and_stable_fingerprints(
    tmp_path, monkeypatch
):
    from cognee.modules.weave.repository_sources import prepare_repository_sources

    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.setattr(
        "cognee.infrastructure.llm.config.get_llm_config",
        lambda: SimpleNamespace(llm_api_key="offline-key"),
    )
    request = SimpleNamespace(
        repository_owner="amberlyhq", repository_name="api", github_repository_id=123
    )
    fingerprints = []
    for folder in ("archive-a", "archive-b"):
        repo = tmp_path / folder / "sha-dependent-root"
        repo.mkdir(parents=True)
        (repo / "package.json").write_text('{"name":"api"}')
        (repo / "index.js").write_text("export function customer() { return 1; }")
        (repo / "README.md").write_text("Customer API documentation.")
        (repo / ".env").write_text("DO_NOT_INGEST=secret")
        sources = await prepare_repository_sources(repo, request, user=None, dataset_id=uuid4())
        assert len(sources) == 2
        assert sources[0][0] == "code"
        assert sources[0][1].system_metadata["source"] == "code_repo"
        assert sources[1][1].external_metadata["source_path"] == "README.md"
        fingerprints.append([(key, fingerprint) for key, _, fingerprint in sources])
    assert fingerprints[0] == fingerprints[1]


def test_shared_recall_can_disclose_more_than_twenty_repository_receipts():
    from cognee.modules.weave.contracts import RecallResponse, RepositoryReference

    repositories = [
        RepositoryReference(
            github_repository_id=i, repository_owner="amberlyhq", repository_name=f"repo-{i}"
        )
        for i in range(1, 22)
    ]
    assert (
        len(
            RecallResponse(
                status="available",
                organization_id=uuid4(),
                mode="repository_context",
                repositories=repositories,
            ).repositories
        )
        == 21
    )
