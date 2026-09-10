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
