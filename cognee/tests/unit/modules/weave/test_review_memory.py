from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError


def test_review_request_cannot_select_a_dataset_or_supply_unbounded_content():
    from cognee.modules.weave.contracts import ReviewMemoryRequest

    data = {
        "github_repository_id": 1,
        "review_id": str(uuid4()),
        "head_sha": "a" * 40,
        "lifecycle_generation": 1,
        "artifact_revision": 0,
        "content": "Final review",
    }
    assert ReviewMemoryRequest(**data).artifact_revision == 0
    for extra in (
        {"dataset_id": str(uuid4())},
        {"head_sha": "bad"},
        {"artifact_revision": -1},
        {"content": "x" * 500001},
    ):
        with pytest.raises(ValidationError):
            ReviewMemoryRequest(**(data | extra))


def test_review_revisions_ignore_delayed_events_and_reject_same_revision_conflicts():
    from cognee.modules.weave.memory_sources import stale_source_revision

    record = SimpleNamespace(artifact_revision=3, content_hash="new", status="processing")
    assert stale_source_revision(record, 2, "old")
    assert not stale_source_revision(record, 3, "new")  # retry incomplete native work
    assert not stale_source_revision(record, 4, "newer")
    assert not stale_source_revision(None, 0, "initial")
    with pytest.raises(ValueError, match="revision conflict"):
        stale_source_revision(record, 3, "different")


def test_review_never_reactivates_a_deleted_or_replaced_repository():
    from cognee.modules.weave.review_memory import repository_accepts_review

    current = SimpleNamespace(active=True, lifecycle_generation=12)
    assert repository_accepts_review(current, 12)
    assert not repository_accepts_review(current, 11)
    assert not repository_accepts_review(None, 12)
    assert not repository_accepts_review(SimpleNamespace(active=False, lifecycle_generation=12), 12)
    assert not repository_accepts_review(
        SimpleNamespace(active=True, lifecycle_generation=12, deletion_pending=True), 12
    )


def test_review_accepts_bounded_completed_sessions_without_dataset_selectors():
    from cognee.modules.weave.contracts import ReviewMemoryRequest

    session = dict(
        invocationId=str(uuid4()),
        sessionId=str(uuid4()),
        threadId=str(uuid4()),
        role="security",
        result="payment retry finding",
        truncated=False,
        steps=[dict(id=str(uuid4()), type="tool.completed", content="payment code")],
    )
    data = dict(
        github_repository_id=1,
        review_id=str(uuid4()),
        head_sha="a" * 40,
        lifecycle_generation=1,
        artifact_revision=0,
        content="Final review",
        sessions=[session],
    )
    parsed = ReviewMemoryRequest(**data)
    assert len(parsed.sessions) == 1
    for changed in (
        dict(dataset_id=str(uuid4())),
        dict(steps=[dict(id="x", type="tool.completed", content="x" * 64001)]),
    ):
        with pytest.raises(ValidationError):
            ReviewMemoryRequest(**(data | {"sessions": [session | changed]}))
