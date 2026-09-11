from types import SimpleNamespace
from pathlib import Path
import pytest

from cognee.modules.weave.contracts import MergeKnowledgeChange
from cognee.modules.weave.merge_memory import merge_readiness, build_batches


def test_change_paths_cannot_escape_archive():
    for path in ["../secret", "/etc/passwd", "a/../b", "a\\b", "C:/secret"]:
        with pytest.raises(ValueError):
            MergeKnowledgeChange(
                base_sha="a" * 40,
                head_sha="b" * 40,
                changed_paths=[path],
                complete=True,
            )


def test_only_exact_current_snapshot_can_learn():
    req = SimpleNamespace(
        requested_sha="b" * 40,
        repository_owner="acme",
        repository_name="code",
        default_branch="main",
        lifecycle_generation=1,
    )
    lifecycle = SimpleNamespace(
        active=True, deletion_pending=False, lifecycle_generation=1
    )
    snapshot = SimpleNamespace(
        status="indexed",
        requested_sha="b" * 40,
        indexed_sha="b" * 40,
        repository_owner="acme",
        repository_name="code",
        default_branch="main",
    )
    assert merge_readiness(req, lifecycle, snapshot) is None
    snapshot.indexed_sha = "a" * 40
    assert merge_readiness(req, lifecycle, snapshot) == "pending"
    snapshot.requested_sha = "c" * 40
    assert merge_readiness(req, lifecycle, snapshot) == "stale_ignored"
    lifecycle.active = False
    assert merge_readiness(req, lifecycle, snapshot) == "stale_ignored"


def test_batches_are_bounded_and_keep_all_prior_facts(tmp_path):
    (tmp_path / "a.py").write_text("x" * 15000)
    (tmp_path / "b.py").write_text("y" * 15000)
    prior = [{"fact_id": str(i), "code_path": "a.py"} for i in range(30)]
    batches = build_batches(tmp_path, ["a.py", "b.py", "gone.py"], prior)
    assert all(len(b["prior_facts"]) <= 12 for b in batches)
    assert {f["fact_id"] for b in batches for f in b["prior_facts"]} == {
        str(i) for i in range(30)
    }
    assert {p for b in batches for p in b["paths"]} == {"a.py", "b.py", "gone.py"}
