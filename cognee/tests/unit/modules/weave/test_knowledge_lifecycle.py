from types import SimpleNamespace
from uuid import uuid4

from cognee.modules.weave.knowledge_lifecycle import (
    fact_digest,
    invalidate_receipt,
    affected_paths,
)


def test_invalidation_preserves_history_and_unrelated_facts():
    facts = [
        {"statement": "pay", "code_path": "pay.py"},
        {"statement": "read", "code_path": "read.py"},
    ]
    receipt = {
        "facts": facts,
        "fact_states": {
            fact_digest(facts[0]): {"status": "current", "checked_sha": "a" * 40}
        },
    }
    updated = invalidate_receipt(receipt, {"pay.py"}, "b" * 40)
    assert updated["fact_states"][fact_digest(facts[0])] == {
        "status": "needs_recheck",
        "checked_sha": "a" * 40,
        "invalidated_sha": "b" * 40,
    }
    assert fact_digest(facts[1]) not in updated["fact_states"]
    assert receipt["fact_states"][fact_digest(facts[0])]["status"] == "current"
    assert invalidate_receipt(updated, {"pay.py"}, "b" * 40) == updated
    assert (
        invalidate_receipt(updated, {"pay.py"}, "a" * 40)["fact_states"][
            fact_digest(facts[0])
        ]["invalidated_sha"]
        == "a" * 40
    )


def test_superseded_facts_do_not_become_pending_again():
    fact = {"statement": "old", "code_path": "pay.py"}
    receipt = {
        "facts": [fact],
        "fact_states": {fact_digest(fact): {"status": "superseded"}},
    }
    assert invalidate_receipt(receipt, {"pay.py"}, "b" * 40) == receipt


def test_impact_includes_dependents_without_crossing_customer_or_repository():
    org = uuid4()
    binding = SimpleNamespace(organization_id=org)

    def node(i, path, repo=42, owner=org):
        return i, {
            "type": "CodeSymbol",
            "file_path": path,
            "github_repository_id": repo,
            "organization_id": str(owner),
        }

    nodes = [
        node("pay", "pay.py"),
        node("api", "api.py"),
        node("foreign", "secret.py", owner=uuid4()),
        node("sibling", "other.py", repo=43),
    ]
    edges = [
        ("api", "pay", "calls", {}),
        ("foreign", "pay", "calls", {}),
        ("sibling", "pay", "calls", {}),
    ]
    assert affected_paths(binding, 42, nodes, edges, {"pay.py"}) == {"pay.py", "api.py"}


def test_post_stamp_retry_preserves_delta_but_returning_commit_gets_new_delta():
    from cognee.modules.weave.knowledge_lifecycle import change_receipt

    old = {
        "facts": [],
        "changed_paths": ["old.py"],
        "complete": False,
        "head_sha": "a" * 40,
    }
    assert change_receipt(old, {"a" * 40}, "a" * 40, set(), True) == old
    fresh = change_receipt(old, {"b" * 40}, "a" * 40, {"new.py"}, True)
    assert fresh["changed_paths"] == ["new.py"]
    assert fresh["complete"] is True
