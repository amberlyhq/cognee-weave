"""Replacing part of a source must preserve its other live facts."""

from copy import deepcopy

import pytest

from cognee.modules.weave.knowledge_lifecycle import fact_digest
from cognee.modules.weave.memory_retrieval import eligible_memory_sources, native_source_tag
from cognee.modules.weave.memory_sources import source_data_id
from cognee.tests.unit.modules.weave.test_merge_memory_retry import (
    merge_case as merge_case,
    merge_records,
    run_merge,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["changed", "needs_recheck"])
@pytest.mark.parametrize("live_count", [2, 15])
async def test_affected_bundle_requalifies_all_live_facts_in_bounded_batches(
    merge_case, monkeypatch, trigger, live_count
):
    from cognee.modules.weave import merge_qualification

    case = merge_case
    original = case.store.sources[(case.binding.organization_id, 42, case.previous_key)]
    facts = [case.fact] + [
        {**deepcopy(case.fact), "statement": f"Untouched fact {index}", "code_path": "stable.py"}
        for index in range(1, live_count)
    ]
    superseded = {**deepcopy(case.fact), "statement": "Already replaced historical fact"}
    original.qualification = {
        "facts": [*facts, superseded],
        "fact_states": {
            fact_digest(fact): {"status": "current", "checked_sha": "a" * 40} for fact in facts
        },
    }
    original.qualification["fact_states"][fact_digest(superseded)] = {"status": "superseded"}
    if trigger == "needs_recheck":
        case.change.changed_paths = []
        original.qualification["fact_states"][fact_digest(case.fact)]["status"] = "needs_recheck"
    (case.repository / "stable.py").write_text("def stable(user):\n    return user.is_active\n")

    # An independent source with no affected facts is not requalified.
    untouched = deepcopy(original)
    untouched.source_key = "review:qualified:independent"
    untouched.data_id = source_data_id(case.binding.organization_id, 42, untouched.source_key)
    untouched_fact = {**deepcopy(facts[1]), "statement": "Independent unchanged source"}
    untouched.qualification = {
        "facts": [untouched_fact],
        "fact_states": {
            fact_digest(untouched_fact): {"status": "current", "checked_sha": "a" * 40}
        },
        "native_source_tag": native_source_tag(untouched.data_id, untouched.content_hash),
    }
    case.store.sources[(case.binding.organization_id, 42, untouched.source_key)] = untouched

    async def qualify(**kwargs):
        case.qualification_calls.append(deepcopy(kwargs))
        return {
            "facts": [
                {key: fact[key] for key in ("statement", "code_path", "certainty", "evidence")}
                for fact in kwargs["prior_facts"]
            ],
            "decisions": [
                {
                    "fact_id": fact["fact_id"],
                    "status": "keep",
                    "reason": "Supported by current code",
                }
                for fact in kwargs["prior_facts"]
            ],
            "coverage_complete": True,
        }

    monkeypatch.setattr(merge_qualification, "qualify_merge", qualify)
    result = await run_merge(case)
    selected = [fact for call in case.qualification_calls for fact in call["prior_facts"]]
    assert {fact["statement"] for fact in selected} == {fact["statement"] for fact in facts}
    assert all(len(call["prior_facts"]) <= 12 for call in case.qualification_calls)
    assert all(
        set(fact["code_path"] for fact in call["prior_facts"]) <= set(call["files"])
        for call in case.qualification_calls
    )
    assert result["status"] == "completed" and result["facts_rechecked"] == live_count
    current = eligible_memory_sources(case.binding, list(case.store.sources.values()), {42})
    current_statements = {
        fact["statement"] for record in current for fact in record.qualification["facts"]
    }
    assert current_statements == {fact["statement"] for fact in [*facts, untouched_fact]}
    saved_original = case.store.sources[(case.binding.organization_id, 42, case.previous_key)]
    assert all(
        state["status"] == "superseded"
        for state in saved_original.qualification["fact_states"].values()
    )
    assert all(record.dataset_id == case.binding.dataset_id for record in merge_records(case))
    calls = len(case.qualification_calls), len(case.native_calls)
    assert (await run_merge(case))["status"] == "unchanged"
    assert (len(case.qualification_calls), len(case.native_calls)) == calls
