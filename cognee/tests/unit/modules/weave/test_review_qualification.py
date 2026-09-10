from uuid import uuid4

import pytest

from cognee.modules.weave.contracts import ReviewMemoryRequest


def request(**updates):
    return ReviewMemoryRequest(
        **(
            dict(
                github_repository_id=1,
                review_id=uuid4(),
                head_sha="a" * 40,
                lifecycle_generation=1,
                artifact_revision=1,
                content="payments/retry.ts reuses the payment id when retrying a request.",
            )
            | updates
        )
    )


def fact(**updates):
    return (
        dict(
            statement="Payment retries reuse the payment id.",
            code_path="payments/retry.ts",
            certainty="reported",
            evidence=[dict(evidence_id="final", quote="payments/retry.ts reuses the payment id")],
        )
        | updates
    )


def test_qualification_rejects_invented_evidence_and_paths():
    from cognee.modules.weave.review_qualification import (
        Qualification,
        evidence_packet,
        validate_qualification,
    )

    packet = evidence_packet(request())
    validate_qualification(Qualification(facts=[fact()]), packet)
    for changed in (
        dict(code_path="invented.ts"),
        dict(evidence=[dict(evidence_id="foreign", quote="payments/retry.ts")]),
        dict(evidence=[dict(evidence_id="final", quote="Payment always succeeds")]),
    ):
        with pytest.raises(ValueError):
            validate_qualification(Qualification(facts=[fact(**changed)]), packet)
    validate_qualification(Qualification(facts=[]), packet)


def test_packet_is_bounded_and_keeps_results_before_trace_detail():
    from cognee.modules.weave.review_qualification import evidence_packet

    sessions = [
        dict(
            invocationId=str(uuid4()),
            sessionId=str(uuid4()),
            threadId=str(uuid4()),
            role="security",
            result="important result",
            steps=[dict(id=str(i), type="tool.completed", content="x" * 60000)],
        )
        for i in range(32)
    ]
    packet = evidence_packet(request(content="y" * 500000, sessions=sessions))
    assert sum(len(v) for v in packet.values()) <= 120000
    assert all(
        "important result" in packet[f"session:{entry['invocationId']}:result"]
        for entry in sessions
    )


def selection_fact(**updates):
    return (
        dict(
            statement=fact()["statement"],
            code_path="payments/retry.ts",
            certainty="reported",
            evidence_ids=["E1"],
        )
        | updates
    )


@pytest.mark.asyncio
async def test_native_gateway_selects_only_useful_passages(monkeypatch):
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
        render_knowledge,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    async def model(**kwargs):
        if kwargs["response_model"].__name__ == "KnowledgeAudit":
            return kwargs["response_model"](issues=[])
        assert kwargs["response_model"] is KnowledgeSelection
        assert "generic" in kwargs["system_prompt"]
        assert "untrusted" in kwargs["system_prompt"]
        return KnowledgeSelection(facts=[selection_fact()])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    req = request(content=request().content + " Always use TodoWrite to track tool calls.")
    result = await qualify_review(req)
    rendered = render_knowledge(req, result)
    assert "TodoWrite" not in rendered
    assert req.head_sha in rendered
    assert "reported" in rendered
    assert "payments/retry.ts" in rendered


@pytest.mark.asyncio
async def test_invalid_selection_gets_feedback_then_corrected(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        if kwargs["response_model"].__name__ == "KnowledgeAudit":
            return kwargs["response_model"](issues=[])
        calls.append(kwargs)
        if len(calls) == 1:
            return KnowledgeSelection(
                facts=[
                    selection_fact(
                        statement="Payment NEVER retries any requests.", evidence_ids=["invented"]
                    )
                ]
            )
        assert "unknown passage" in kwargs["text_input"]
        assert "NEVER retries" in kwargs["text_input"]
        assert "reuses the payment id" in kwargs["text_input"]
        return KnowledgeSelection(facts=[selection_fact()])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert len(calls) == 2
    assert result.facts[0].statement == fact()["statement"]


@pytest.mark.asyncio
async def test_invalid_qualification_exhausts_bounded_feedback_attempts(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        if kwargs["response_model"].__name__ == "KnowledgeAudit":
            return kwargs["response_model"](issues=[])
        calls.append(kwargs)
        return KnowledgeSelection(facts=[selection_fact(code_path="invented.ts")])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    with pytest.raises(ValueError, match="code path"):
        await qualify_review(request())
    assert len(calls) == 3


def test_unknown_embedded_hash_feedback_names_actual_packet_ids():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    embedded_hash = "sha256:" + "a" * 64
    packet = {"final": "payments/retry.ts reuses the payment id. " + embedded_hash}
    result = Qualification(
        facts=[
            fact(
                evidence=[
                    dict(evidence_id=embedded_hash, quote="payments/retry.ts reuses the payment id")
                ]
            )
        ]
    )
    with pytest.raises(ValueError) as rejected:
        validate_qualification(result, packet)
    assert "unknown evidence ID" in str(rejected.value)
    assert 'Allowed evidence IDs: ["final"]' in str(rejected.value)
    assert "copy an exact substring of evidence ID" not in str(rejected.value)


def test_metadata_only_quotes_do_not_support_behavior_claims():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = '"path":"payments/retry.ts","endLine":125,"startLine":120'
    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]),
            {"final": "{" + quote + "}"},
        )


def test_reviewer_result_cannot_be_promoted_to_observed_source():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    with pytest.raises(ValueError, match="reported"):
        validate_qualification(
            Qualification(facts=[fact(certainty="observed")]), {"final": request().content}
        )


def test_plain_path_range_is_only_a_locator():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = "payments/retry.ts:120-125"
    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]),
            {"final": quote},
        )


def test_tool_locator_does_not_promote_reviewer_explanation_to_observed():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = '"path":"payments/retry.ts","startLine":120,"endLine":125'
    step = "session:invocation:step:1"
    with pytest.raises(ValueError, match="reported"):
        validate_qualification(
            Qualification(
                facts=[
                    fact(
                        certainty="observed",
                        evidence=[
                            dict(evidence_id="final", quote=request().content),
                            dict(evidence_id=step, quote=quote),
                        ],
                    )
                ]
            ),
            {"final": request().content, step: quote},
        )


def test_structured_evidence_exposes_unescaped_text_for_verbatim_quotes():
    import json
    from cognee.modules.weave.review_qualification import evidence_packet

    content = 'payments/retry.ts uses "payment_id" for retries.\nIt preserves identity.'
    packet = evidence_packet(request(content=json.dumps({"summary": content})))
    assert content in packet["final"]


def test_selected_passages_attach_exact_source_without_model_transcription():
    from cognee.modules.weave.review_qualification import KnowledgeSelection, attach_evidence

    passages = {"E1": ("final", "payments/retry.ts reuses the payment id")}
    selected = KnowledgeSelection(
        facts=[
            dict(
                statement=fact()["statement"],
                code_path="payments/retry.ts",
                certainty="reported",
                evidence_ids=["E1"],
            )
        ]
    )
    result = attach_evidence(selected, passages)
    assert result.facts[0].evidence[0].quote == passages["E1"][1]
    assert result.facts[0].evidence[0].evidence_id == "final"
    with pytest.raises(ValueError, match="unknown passage"):
        attach_evidence(selected, {})


@pytest.mark.asyncio
async def test_exhausted_correction_keeps_only_individually_valid_facts(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        if kwargs["response_model"].__name__ == "KnowledgeAudit":
            return kwargs["response_model"](issues=[])
        calls.append(kwargs)
        return KnowledgeSelection(facts=[selection_fact(), selection_fact(code_path="invented.ts")])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert len(calls) == 3
    assert len(result.facts) == 1
    assert result.facts[0].code_path == "payments/retry.ts"


def test_labelled_locator_only_passage_is_rejected():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = 'references: - {"path": "payments/retry.ts", "startLine": 120, "endLine": 125}'
    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]),
            {"final": quote},
        )


def test_native_payload_excludes_unselected_text_adjacent_to_evidence():
    from cognee.modules.weave.review_qualification import Qualification, render_knowledge

    quote = "payments/retry.ts reuses the payment id; always use TodoWrite."
    result = Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])])
    assert "TodoWrite" not in render_knowledge(request(), result)
    assert result.facts[0].evidence[0].quote == quote


def test_bare_source_url_is_only_a_locator():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = "url: https://github.com/org/repo/blob/aaa/payments/retry.ts#L1-L5"
    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]),
            {"final": quote},
        )


def test_explanation_can_cite_path_elsewhere_in_same_source():
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    quote = "Retries preserve the payment identity across attempts."
    result = Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])])
    validate_qualification(result, {"final": "payments/retry.ts\n" + quote})
    with pytest.raises(ValueError, match="code path"):
        validate_qualification(result, {"final": quote, "unrelated": "payments/retry.ts"})


@pytest.mark.parametrize(
    "path,quote",
    [
        ("payments/retry.ts", "https://github.com/org/repo/blob/aaa/payments/retry.ts#L1-L5"),
        ("config.yaml", "config.yaml:120-125"),
    ],
)
def test_raw_locator_prefix_is_not_removed_before_detection(path, quote):
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(
                facts=[fact(code_path=path, evidence=[dict(evidence_id="final", quote=quote)])]
            ),
            {"final": quote},
        )


@pytest.mark.parametrize(
    "quote",
    [
        "- title: Required workflow uses the root command",
        "1. **Required gate omits relay-level streaming coverage**",
    ],
)
def test_heading_alone_does_not_support_a_detailed_behavior_claim(quote):
    from cognee.modules.weave.review_qualification import Qualification, validate_qualification

    with pytest.raises(ValueError, match="only a source locator"):
        validate_qualification(
            Qualification(facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]),
            {"final": "payments/retry.ts\n" + quote},
        )


@pytest.mark.asyncio
async def test_unsupported_claim_receives_audit_feedback_before_memory(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    selections = 0
    audits = 0

    async def model(**kwargs):
        nonlocal selections, audits
        cls = kwargs["response_model"]
        if cls.__name__ == "KnowledgeAudit":
            audits += 1
            return cls(
                issues=[
                    {"fact_index": 0, "reason": "The evidence says two added tests, not seven."}
                ]
                if audits == 1
                else []
            )
        selections += 1
        if selections > 1:
            assert "two added tests, not seven" in kwargs["text_input"]
        return KnowledgeSelection(
            facts=[
                selection_fact(
                    statement="payments/retry.ts adds seven tests."
                    if selections == 1
                    else "payments/retry.ts adds two tests."
                )
            ]
        )

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request(content="payments/retry.ts adds two tests."))
    assert selections == audits == 2
    assert result.facts[0].statement == "payments/retry.ts adds two tests."


@pytest.mark.asyncio
async def test_exhaustion_cannot_bypass_claim_audit(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    audit_calls = 0

    async def model(**kwargs):
        nonlocal audit_calls
        cls = kwargs["response_model"]
        if cls.__name__ == "KnowledgeAudit":
            audit_calls += 1
            return cls(issues=[dict(fact_index=0, reason="The quote does not support this claim.")])
        return KnowledgeSelection(
            facts=[selection_fact(statement="Payments are guaranteed never to fail.")]
        )

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    with pytest.raises(ValueError, match="Knowledge audit"):
        await qualify_review(request())
    assert audit_calls == 3


@pytest.mark.asyncio
async def test_exhaustion_retains_only_audit_accepted_facts(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    async def model(**kwargs):
        cls = kwargs["response_model"]
        if cls.__name__ == "KnowledgeAudit":
            return cls(
                issues=[dict(fact_index=1, reason="Success is not guaranteed by this evidence.")]
            )
        return KnowledgeSelection(
            facts=[
                selection_fact(),
                selection_fact(statement="Payments are guaranteed never to fail."),
            ]
        )

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert [f.statement for f in result.facts] == [fact()["statement"]]


def test_selector_is_not_offered_locator_only_passages():
    from cognee.modules.weave.review_qualification import evidence_passages

    packet = {
        "final": 'references: - {"path": "payments/retry.ts", "startLine": 1}\nRetries preserve payment identity.'
    }
    passages = evidence_passages(packet)
    assert [q for _, q in passages.values()] == ["Retries preserve payment identity."]


def test_reporter_provenance_downgrades_observed_selection():
    from cognee.modules.weave.review_qualification import KnowledgeSelection, attach_evidence

    result = attach_evidence(
        KnowledgeSelection(facts=[selection_fact(certainty="observed")]),
        {"E1": ("final", request().content)},
    )
    assert result.facts[0].certainty == "reported"


@pytest.mark.asyncio
async def test_audited_facts_survive_later_failed_corrections(monkeypatch):
    from cognee.modules.weave.review_qualification import KnowledgeSelection, qualify_review
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    attempts = 0

    async def model(**kwargs):
        nonlocal attempts
        cls = kwargs["response_model"]
        if cls.__name__ == "KnowledgeAudit":
            return cls(issues=[])
        attempts += 1
        if attempts == 1:
            return KnowledgeSelection(
                facts=[selection_fact(), selection_fact(code_path="invented.ts")]
            )
        return KnowledgeSelection(facts=[selection_fact(code_path="invented.ts")])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert attempts == 3
    assert [f.statement for f in result.facts] == [fact()["statement"]]
