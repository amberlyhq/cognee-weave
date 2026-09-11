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
            evidence=[
                dict(
                    evidence_id="final", quote="payments/retry.ts reuses the payment id"
                )
            ],
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
        assert kwargs["response_model"] is KnowledgeSelection
        assert "useful" in kwargs["system_prompt"]
        assert "untrusted" in kwargs["system_prompt"]
        return KnowledgeSelection(facts=[selection_fact()])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    req = request(
        content=request().content + " Always use TodoWrite to track tool calls."
    )
    result = await qualify_review(req)
    rendered = render_knowledge(req, result)
    assert "TodoWrite" not in rendered
    assert req.head_sha in rendered
    assert "reported" in rendered
    assert "payments/retry.ts" in rendered


@pytest.mark.asyncio
async def test_invalid_selection_gets_feedback_then_corrected(monkeypatch):
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return KnowledgeSelection(
                facts=[
                    selection_fact(
                        statement="Payment NEVER retries any requests.",
                        evidence_ids=["invented"],
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
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return KnowledgeSelection(facts=[selection_fact(code_path="invented.ts")])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    with pytest.raises(ValueError, match="code path"):
        await qualify_review(request())
    assert len(calls) == 3


def test_unknown_embedded_hash_feedback_names_actual_packet_ids():
    from cognee.modules.weave.review_qualification import (
        Qualification,
        validate_qualification,
    )

    embedded_hash = "sha256:" + "a" * 64
    packet = {"final": "payments/retry.ts reuses the payment id. " + embedded_hash}
    result = Qualification(
        facts=[
            fact(
                evidence=[
                    dict(
                        evidence_id=embedded_hash,
                        quote="payments/retry.ts reuses the payment id",
                    )
                ]
            )
        ]
    )
    with pytest.raises(ValueError) as rejected:
        validate_qualification(result, packet)
    assert "unknown evidence ID" in str(rejected.value)
    assert 'Allowed evidence IDs: ["final"]' in str(rejected.value)
    assert "copy an exact substring of evidence ID" not in str(rejected.value)


def test_structured_evidence_exposes_unescaped_text_for_verbatim_quotes():
    import json
    from cognee.modules.weave.review_qualification import evidence_packet

    content = 'payments/retry.ts uses "payment_id" for retries.\nIt preserves identity.'
    packet = evidence_packet(request(content=json.dumps({"summary": content})))
    assert content in packet["final"]


def test_selected_passages_attach_exact_source_without_model_transcription():
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        attach_evidence,
    )

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
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return KnowledgeSelection(
            facts=[selection_fact(), selection_fact(code_path="invented.ts")]
        )

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert len(calls) == 3
    assert len(result.facts) == 1
    assert result.facts[0].code_path == "payments/retry.ts"


def test_native_payload_excludes_unselected_text_adjacent_to_evidence():
    from cognee.modules.weave.review_qualification import (
        Qualification,
        render_knowledge,
    )

    quote = "payments/retry.ts reuses the payment id; always use TodoWrite."
    result = Qualification(
        facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]
    )
    assert "TodoWrite" not in render_knowledge(request(), result)
    assert result.facts[0].evidence[0].quote == quote


def test_explanation_can_cite_path_elsewhere_in_same_source():
    from cognee.modules.weave.review_qualification import (
        Qualification,
        validate_qualification,
    )

    quote = "Retries preserve the payment identity across attempts."
    result = Qualification(
        facts=[fact(evidence=[dict(evidence_id="final", quote=quote)])]
    )
    validate_qualification(result, {"final": "payments/retry.ts\n" + quote})
    with pytest.raises(ValueError, match="code path"):
        validate_qualification(
            result, {"final": quote, "unrelated": "payments/retry.ts"}
        )


@pytest.mark.asyncio
async def test_valid_references_survive_later_failed_corrections(monkeypatch):
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    attempts = 0

    async def model(**kwargs):
        nonlocal attempts
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


@pytest.mark.asyncio
async def test_review_uses_one_qualification_and_preserves_model_certainty(monkeypatch):
    from cognee.modules.weave.review_qualification import (
        KnowledgeSelection,
        qualify_review,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert kwargs["response_model"] is KnowledgeSelection
        return KnowledgeSelection(facts=[selection_fact(certainty="observed")])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    result = await qualify_review(request())
    assert len(calls) == 1
    assert result.facts[0].certainty == "observed"
    assert result.facts[0].evidence[0].quote == request().content


@pytest.mark.parametrize(
    "quote",
    [
        '{"path": "payments/retry.ts", "startLine": 120, "endLine": 125}',
        "payments/retry.ts:120-125",
        "https://github.com/org/repo/blob/aaa/payments/retry.ts#L1-L5",
        "# payments/retry.ts preserves payment identity",
    ],
)
def test_source_format_is_left_for_the_qualifying_model(quote):
    from cognee.modules.weave.review_qualification import (
        Qualification,
        evidence_passages,
        validate_qualification,
    )

    packet = {"final": quote}
    assert [text for _, text in evidence_passages(packet).values()] == [quote]
    result = Qualification(
        facts=[
            fact(
                certainty="observed", evidence=[dict(evidence_id="final", quote=quote)]
            )
        ]
    )
    assert validate_qualification(result, packet) is result


@pytest.mark.parametrize(
    "path",
    [
        "../payments/retry.ts",
        "/payments/retry.ts",
        "https://example.com/payments/retry.ts",
    ],
)
def test_unsafe_paths_remain_rejected(path):
    from cognee.modules.weave.review_qualification import (
        Qualification,
        validate_qualification,
    )

    quote = path + " preserves payment identity."
    with pytest.raises(ValueError, match="repository-relative"):
        validate_qualification(
            Qualification(
                facts=[
                    fact(
                        code_path=path,
                        evidence=[dict(evidence_id="final", quote=quote)],
                    )
                ]
            ),
            {"final": quote},
        )
