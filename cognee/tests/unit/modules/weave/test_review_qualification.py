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


@pytest.mark.asyncio
async def test_native_gateway_gets_qualification_instructions_and_only_selected_facts_render(
    monkeypatch,
):
    from cognee.modules.weave.review_qualification import (
        Qualification,
        qualify_review,
        render_knowledge,
    )
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    async def model(**kwargs):
        assert kwargs["response_model"] is Qualification
        assert "generic" in kwargs["system_prompt"]
        assert "untrusted" in kwargs["system_prompt"]
        return Qualification(facts=[fact()])

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    req = request(content=request().content + " Always use TodoWrite to track tool calls.")
    result = await qualify_review(req)
    rendered = render_knowledge(req, result)
    assert "TodoWrite" not in rendered
    assert req.head_sha in rendered
    assert "reported" in rendered
    assert "payments/retry.ts" in rendered
