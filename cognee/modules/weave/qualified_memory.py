"""Durable qualification receipts around native remember/update.

Caller holds the organization lock and sets native database/model context.
"""

import hashlib

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.weave.memory_sources import source_data_id, sync_source
from cognee.modules.weave.models import WeaveMemorySource
from cognee.modules.weave.organizations import set_weave_organization_scope
from cognee.modules.weave.review_qualification import (
    QUALIFICATION_VERSION,
    Qualification,
    qualify_review,
    render_knowledge,
    request_hash,
)
from cognee.tasks.ingestion.data_item import DataItem


async def sync_qualified_review(binding, user, request, *, llm, embedding):
    # Separate identity keeps existing historical memories intact during rollout.
    source_key = f"review:qualified:{request.review_id}"
    identity = (binding.organization_id, request.github_repository_id, source_key)
    input_hash = request_hash(request)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        receipt = record.qualification if record is not None else None
    if receipt:
        if request.artifact_revision < receipt["artifact_revision"]:
            return "stale_ignored"
        if request.artifact_revision == receipt["artifact_revision"]:
            if receipt["input_hash"] != input_hash:
                raise ValueError("Qualified review revision conflict")
            result = Qualification(facts=receipt["facts"])
        else:
            result = await qualify_review(request)
    else:
        result = await qualify_review(request)
    content = render_knowledge(request, result)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    # Save the accepted selection before any native LLM/embedding work. Failures
    # resume this exact payload; they never fall back to raw sessions.
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        if record is None:
            record = WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=request.github_repository_id,
                source_key=source_key,
                data_id=source_data_id(*identity),
                dataset_id=binding.dataset_id,
                content_hash=content_hash,
                status="qualified",
            )
            session.add(record)
        if record.qualification is None or record.qualification["input_hash"] != input_hash:
            record.status = "qualified"
        prior_states = (record.qualification or {}).get("fact_states", {})
        record.qualification = dict(
            version=QUALIFICATION_VERSION,
            input_hash=input_hash,
            artifact_revision=request.artifact_revision,
            facts=result.model_dump()["facts"],
            fact_states=prior_states,
        )
        await session.commit()
    if not result.facts:
        # A replacement review may no longer justify previously qualified facts.
        # Remove only this source's native data; keep the empty receipt for replay.
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, binding.organization_id)
            data = await session.get(Data, source_data_id(*identity))
        if data is not None:
            if (
                data.dataset_id != binding.dataset_id
                or data.owner_id != user.id
                or data.tenant_id != binding.tenant_id
            ):
                raise ValueError("Foreign qualified review source")
            import cognee

            await cognee.forget(data_id=data.id, dataset_id=binding.dataset_id, user=user)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, binding.organization_id)
            record = await session.get(WeaveMemorySource, identity)
            record.status = "completed"
            record.content_hash = content_hash
            record.artifact_revision = request.artifact_revision
            await session.commit()
        return "unchanged"
    item = DataItem(
        data=content,
        label=f"qualified-review-{request.review_id}.txt",
        external_metadata=dict(
            memory_kind="qualified_review",
            review_id=str(request.review_id),
            github_repository_id=request.github_repository_id,
            head_sha=request.head_sha,
            qualification_version=QUALIFICATION_VERSION,
        ),
    )
    return await sync_source(
        binding,
        user,
        request.github_repository_id,
        source_key,
        item,
        content_hash,
        llm=llm,
        embedding=embedding,
        artifact_revision=request.artifact_revision,
        improve_after_update=False,
        self_improvement=False,
    )
