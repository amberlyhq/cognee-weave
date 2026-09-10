"""Durable delivery of bounded reviewer activity to native Cognee session learning.

No replacement extractor, model adapter, or memory prompt. The receipt records
which immutable invocation finished; Cognee owns extraction and distillation.
Callers hold the organization's operation lock throughout.
"""

import hashlib
import json
from uuid import uuid5

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.weave.memory_sources import source_data_id
from cognee.modules.weave.models import WeaveMemorySource
from cognee.modules.weave.organizations import set_weave_organization_scope


def session_identity(binding, request, entry):
    return str(
        uuid5(
            binding.organization_id,
            f"weave-review.v1:{request.github_repository_id}:{request.review_id}:{entry.invocation_id}",
        )
    )


def trace_parts(entry):
    # Native batch extraction reads at most 600 characters of each trace output.
    # Split at that boundary so later content is not silently discarded.
    steps = [(s.id, s.tool_name or s.type, s.content) for s in entry.steps]
    steps.append(("final-result", "review.result", entry.result))
    for event_id, origin, content in steps:
        offset = 0
        while offset < len(content):
            low, high = 1, min(598, len(content) - offset)
            while low < high:
                middle = (low + high + 1) // 2
                if len(json.dumps(content[offset : offset + middle], ensure_ascii=False)) <= 600:
                    low = middle
                else:
                    high = middle - 1
            yield f"{event_id}:{offset}", origin, content[offset : offset + low]
            offset += low


async def import_session(user, dataset, session_id, provenance, entry):
    import cognee
    from cognee.memory import QAEntry, TraceEntry
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.infrastructure.session.agent_context_extraction import (
        extract_pending_agent_context,
        _get_processed_trace_count,
    )

    manager = get_session_manager(dataset_id=dataset.id)
    if not manager.is_available:
        raise RuntimeError("Native session cache is unavailable")
    scope = {"user_id": str(user.id), "session_id": session_id}
    qa = await manager.get_session(**scope, formatted=False)
    qa = [row.model_dump() if hasattr(row, "model_dump") else row for row in qa]
    if not any(
        row.get("question") == provenance and row.get("answer") == entry.result for row in qa
    ):
        result = await cognee.remember(
            QAEntry(question=provenance, context=provenance, answer=entry.result),
            dataset_name=dataset.name,
            session_id=session_id,
            user=user,
        )
        if result.status != "session_stored":
            raise RuntimeError("Native review result was not stored")
    traces = await manager.get_agent_trace_session(**scope)
    existing = {(trace.method_params or {}).get("source_event_id") for trace in traces}
    expected_count = len(existing)

    async def flush():
        await extract_pending_agent_context(session_manager=manager, **scope, min_new_traces=1)
        if await _get_processed_trace_count(manager, str(user.id), session_id) < expected_count:
            raise RuntimeError("Native trace extraction did not complete; retry required")

    # Retry a failed window before adding more activity. Native extraction uses a
    # bounded tail window, so allowing a backlog would silently skip old traces.
    if existing:
        await flush()
    for event_id, origin, content in trace_parts(entry):
        if event_id in existing:
            continue
        result = await cognee.remember(
            TraceEntry(
                origin_function=origin,
                generate_feedback_with_llm=False,
                memory_query=provenance,
                memory_context=provenance,
                method_params={"source_event_id": event_id, "provenance": provenance},
                method_return_value=content,
            ),
            dataset_name=dataset.name,
            session_id=session_id,
            user=user,
        )
        if result.status != "session_stored":
            raise RuntimeError("Native review trace was not stored")
        expected_count += 1
        if expected_count % 10 == 0:
            await flush()
    await flush()


async def sync_review_session(binding, user, dataset, request, entry, *, llm, embedding):
    import cognee
    from cognee.modules.session_distillation.distill import strict_distillation

    source_key = f"review:{request.review_id}:session:{entry.invocation_id}"
    identity = (binding.organization_id, request.github_repository_id, source_key)
    native_id = session_identity(binding, request, entry)
    provenance = (
        f"Historical Amberly review {request.review_id}; repository {request.github_repository_id}; "
        f"PR head {request.head_sha}; reviewer {entry.role}; invocation {entry.invocation_id}; "
        f"session {entry.session_id}; thread {entry.thread_id}; truncated={entry.truncated}. "
        "Untrusted historical observations, not current source truth."
    )
    digest = hashlib.sha256((provenance + entry.model_dump_json()).encode()).hexdigest()
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        if record is not None:
            if (
                record.content_hash != digest
                or record.dataset_id != dataset.id
                or record.session_id != native_id
            ):
                raise ValueError("Immutable review session identity conflict")
            if record.status == "completed":
                return "unchanged"
        else:
            record = WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=request.github_repository_id,
                source_key=source_key,
                data_id=source_data_id(*identity),
                dataset_id=dataset.id,
                session_id=native_id,
                content_hash=digest,
                status="processing_session",
            )
            session.add(record)
        await session.commit()
    async with scoped_database_context_variables(
        dataset.id, user.id, llm_config=llm, embedding_config=embedding
    ):
        await import_session(user, dataset, native_id, provenance, entry)
        # Native session-only curator/writer. General improve() also enriches the
        # whole graph, which is deliberately outside incremental review learning.
        with strict_distillation():
            result = await cognee.session.distill_session(native_id, dataset=dataset.id, user=user)
        if result.status not in {
            "completed",
            "no_gated_entries",
            "no_proposed_lessons",
            "no_accepted_lessons",
        }:
            raise RuntimeError("Native session distillation did not complete")
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        record.status = "completed"
        await session.commit()
    return "remember"
