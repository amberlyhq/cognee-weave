"""Apply host-selected notes through native Cognee. No agent or semantic policy."""

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.data.models import Data
from cognee.modules.weave.agent_memory_contracts import (
    MemoryApplyResponse,
    MemoryNoteResponse,
    MemoryNotesResponse,
    MemoryOperationReceipt,
)
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.memory_sources import source_data_id, sync_source, forget_source
from cognee.modules.weave.memory_retrieval import native_source_tag
from cognee.modules.weave.models import (
    WeaveMemoryJob,
    WeaveMemoryNote,
    WeaveMemoryOperation,
    WeaveMemorySource,
    WeaveRepositoryLifecycle,
    WeaveRepositorySnapshot,
)
from cognee.modules.weave.native_memory import customer_dataset
from cognee.modules.weave.organizations import (
    get_organization_binding,
    set_weave_organization_scope,
)
from cognee.modules.weave.scope import native_organization
from cognee.tasks.ingestion.data_item import DataItem


class MemoryConflict(LookupError):
    def __init__(self, message, code="version_conflict"):
        super().__init__(message)
        self.code = code


async def repository_scope(organization_id, repository_id, generation=None, source_sha=None):
    binding = await get_organization_binding(organization_id)
    if binding is None or await customer_dataset(binding) is None:
        raise LookupError("Customer dataset is not ready")
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        lifecycle = await session.get(WeaveRepositoryLifecycle, (organization_id, repository_id))
        if not lifecycle or not lifecycle.active or lifecycle.deletion_pending:
            raise LookupError("Repository is removed or unavailable")
        if generation is not None and lifecycle.lifecycle_generation != generation:
            raise LookupError("Repository lifecycle changed")
        snapshot = await session.scalar(
            select(WeaveRepositorySnapshot).where(
                WeaveRepositorySnapshot.organization_id == organization_id,
                WeaveRepositorySnapshot.github_repository_id == repository_id,
                WeaveRepositorySnapshot.deleted_at.is_(None),
            )
        )
        if snapshot is None or snapshot.status != "indexed":
            raise LookupError("Repository source indexing is not complete")
        if source_sha is not None and (
            snapshot.indexed_sha != source_sha or snapshot.requested_sha != source_sha
        ):
            raise MemoryConflict("Default branch memory source is stale", "stale_source")
        return binding, lifecycle.lifecycle_generation


async def list_memory_notes(organization_id, repository_id, **kwargs):
    async with weave_operation_lock(organization_id, wait=False) as acquired:
        if not acquired:
            raise LookupError("Customer source indexing is busy")
        binding, _ = await repository_scope(organization_id, repository_id)
        from cognee.modules.weave.legacy_memory_notes import import_legacy_notes

        await import_legacy_notes(binding, repository_id)
        return await _list_memory_notes(organization_id, repository_id, **kwargs)


async def _list_memory_notes(
    organization_id, repository_id, *, include_retired=True, offset=0, limit=100
):
    _, generation = await repository_scope(organization_id, repository_id)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, organization_id)
        query = select(WeaveMemoryNote).where(
            WeaveMemoryNote.organization_id == organization_id,
            WeaveMemoryNote.github_repository_id == repository_id,
        )
        if not include_retired:
            query = query.where(WeaveMemoryNote.status == "active")
        records = list(
            await session.scalars(
                query.order_by(WeaveMemoryNote.note_id).offset(offset).limit(limit + 1)
            )
        )
        notes = [
            MemoryNoteResponse(
                **{key: getattr(row, key) for key in MemoryNoteResponse.model_fields}
            )
            for row in records[:limit]
        ]
    return MemoryNotesResponse(
        organization_id=organization_id,
        github_repository_id=repository_id,
        lifecycle_generation=generation,
        notes=notes,
        next_offset=offset + limit if len(records) > limit else None,
    )


def operation_version(note, operation):
    if (note.version if note else 0) != operation.expected_version:
        raise MemoryConflict("Memory note version changed; reload notes")
    if operation.action == "add":
        if note is not None:
            raise MemoryConflict("Memory note already exists")
    elif note is None or note.status != "active":
        raise MemoryConflict("Memory note is missing or retired")
    return operation.expected_version + (operation.action != "unchanged")


async def apply_memory_notes(organization_id, repository_id, request):
    async with weave_operation_lock(organization_id, wait=False) as acquired:
        if not acquired:
            raise LookupError("Customer source indexing is busy")
        binding, _ = await repository_scope(
            organization_id,
            repository_id,
            request.lifecycle_generation,
            None,
        )
        from cognee.modules.weave.legacy_memory_notes import import_legacy_notes

        await import_legacy_notes(binding, repository_id)
        payload = request.model_dump(mode="json")
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.get(WeaveMemoryJob, (organization_id, request.job_id))
            if job is not None and job.github_repository_id != repository_id:
                raise LookupError("Memory job payload conflict")
            if job is not None and job.status in {"superseded", "superseding"}:
                raise MemoryConflict("Memory job was superseded", "stale_source")
            if job is not None and job.payload_hash != payload_hash:
                raise LookupError("Memory job payload conflict")
            if job is not None and job.status == "completed":
                records = list(
                    await session.scalars(
                        select(WeaveMemoryOperation).where(
                            WeaveMemoryOperation.organization_id == organization_id,
                            WeaveMemoryOperation.job_id == request.job_id,
                        )
                    )
                )
                by_id = {record.operation_id: record for record in records}
                return MemoryApplyResponse(
                    organization_id=organization_id,
                    github_repository_id=repository_id,
                    job_id=request.job_id,
                    source_sha=request.source_sha,
                    receipts=[
                        MemoryOperationReceipt(**by_id[operation.operation_id].receipt)
                        for operation in sorted(
                            request.operations, key=lambda item: item.action == "retire"
                        )
                    ],
                )
            if request.source_kind == "default_branch":
                await repository_scope(
                    organization_id, repository_id, request.lifecycle_generation, request.source_sha
                )
            if job is None:
                # Validate all versions before the first native write.
                for operation in request.operations:
                    prior = await session.get(
                        WeaveMemoryOperation, (organization_id, operation.operation_id)
                    )
                    if prior is not None:
                        raise LookupError("Memory operation belongs to another job")
                    pending = await session.scalar(
                        select(WeaveMemoryOperation).where(
                            WeaveMemoryOperation.organization_id == organization_id,
                            WeaveMemoryOperation.github_repository_id == repository_id,
                            WeaveMemoryOperation.note_id == operation.note_id,
                            WeaveMemoryOperation.status.in_(["processing", "superseding"]),
                        )
                    )
                    if pending is not None:
                        raise MemoryConflict(
                            "Memory note has an unfinished operation; supersede the obsolete job first"
                        )
                    note = await session.get(
                        WeaveMemoryNote, (organization_id, repository_id, operation.note_id)
                    )
                    operation_version(note, operation)
                job = WeaveMemoryJob(
                    organization_id=organization_id,
                    job_id=request.job_id,
                    github_repository_id=repository_id,
                    payload=payload,
                    payload_hash=payload_hash,
                    status="processing",
                )
                session.add(job)
                await session.commit()
        user = await get_user(binding.service_user_id)
        llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
        token = native_organization.set(organization_id)
        receipts = []
        try:
            async with scoped_database_context_variables(
                binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
            ):
                # Never retire a replaced note before all replacement writes complete.
                for operation in sorted(
                    request.operations, key=lambda item: item.action == "retire"
                ):
                    receipts.append(
                        await apply_operation(
                            binding,
                            user,
                            repository_id,
                            request,
                            operation,
                            llm=llm,
                            embedding=embedding,
                        )
                    )
        finally:
            native_organization.reset(token)
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.get(WeaveMemoryJob, (organization_id, request.job_id))
            job.status = "completed"
            await session.commit()
        return MemoryApplyResponse(
            organization_id=organization_id,
            github_repository_id=repository_id,
            job_id=request.job_id,
            source_sha=request.source_sha,
            receipts=receipts,
        )


async def apply_operation(binding, user, repository_id, request, operation, *, llm, embedding):
    org = binding.organization_id
    identity = (org, repository_id, operation.note_id)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, org)
        progress = await session.get(WeaveMemoryOperation, (org, operation.operation_id))
        if progress is not None:
            if progress.job_id != request.job_id or progress.github_repository_id != repository_id:
                raise LookupError("Memory operation belongs to another job")
            if progress.status == "completed":
                return MemoryOperationReceipt(**progress.receipt)
            version = progress.version
            # Another job must not advance a note during retry of incomplete native work.
            note = await session.get(WeaveMemoryNote, identity)
            operation_version(note, operation)
        else:
            note = await session.get(WeaveMemoryNote, identity)
            version = operation_version(note, operation)
            progress = WeaveMemoryOperation(
                organization_id=org,
                operation_id=operation.operation_id,
                job_id=request.job_id,
                github_repository_id=repository_id,
                note_id=operation.note_id,
                version=version,
                status="processing",
            )
            session.add(progress)
            await session.commit()
    from cognee.modules.weave.legacy_memory_notes import note_source_key

    source_key = await note_source_key(binding, repository_id, operation.note_id)
    source_identity = (org, repository_id, source_key)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, org)
        progress = await session.get(WeaveMemoryOperation, (org, operation.operation_id))
        if progress.receipt and progress.receipt.get("source_key"):
            source_key = progress.receipt["source_key"]
            source_identity = (org, repository_id, source_key)
        else:
            previous_source = await session.get(WeaveMemorySource, source_identity)
            progress.receipt = {
                "source_key": source_key,
                "previous_source_hash": previous_source.content_hash if previous_source else None,
                "previous_source_qualification": previous_source.qualification
                if previous_source
                else None,
            }
            await session.commit()
    if operation.action in {"add", "update"}:
        content_hash = hashlib.sha256(operation.content.encode()).hexdigest()
        tag = native_source_tag(source_data_id(*source_identity), content_hash)
        item = DataItem(
            data=operation.content,
            label=f"memory-note-{operation.note_id}.txt",
            external_metadata={
                "memory_kind": "agent_note",
                "note_id": str(operation.note_id),
                "github_repository_id": repository_id,
                "source_sha": request.source_sha,
                "source_paths": operation.source_paths,
                "provenance": operation.provenance,
            },
        )
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, org)
            current_source = await session.get(WeaveMemorySource, source_identity)
            native_data = await session.get(Data, source_data_id(*source_identity))
            if native_data is not None and (
                native_data.dataset_id,
                native_data.owner_id,
                native_data.tenant_id,
            ) != (binding.dataset_id, user.id, binding.tenant_id):
                raise ValueError("Foreign native note document")
            native_complete = (
                current_source is not None
                and native_data is not None
                and current_source.content_hash == content_hash
                and current_source.status in {"completed", "processing_note"}
            )
            native_revision = (
                max(
                    version,
                    (current_source.artifact_revision or 0)
                    + (current_source.content_hash != content_hash),
                )
                if current_source is not None
                else version
            )
        if not native_complete:
            await sync_source(
                binding,
                user,
                repository_id,
                source_key,
                item,
                content_hash,
                llm=llm,
                embedding=embedding,
                artifact_revision=native_revision,
                self_improvement=False,
                node_set=[tag],
            )
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, org)
            source = await session.get(WeaveMemorySource, source_identity)
            if source.qualification and not source.qualification.get("memory_note"):
                from cognee.modules.weave.review_code_links import forget_review_code_links

                await forget_review_code_links(binding, source)
            source.status = "processing_note"
            source.dataset_id = binding.dataset_id
            source.qualification = {"memory_note": True, "native_source_tag": tag}
            await session.commit()
        from cognee.modules.weave.note_code_links import sync_note_code_links

        await sync_note_code_links(binding, repository_id, source_key, operation.source_paths)
    elif operation.action == "retire":
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, org)
            source = await session.get(WeaveMemorySource, source_identity)
        if source is not None:
            await forget_source(binding, user, source)
    receipt = MemoryOperationReceipt(
        operation_id=operation.operation_id,
        note_id=operation.note_id,
        version=version,
        action=operation.action,
    )
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, org)
        note = await session.get(WeaveMemoryNote, identity)
        if operation.action == "add":
            note = WeaveMemoryNote(
                organization_id=org, github_repository_id=repository_id, note_id=operation.note_id
            )
            session.add(note)
        if operation.action != "unchanged":
            note.version = version
            note.status = "retired" if operation.action == "retire" else "active"
            if operation.action != "retire":
                note.content = operation.content
                note.source_paths = operation.source_paths
                note.provenance = operation.provenance
            note.source_sha = request.source_sha
            note.updated_at = datetime.now(timezone.utc)
        if operation.action in {"add", "update"}:
            source = await session.get(WeaveMemorySource, source_identity)
            source.status = "completed"
            data = await session.get(Data, source.data_id)
            data.external_metadata = item.external_metadata
        progress = await session.get(WeaveMemoryOperation, (org, operation.operation_id))
        progress.status = "completed"
        progress.receipt = receipt.model_dump(mode="json")
        await session.commit()
    return receipt
