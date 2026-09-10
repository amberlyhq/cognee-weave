"""Stable source identity and retry receipts around unchanged Cognee APIs.

All callers must hold weave_operation_lock for the customer. Never infer source
ownership from graph entities, content deduplication, or user-writable metadata.
"""

from uuid import UUID, uuid5

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.weave.models import WeaveMemorySource
from cognee.modules.weave.organizations import set_weave_organization_scope


def source_data_id(organization_id: UUID, repository_id: int, source_key: str) -> UUID:
    return uuid5(organization_id, f"weave-source.v1:{repository_id}:{source_key}")


def source_action(record, content_hash: str, *, exists: bool) -> str:
    if exists and record and record.status == "completed" and record.content_hash == content_hash:
        return "unchanged"
    return "update" if exists else "remember"


def stale_source_revision(record, revision, content_hash):
    if record is None or revision is None or record.artifact_revision is None:
        return False
    if revision < record.artifact_revision:
        return True
    if revision == record.artifact_revision and content_hash != record.content_hash:
        raise ValueError("Source revision conflict")
    return False


def assert_native_completed(result) -> None:
    if isinstance(result, dict):
        if not result:
            raise RuntimeError("Native pipeline returned no completion receipt")
        for run in result.values():
            assert_native_completed(run)
        return
    if getattr(result, "status", None) not in {
        "completed",
        "PipelineRunCompleted",
        "PipelineRunAlreadyCompleted",
    }:
        raise RuntimeError("Native memory processing did not complete")
    raw = getattr(result, "raw_result", None)
    if raw:
        assert_native_completed(raw)
    for item in getattr(result, "data_ingestion_info", None) or []:
        if item.get("run_info") is not None:
            assert_native_completed(item["run_info"])


async def source_records(binding, repository_id=None):
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        query = select(WeaveMemorySource).where(
            WeaveMemorySource.organization_id == binding.organization_id
        )
        if repository_id is not None:
            query = query.where(WeaveMemorySource.github_repository_id == repository_id)
        return list(await session.scalars(query.order_by(WeaveMemorySource.source_key)))


async def sync_source(
    binding,
    user,
    repository_id,
    source_key,
    item,
    content_hash,
    *,
    llm,
    embedding,
    artifact_revision=None,
    improve_after_update=False,
    self_improvement=True,
):
    import cognee

    identity = (binding.organization_id, repository_id, source_key)
    data_id = source_data_id(*identity)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        if stale_source_revision(record, artifact_revision, content_hash):
            return "stale_ignored"
        data = await session.get(Data, data_id)
        if data is not None and (
            data.dataset_id != binding.dataset_id
            or data.owner_id != user.id
            or data.tenant_id != binding.tenant_id
        ):
            raise ValueError("Native source identity does not belong to the customer dataset")
        action = source_action(record, content_hash, exists=data is not None)
        if action == "unchanged":
            if artifact_revision is not None:
                record.artifact_revision = artifact_revision
                await session.commit()
            return action
        if record is None:
            record = WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=repository_id,
                source_key=source_key,
                data_id=data_id,
            )
            session.add(record)
        retry_improvement = record.status == "processing_remember"
        record.content_hash = content_hash
        record.artifact_revision = artifact_revision
        record.status = (
            "processing_remember" if action == "remember" or retry_improvement else "processing"
        )
        await session.commit()

    item.data_id = data_id
    if action == "update":
        result = await cognee.update(
            data_id=data_id, data=item, dataset_id=binding.dataset_id, user=user
        )
    else:
        result = await cognee.remember(
            item,
            dataset_id=binding.dataset_id,
            user=user,
            llm_config=llm,
            embedding_config=embedding,
            self_improvement=self_improvement,
        )
    assert_native_completed(result)
    if action == "update" and (improve_after_update or (retry_improvement and self_improvement)):
        assert_native_completed(await cognee.improve(dataset=binding.dataset_id, user=user))
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        record = await session.get(WeaveMemorySource, identity)
        record.status = "completed"
        await session.commit()
    return action


async def forget_source(binding, user, record):
    import cognee

    if record.organization_id != binding.organization_id:
        raise ValueError("Foreign memory source")
    identity = (record.organization_id, record.github_repository_id, record.source_key)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        current = await session.get(WeaveMemorySource, identity)
        if current is None:
            return
        current.status = "deleting"
        await session.commit()
    if record.session_id:
        from cognee.infrastructure.session.get_session_manager import get_session_manager
        manager = get_session_manager(dataset_id=record.dataset_id)
        if not manager.is_available:
            raise RuntimeError("Native session cache is unavailable during deletion")
        await manager.delete_session(user_id=str(user.id), session_id=record.session_id)
    else:
        await cognee.forget(data_id=record.data_id, dataset_id=binding.dataset_id, user=user)
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        current = await session.get(WeaveMemorySource, identity)
        if current is not None:
            await session.delete(current)
            await session.commit()
