"""Expose existing native documents to the host maintainer without re-qualifying them."""

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.models import Data
from cognee.modules.weave.memory_sources import source_data_id
from cognee.modules.weave.models import WeaveMemoryNote, WeaveMemorySource
from cognee.modules.weave.organizations import set_weave_organization_scope


async def import_legacy_notes(binding, repository_id):
    """Called under the organization lock. Native content and source IDs stay unchanged."""
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        sources = list(
            await session.scalars(
                select(WeaveMemorySource).where(
                    WeaveMemorySource.organization_id == binding.organization_id,
                    WeaveMemorySource.github_repository_id == repository_id,
                    WeaveMemorySource.status == "completed",
                )
            )
        )
        for source in sources:
            if not source.source_key.startswith(("review:qualified:", "review:merge:")):
                continue
            if source.dataset_id != binding.dataset_id or source.data_id != source_data_id(
                binding.organization_id, repository_id, source.source_key
            ):
                raise ValueError("Foreign legacy source receipt")
            identity = (binding.organization_id, repository_id, source.data_id)
            if await session.get(WeaveMemoryNote, identity) is not None:
                continue
            data = await session.get(Data, source.data_id)
            if data is None:
                continue
            if (data.dataset_id, data.owner_id, data.tenant_id) != (
                binding.dataset_id,
                binding.service_user_id,
                binding.tenant_id,
            ):
                raise ValueError("Foreign legacy native document")
            async with open_data_file(data.raw_data_location, mode="r", encoding="utf-8") as file:
                content = file.read()
            qualification = source.qualification or {}
            paths = {
                fact["code_path"]
                for fact in qualification.get("facts", [])
                if fact.get("code_path")
            }
            paths.update(qualification.get("evidence_paths", {}).values())
            metadata = data.external_metadata or {}
            session.add(
                WeaveMemoryNote(
                    organization_id=binding.organization_id,
                    github_repository_id=repository_id,
                    note_id=source.data_id,
                    version=max(source.artifact_revision or 0, 1),
                    status="active",
                    content=content,
                    source_sha=metadata.get("head_sha") or "0" * 40,
                    source_paths=sorted(paths),
                    provenance={
                        "legacy_source_key": source.source_key,
                        "metadata": metadata,
                        "qualification_receipt": qualification,
                    },
                    updated_at=source.updated_at,
                )
            )
        await session.commit()


async def note_source_key(binding, repository_id, note_id):
    async with get_relational_engine().get_async_session() as session:
        await set_weave_organization_scope(session, binding.organization_id)
        source = await session.scalar(
            select(WeaveMemorySource).where(
                WeaveMemorySource.organization_id == binding.organization_id,
                WeaveMemorySource.github_repository_id == repository_id,
                WeaveMemorySource.data_id == note_id,
            )
        )
    return source.source_key if source else f"review:note:{note_id}"
