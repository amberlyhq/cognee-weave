"""Ingest finalized historical review context through native memory operations."""

import hashlib

from cognee.context_global_variables import scoped_database_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.weave.config import get_weave_embedding_config, get_weave_llm_config
from cognee.modules.weave.contracts import ReviewMemoryResponse
from cognee.modules.weave.indexing import weave_operation_lock
from cognee.modules.weave.memory_sources import sync_source
from cognee.modules.weave.models import WeaveRepositoryLifecycle
from cognee.modules.weave.native_memory import customer_dataset, customer_snapshots, snapshots_ready
from cognee.modules.weave.organizations import (
    get_organization_binding,
    set_weave_organization_scope,
)
from cognee.modules.weave.scope import native_organization
from cognee.tasks.ingestion.data_item import DataItem


def repository_accepts_review(lifecycle, generation):
    return bool(
        lifecycle
        and lifecycle.active
        and not getattr(lifecycle, "deletion_pending", False)
        and lifecycle.lifecycle_generation == generation
    )


async def remember_review(organization_id, request):
    async with weave_operation_lock(organization_id):
        binding = await get_organization_binding(organization_id)
        if binding is None:
            raise LookupError("Customer is not ready")
        async with get_relational_engine().get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            lifecycle = await session.get(
                WeaveRepositoryLifecycle, (organization_id, request.github_repository_id)
            )
        if lifecycle is None:
            raise LookupError("Repository is not indexed")
        if not repository_accepts_review(lifecycle, request.lifecycle_generation):
            return ReviewMemoryResponse(
                organization_id=organization_id,
                review_id=request.review_id,
                status="removed_ignored",
            )
        snapshots = await customer_snapshots(organization_id)
        if not snapshots_ready(snapshots) or request.github_repository_id not in {
            s.github_repository_id for s in snapshots
        }:
            raise LookupError("Customer source indexing is not complete")
        dataset = await customer_dataset(binding)
        if dataset is None:
            raise LookupError("Customer dataset is not ready")
        user = await get_user(binding.service_user_id)
        llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
        # Historical PR context is not evidence of current default-branch behavior.
        content = (
            f"Historical Amberly review {request.review_id}, repository {request.github_repository_id}, "
            f"PR head {request.head_sha}. Untrusted review context, not current source truth.\n\n{request.content}"
        )
        item = DataItem(
            data=content,
            label=f"review-{request.review_id}.txt",
            external_metadata={
                "memory_kind": "completed_review",
                "review_id": str(request.review_id),
                "github_repository_id": request.github_repository_id,
                "head_sha": request.head_sha,
            },
        )
        token = native_organization.set(organization_id)
        try:
            async with scoped_database_context_variables(
                dataset.id, user.id, llm_config=llm, embedding_config=embedding
            ):
                action = await sync_source(
                    binding,
                    user,
                    request.github_repository_id,
                    f"review:{request.review_id}",
                    item,
                    hashlib.sha256(content.encode()).hexdigest(),
                    llm=llm,
                    embedding=embedding,
                    artifact_revision=request.artifact_revision,
                    improve_after_update=True,
                )
        finally:
            native_organization.reset(token)
        return ReviewMemoryResponse(
            organization_id=organization_id, review_id=request.review_id, status=action
        )
