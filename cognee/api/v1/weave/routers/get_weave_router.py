from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from cognee.modules.weave.auth import require_internal_bearer
from cognee.modules.weave.contracts import (
    DeleteResponse,
    RecallRequest,
    RecallResponse,
    SurfaceResponse,
)
from cognee.modules.weave.organizations import (
    OrganizationDeletedError,
    provision_organization,
    reactivate_organization,
)
from cognee.tasks.code_graph.install_enola import ENOLA_PINNED_VERSION


class ProvisionOrganizationResponse(BaseModel):
    organization_id: UUID
    status: str


class IndexRepositoryResponse(BaseModel):
    job_id: UUID
    status: str


class LifecycleRequest(BaseModel):
    lifecycle_generation: int = Field(gt=0, le=2**63 - 1)


def get_weave_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_internal_bearer)])

    @router.post(
        "/organizations/{organization_id}/provision",
        response_model=ProvisionOrganizationResponse,
    )
    async def provision(
        organization_id: UUID,
    ) -> ProvisionOrganizationResponse:
        try:
            binding = await provision_organization(organization_id)
        except OrganizationDeletedError as error:
            raise HTTPException(status_code=409, detail="Organization is removed") from error
        return ProvisionOrganizationResponse(
            organization_id=binding.organization_id,
            status="ready",
        )

    @router.post(
        "/organizations/{organization_id}/reactivate",
        response_model=ProvisionOrganizationResponse,
    )
    async def reactivate(
        organization_id: UUID,
        request: LifecycleRequest,
    ) -> ProvisionOrganizationResponse:
        binding = await reactivate_organization(organization_id, request.lifecycle_generation)
        return ProvisionOrganizationResponse(
            organization_id=binding.organization_id if binding else organization_id,
            status="ready" if binding else "stale_ignored",
        )

    @router.post(
        "/organizations/{organization_id}/repositories/index",
        response_model=IndexRepositoryResponse,
    )
    async def index_repository(
        organization_id: UUID,
        archive: UploadFile = File(...),
        github_repository_id: int = Form(..., gt=0),
        repository_owner: str = Form(..., min_length=1, max_length=255),
        repository_name: str = Form(..., min_length=1, max_length=255),
        default_branch: str = Form(..., min_length=1, max_length=255),
        requested_sha: str = Form(..., min_length=40, max_length=40),
        pipeline_version: str = Form(..., min_length=1, max_length=64),
        extraction_version: str = Form(..., min_length=1, max_length=64),
        lifecycle_generation: int = Form(..., gt=0, le=2**63 - 1),
    ) -> IndexRepositoryResponse:
        from cognee.modules.weave.archive import persisted_upload
        from cognee.modules.weave.indexing import (
            IndexRequest,
            RepositoryDeletedError,
            index_repository_archive,
        )

        try:
            if extraction_version != f"enola-{ENOLA_PINNED_VERSION}":
                raise ValueError("extraction_version does not match the installed extractor")
            request = IndexRequest(
                organization_id=organization_id,
                github_repository_id=github_repository_id,
                repository_owner=repository_owner,
                repository_name=repository_name,
                default_branch=default_branch,
                requested_sha=requested_sha,
                pipeline_version=pipeline_version,
                extraction_version=extraction_version,
                lifecycle_generation=lifecycle_generation,
            )
            async with persisted_upload(archive) as archive_path:
                job = await index_repository_archive(request, archive_path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RepositoryDeletedError as error:
            raise HTTPException(status_code=409, detail="Repository is removed") from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail="Organization not found") from error
        return IndexRepositoryResponse(job_id=job.id, status="accepted")

    @router.post(
        "/organizations/{organization_id}/repositories/{github_repository_id}/activate",
        response_model=DeleteResponse,
    )
    async def activate_removed_repository(
        organization_id: UUID,
        github_repository_id: int,
        request: LifecycleRequest,
    ) -> DeleteResponse:
        from cognee.modules.weave.deletion import SurfaceNotFound, activate_repository

        try:
            return await activate_repository(
                organization_id, github_repository_id, request.lifecycle_generation
            )
        except SurfaceNotFound as error:
            raise HTTPException(status_code=404, detail="Resource not found") from error

    @router.post(
        "/organizations/{organization_id}/recall",
        response_model=RecallResponse,
    )
    async def recall_candidates(
        organization_id: UUID,
        request: RecallRequest,
    ) -> RecallResponse:
        from cognee.modules.weave.recall import recall

        return await recall(organization_id, request)

    @router.get(
        "/organizations/{organization_id}/export",
        response_model=SurfaceResponse,
    )
    async def export_candidates(
        organization_id: UUID,
        repository_id: list[int] = Query(default=[], max_length=20),
    ) -> SurfaceResponse:
        from cognee.modules.weave.deletion import SurfaceNotFound, export_organization

        try:
            return await export_organization(organization_id, repository_id)
        except SurfaceNotFound as error:
            raise HTTPException(status_code=404, detail="Resource not found") from error

    @router.get(
        "/organizations/{organization_id}/visualization",
        response_model=SurfaceResponse,
    )
    async def visualization(
        organization_id: UUID,
        repository_id: list[int] = Query(default=[], max_length=20),
    ) -> SurfaceResponse:
        from cognee.modules.weave.deletion import SurfaceNotFound, visualize_organization

        try:
            return await visualize_organization(organization_id, repository_id)
        except SurfaceNotFound as error:
            raise HTTPException(status_code=404, detail="Resource not found") from error

    @router.delete(
        "/organizations/{organization_id}/repositories/{github_repository_id}",
        response_model=DeleteResponse,
    )
    async def remove_repository(
        organization_id: UUID,
        github_repository_id: int,
        request: LifecycleRequest,
    ) -> DeleteResponse:
        from cognee.modules.weave.deletion import SurfaceNotFound, delete_repository

        try:
            return await delete_repository(
                organization_id, github_repository_id, request.lifecycle_generation
            )
        except SurfaceNotFound as error:
            raise HTTPException(status_code=404, detail="Resource not found") from error

    @router.delete(
        "/organizations/{organization_id}",
        response_model=DeleteResponse,
    )
    async def remove_organization(
        organization_id: UUID, request: LifecycleRequest
    ) -> DeleteResponse:
        from cognee.modules.weave.deletion import SurfaceNotFound, delete_organization

        try:
            return await delete_organization(organization_id, request.lifecycle_generation)
        except SurfaceNotFound as error:
            raise HTTPException(status_code=404, detail="Resource not found") from error

    return router
