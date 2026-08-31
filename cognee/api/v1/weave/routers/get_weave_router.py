from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from cognee.modules.weave.auth import require_internal_bearer
from cognee.modules.weave.organizations import provision_organization


class ProvisionOrganizationResponse(BaseModel):
    organization_id: UUID
    status: str


class IndexRepositoryResponse(BaseModel):
    job_id: UUID
    status: str


def get_weave_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_internal_bearer)])

    @router.post(
        "/organizations/{organization_id}/provision",
        response_model=ProvisionOrganizationResponse,
    )
    async def provision(
        organization_id: UUID,
    ) -> ProvisionOrganizationResponse:
        binding = await provision_organization(organization_id)
        return ProvisionOrganizationResponse(
            organization_id=binding.organization_id,
            status="ready",
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
    ) -> IndexRepositoryResponse:
        from cognee.modules.weave.archive import persisted_upload
        from cognee.modules.weave.indexing import IndexRequest, index_repository_archive

        try:
            request = IndexRequest(
                organization_id=organization_id,
                github_repository_id=github_repository_id,
                repository_owner=repository_owner,
                repository_name=repository_name,
                default_branch=default_branch,
                requested_sha=requested_sha,
                pipeline_version=pipeline_version,
                extraction_version=extraction_version,
            )
            async with persisted_upload(archive) as archive_path:
                job = await index_repository_archive(request, archive_path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail="Organization not found") from error
        return IndexRepositoryResponse(job_id=job.id, status="accepted")

    return router
