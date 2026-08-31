from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cognee.modules.weave.auth import require_internal_bearer
from cognee.modules.weave.organizations import provision_organization


class ProvisionOrganizationResponse(BaseModel):
    organization_id: UUID
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

    return router
