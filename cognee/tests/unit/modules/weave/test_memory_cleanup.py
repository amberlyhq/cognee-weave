from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI


@pytest.mark.asyncio
async def test_cleanup_endpoint_rejects_off_and_requires_auth(monkeypatch):
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router

    monkeypatch.setenv("WEAVE_INTERNAL_TOKEN", "cleanup-unit-token")
    app = FastAPI()
    app.include_router(get_weave_router())
    path = f"/organizations/{uuid4()}/repositories/1/memory-notes/jobs/{uuid4()}/cleanup"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        response = await client.post(
            path,
            headers={"Authorization": "Bearer cleanup-unit-token"},
            json={"lifecycle_generation": 1, "mode": "off"},
        )
        assert response.status_code == 422
        assert (
            await client.post(path, json={"lifecycle_generation": 1, "mode": "preview"})
        ).status_code == 401


@pytest.mark.parametrize(
    ("names", "vectors", "name_match"),
    [
        (["Client", "Client"], [[1.0, 0.0], [0.0, 1.0]], True),
        (["OrderClient", "BillingClient"], [[1.0, 0.0], [0.99, 0.01]], True),
        (["OrderClient", "BillingClient"], [[1.0, 0.0], [0.99, 0.01]], False),
    ],
)
def test_native_matching_does_not_consider_distinct_service_identity(names, vectors, name_match):
    """Characterize native limits; this is not a claim that these merges are correct."""
    from cognee.modules.engine.models.Entity import Entity
    from cognee.tasks.memify.consolidate_entities import _cluster_entities, _resolve_config

    members = []
    for name, service in zip(names, ("orders", "billing")):
        entity = Entity(name=name, description=f"Client for {service}")
        entity.id = uuid4()
        members.append(
            {
                "id": str(entity.id),
                "name": entity.name,
                "type": "client",
                "props": {**entity.model_dump(), "qualified_service_identity": service},
            }
        )
    assert members[0]["id"] != members[1]["id"]
    clusters = _cluster_entities(members, vectors, _resolve_config({"name_match": name_match}))
    assert len(clusters) == 1
    assert {member["id"] for member in clusters[0]} == {member["id"] for member in members}
