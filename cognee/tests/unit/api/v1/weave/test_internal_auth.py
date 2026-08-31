from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_internal_bearer_auth_accepts_only_the_configured_token(monkeypatch):
    from cognee.modules.weave import auth

    compared = []

    def recording_compare(left, right):
        compared.append((left, right))
        return left == right

    monkeypatch.setattr(auth.secrets, "compare_digest", recording_compare)

    assert auth.validate_internal_bearer("Bearer correct", "correct") is None
    with pytest.raises(auth.HTTPException) as invalid:
        auth.validate_internal_bearer("Bearer wrong", "correct")

    assert invalid.value.status_code == 401
    assert compared == [("correct", "correct"), ("wrong", "correct")]


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "Bearer", "Bearer ", "Bearer one two"],
)
def test_internal_bearer_auth_rejects_missing_or_malformed_headers(authorization):
    from cognee.modules.weave.auth import HTTPException
    from cognee.modules.weave.auth import validate_internal_bearer

    with pytest.raises(HTTPException) as error:
        validate_internal_bearer(authorization, "configured")

    assert error.value.status_code == 401


def test_internal_router_rejects_missing_and_invalid_organization_ids(monkeypatch):
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router

    monkeypatch.setenv("WEAVE_INTERNAL_TOKEN", "secret")
    app = FastAPI()
    app.include_router(get_weave_router(), prefix="/api/v1/weave")
    client = TestClient(app)

    missing = client.post("/api/v1/weave/organizations//provision")
    invalid = client.post(
        "/api/v1/weave/organizations/not-a-uuid/provision",
        headers={"Authorization": "Bearer secret"},
    )

    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_internal_router_requires_internal_bearer_token(monkeypatch):
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router

    monkeypatch.setenv("WEAVE_INTERNAL_TOKEN", "secret")
    app = FastAPI()
    app.include_router(get_weave_router(), prefix="/api/v1/weave")
    client = TestClient(app)
    path = f"/api/v1/weave/organizations/{uuid4()}/provision"

    assert client.post(path).status_code == 401
    assert client.post(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_openapi_request_models_do_not_expose_storage_selectors(monkeypatch):
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router

    monkeypatch.setenv("WEAVE_INTERNAL_TOKEN", "secret")
    app = FastAPI()
    app.include_router(get_weave_router(), prefix="/api/v1/weave")
    schema = app.openapi()
    rendered = str(schema).lower()

    for forbidden in (
        "tenant_id",
        "dataset_id",
        "schema_name",
        "graph_database",
        "vector_database",
        "sql",
        "cypher",
    ):
        assert forbidden not in rendered
