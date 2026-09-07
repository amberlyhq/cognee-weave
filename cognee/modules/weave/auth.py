import secrets
from typing import Annotated, Optional

from fastapi import Header, HTTPException, status

from .config import get_internal_token


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid internal credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def validate_internal_bearer(authorization: Optional[str], expected_token: str) -> None:
    """Validate Amberly's internal bearer token using constant-time comparison."""

    scheme = ""
    candidate = ""
    if authorization:
        parts = authorization.split(" ")
        if len(parts) == 2:
            scheme, candidate = parts

    token_matches = secrets.compare_digest(candidate, expected_token)
    if scheme.lower() != "bearer" or not candidate or not expected_token or not token_matches:
        raise _unauthorized()


async def require_internal_bearer(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> None:
    validate_internal_bearer(authorization, get_internal_token())
