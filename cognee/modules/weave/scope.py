from contextvars import ContextVar
from uuid import UUID

# Only the internal tenant boundary sets this. It is not a request parameter.
native_organization: ContextVar[UUID | None] = ContextVar("weave_native_organization", default=None)
