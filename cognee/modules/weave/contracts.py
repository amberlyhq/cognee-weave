from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

RecallMode = Literal["repository_context", "symbol_context", "impact_context"]
RecallStatus = Literal["available", "stale", "unavailable", "timed_out"]


class RecallRequest(BaseModel):
    """Bounded selectors accepted from Amberly; storage routing stays internal."""

    model_config = ConfigDict(extra="forbid")

    mode: RecallMode = "repository_context"
    query: str = Field(min_length=1, max_length=2000)
    github_repository_ids: list[int] = Field(default_factory=list, max_length=20)
    primary_github_repository_id: int | None = Field(default=None, ge=1, le=2**63 - 1)
    seeds: list[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=10, ge=1, le=25)
    depth: int = Field(default=1, ge=0, le=4)

    @field_validator("github_repository_ids")
    @classmethod
    def validate_repository_ids(cls, value: list[int]) -> list[int]:
        if any(item <= 0 or item > 2**63 - 1 for item in value):
            raise ValueError("repository IDs must be positive signed 64-bit integers")
        return list(dict.fromkeys(value))

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 255 for item in normalized):
            raise ValueError("seeds must contain 1 to 255 characters")
        return list(dict.fromkeys(normalized))


class RepositoryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    github_repository_id: int
    repository_owner: str
    repository_name: str
    indexed_default_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    requested_default_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_ref: str | None = None
    age_seconds: int | None = Field(default=None, ge=0)


class RecallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    github_repository_id: int
    repository_owner: str
    repository_name: str
    indexed_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_path: str
    fact_identity: str
    kind: str
    name: str
    line: int | None = Field(default=None, ge=0)
    score: float | None = None


class RecallDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "organization_not_provisioned",
        "no_indexed_repository",
        "backend_unavailable",
        "deadline_exceeded",
    ]


class RecallResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RecallStatus
    organization_id: UUID
    mode: RecallMode
    indexed_default_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    age_seconds: int | None = Field(default=None, ge=0)
    repositories: list[RepositoryReference] = Field(default_factory=list, max_length=1000)
    graph_candidates: list[RecallCandidate] = Field(default_factory=list, max_length=25)
    vector_candidates: list[RecallCandidate] = Field(default_factory=list, max_length=25)
    diagnostics: list[RecallDiagnostic] = Field(default_factory=list, max_length=5)
    native_memory: str | None = Field(default=None, max_length=64000)


class SurfaceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_fact_identity: str
    target_fact_identity: str
    relation_type: str


class SurfaceResponse(BaseModel):
    """Bounded export/visualization data from one resolved organization scope."""

    model_config = ConfigDict(extra="forbid")

    organization_id: UUID
    surface: Literal["export", "visualization"]
    repositories: list[RepositoryReference] = Field(default_factory=list, max_length=1000)
    nodes: list[RecallCandidate] = Field(default_factory=list, max_length=500)
    edges: list[SurfaceEdge] = Field(default_factory=list, max_length=1000)
    native_graph: str | None = Field(default=None, max_length=1000000)


class ReviewMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    github_repository_id: int = Field(gt=0, le=2**53 - 1)
    review_id: UUID
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lifecycle_generation: int = Field(gt=0, le=2**63 - 1)
    artifact_revision: int = Field(ge=0, le=2**53 - 1)
    content: str = Field(min_length=1, max_length=500000)


class ReviewMemoryResponse(BaseModel):
    organization_id: UUID
    review_id: UUID
    status: Literal["remember", "update", "unchanged", "stale_ignored", "removed_ignored"]


class DeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: UUID
    github_repository_id: int | None = None
    status: Literal["deleted"] = "deleted"
