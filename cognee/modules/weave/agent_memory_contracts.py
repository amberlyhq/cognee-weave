"""Storage contracts. Governance owns the meaning and selection of every note."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: UUID
    note_id: UUID
    action: Literal["add", "update", "retire", "unchanged"]
    expected_version: int = Field(ge=0)
    content: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    reason: str

    @model_validator(mode="after")
    def validate_operation(self):
        if self.action in {"add", "update"} and not self.content:
            raise ValueError("Add and update require content")
        if self.action == "add" and self.expected_version != 0:
            raise ValueError("Add requires expected_version 0")
        return self


class MemoryApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: UUID
    lifecycle_generation: int = Field(gt=0)
    source_kind: Literal["review", "default_branch"]
    source_id: str = Field(min_length=1)
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    operations: list[MemoryOperation]

    @model_validator(mode="after")
    def unique_operations(self):
        for key in ("operation_id", "note_id"):
            values = [getattr(item, key) for item in self.operations]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate {key} in memory job")
        return self


class MemoryOperationReceipt(BaseModel):
    operation_id: UUID
    note_id: UUID
    version: int
    action: Literal["add", "update", "retire", "unchanged"]
    status: Literal["completed"] = "completed"


class MemoryApplyResponse(BaseModel):
    organization_id: UUID
    github_repository_id: int
    job_id: UUID
    source_sha: str
    status: Literal["completed"] = "completed"
    receipts: list[MemoryOperationReceipt]


class MemoryNoteResponse(BaseModel):
    note_id: UUID
    version: int
    status: Literal["active", "retired"]
    content: str
    source_sha: str
    source_paths: list[str]
    provenance: dict[str, Any]
    updated_at: datetime


class MemoryNotesResponse(BaseModel):
    organization_id: UUID
    github_repository_id: int
    lifecycle_generation: int
    notes: list[MemoryNoteResponse]
    next_offset: int | None


class MemorySupersedeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lifecycle_generation: int = Field(gt=0)


class MemorySupersedeResponse(BaseModel):
    organization_id: UUID
    github_repository_id: int
    job_id: UUID
    status: Literal["superseded", "completed"]


class MemoryCleanupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lifecycle_generation: int = Field(gt=0)
    mode: Literal["preview", "apply"]


class NativeCleanupRun(BaseModel):
    pipeline_run_id: UUID
    dataset_id: UUID
    status: str


class MemoryCleanupResponse(BaseModel):
    organization_id: UUID
    github_repository_id: int
    job_id: UUID
    mode: Literal["preview", "apply"]
    status: Literal["completed"] = "completed"
    native_runs: list[NativeCleanupRun]
