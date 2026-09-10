from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    UUID,
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import validates

from cognee.infrastructure.databases.relational import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WeaveOrganizationBinding(Base):
    """Immutable mapping from an Amberly organization to Cognee resources."""

    __tablename__ = "weave_organization_bindings"

    organization_id = Column(UUID, primary_key=True)
    tenant_id = Column(
        UUID, ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    service_user_id = Column(
        UUID, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    primary_dataset_id = Column(
        UUID, ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    lifecycle_generation = Column(BigInteger, nullable=False, default=0)
    observed_lifecycle_generation = Column(BigInteger, nullable=False, default=0)
    deletion_pending = Column(Boolean, nullable=False, default=False, server_default="false")

    @validates("organization_id", "tenant_id", "service_user_id", "primary_dataset_id")
    def _keep_identity_immutable(self, key, value):
        current = getattr(self, key, None)
        if current is not None and current != value:
            raise ValueError(f"Weave organization binding {key} is immutable")
        return value


class WeaveRepositoryLifecycle(Base):
    """Ordered GitHub lifecycle state independent of repository index state."""

    __tablename__ = "weave_repository_lifecycles"

    organization_id = Column(
        UUID,
        ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
        primary_key=True,
    )
    github_repository_id = Column(BigInteger, primary_key=True)
    lifecycle_generation = Column(BigInteger, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)
    deletion_pending = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class WeaveMemorySource(Base):
    """Integration receipts, not graph ownership or custom memory semantics.

    No FK to Data: native update deletes and recreates that row under the same ID.
    A processing receipt survives failures so recall cannot serve partial memory.
    """

    __tablename__ = "weave_memory_sources"
    organization_id = Column(
        UUID,
        ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
        primary_key=True,
    )
    github_repository_id = Column(BigInteger, primary_key=True)
    source_key = Column(String(1024), primary_key=True)
    data_id = Column(UUID, nullable=False, unique=True)
    dataset_id = Column(UUID, nullable=True)
    session_id = Column(String(255), nullable=True)
    qualification = Column(JSON, nullable=True)
    content_hash = Column(String(64), nullable=False)
    artifact_revision = Column(BigInteger, nullable=True)
    status = Column(String(32), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class WeaveRepositorySnapshot(Base):
    """Latest exact-SHA indexing state for one GitHub repository."""

    __tablename__ = "weave_repository_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "github_repository_id",
            name="uq_weave_repository_org_github_id",
        ),
        Index("ix_weave_repository_org_status", "organization_id", "status"),
    )

    id = Column(UUID, primary_key=True, default=uuid4)
    organization_id = Column(
        UUID,
        ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
        nullable=False,
    )
    github_repository_id = Column(BigInteger, nullable=False)
    repository_owner = Column(String(255), nullable=False)
    repository_name = Column(String(255), nullable=False)
    default_branch = Column(String(255), nullable=False)
    requested_sha = Column(String(40), nullable=True)
    indexed_sha = Column(String(40), nullable=True)
    pipeline_version = Column(String(64), nullable=True)
    extraction_version = Column(String(64), nullable=True)
    status = Column(String(32), nullable=False, default="not_indexed")
    error_code = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)
    deleted_at = Column(DateTime(timezone=True), nullable=True)


class WeaveIndexJob(Base):
    """Auditable exact-SHA indexing attempt, scoped by organization."""

    __tablename__ = "weave_index_jobs"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "github_repository_id",
            "requested_sha",
            "pipeline_version",
            "extraction_version",
            name="uq_weave_index_job_identity",
        ),
        Index("ix_weave_index_job_org_status", "organization_id", "status"),
    )

    id = Column(UUID, primary_key=True, default=uuid4)
    organization_id = Column(
        UUID,
        ForeignKey("weave_organization_bindings.organization_id", ondelete="CASCADE"),
        nullable=False,
    )
    github_repository_id = Column(BigInteger, nullable=False)
    requested_sha = Column(String(40), nullable=False)
    indexed_sha = Column(String(40), nullable=True)
    pipeline_version = Column(String(64), nullable=False)
    extraction_version = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="queued")
    error_code = Column(String(128), nullable=True)
    attempt_count = Column(BigInteger, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)
