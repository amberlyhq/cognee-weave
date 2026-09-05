import asyncio
import hashlib
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select, text

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.weave.models import (
    WeaveIndexJob,
    WeaveOrganizationBinding,
    WeaveRepositoryLifecycle,
    WeaveRepositorySnapshot,
)
from cognee.modules.weave.organizations import (
    get_organization_binding,
    set_weave_organization_scope,
)

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_GITHUB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class RepositoryDeletedError(RuntimeError):
    """Indexing cannot implicitly reverse a verified repository removal."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class IndexRequest:
    organization_id: UUID
    github_repository_id: int
    repository_owner: str
    repository_name: str
    default_branch: str
    requested_sha: str
    pipeline_version: str
    extraction_version: str
    lifecycle_generation: int = 1

    def __post_init__(self) -> None:
        if not _FULL_SHA.fullmatch(self.requested_sha):
            raise ValueError("requested_sha must be a lowercase 40-character Git SHA")
        if self.github_repository_id <= 0 or self.github_repository_id > 2**63 - 1:
            raise ValueError("github_repository_id must be a positive signed 64-bit integer")
        if self.lifecycle_generation <= 0 or self.lifecycle_generation > 2**63 - 1:
            raise ValueError("lifecycle_generation must be a positive signed 64-bit integer")
        if not _GITHUB_NAME.fullmatch(self.repository_owner) or not _GITHUB_NAME.fullmatch(
            self.repository_name
        ):
            raise ValueError("GitHub repository identity is invalid")
        if ".." in self.default_branch or self.default_branch.startswith(("/", "-")):
            raise ValueError("default_branch is invalid")
        for name, value in (
            ("repository_owner", self.repository_owner),
            ("repository_name", self.repository_name),
            ("default_branch", self.default_branch),
            ("pipeline_version", self.pipeline_version),
            ("extraction_version", self.extraction_version),
        ):
            if not value or len(value) > 255:
                raise ValueError(f"{name} is invalid")


@dataclass(frozen=True)
class IndexJobState:
    id: UUID
    request: IndexRequest
    status: str = "queued"
    error_code: Optional[str] = None
    indexed_sha: Optional[str] = None
    attempt_count: int = 1
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            object.__setattr__(self, "created_at", _now())


@dataclass(frozen=True)
class RepositorySnapshotState:
    organization_id: UUID
    github_repository_id: int
    repository_owner: str
    repository_name: str
    default_branch: str
    requested_sha: str
    indexed_sha: Optional[str]
    pipeline_version: str
    extraction_version: str
    status: str
    error_code: Optional[str] = None


class InMemoryIndexStateStore:
    """Reference implementation of the exact-SHA promotion rules."""

    def __init__(self):
        self.jobs: dict[UUID, IndexJobState] = {}
        self.snapshots: dict[tuple[UUID, int], RepositorySnapshotState] = {}
        self._identities: dict[tuple, UUID] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(request: IndexRequest) -> tuple[UUID, int]:
        return request.organization_id, request.github_repository_id

    @staticmethod
    def _identity(request: IndexRequest) -> tuple:
        return (
            request.organization_id,
            request.github_repository_id,
            request.requested_sha,
            request.pipeline_version,
            request.extraction_version,
        )

    async def accept(
        self, request: IndexRequest, *, reclaim_running: bool = False
    ) -> IndexJobState:
        async with self._lock:
            identity = self._identity(request)
            existing_id = self._identities.get(identity)
            if existing_id is not None:
                existing = self.jobs[existing_id]
                if existing.status == "failed" or (
                    reclaim_running and existing.status == "running"
                ):
                    existing = replace(
                        existing,
                        status="queued",
                        error_code=None,
                        indexed_sha=None,
                        attempt_count=existing.attempt_count + 1,
                        completed_at=None,
                    )
                    self.jobs[existing.id] = existing
                return existing

            job = IndexJobState(id=uuid4(), request=request)
            self.jobs[job.id] = job
            self._identities[identity] = job.id
            previous = self.snapshots.get(self._key(request))
            self.snapshots[self._key(request)] = RepositorySnapshotState(
                organization_id=request.organization_id,
                github_repository_id=request.github_repository_id,
                repository_owner=request.repository_owner,
                repository_name=request.repository_name,
                default_branch=request.default_branch,
                requested_sha=request.requested_sha,
                indexed_sha=previous.indexed_sha if previous else None,
                pipeline_version=request.pipeline_version,
                extraction_version=request.extraction_version,
                status="queued",
            )
            return job

    async def fail(self, job_id: UUID, error_code: str) -> IndexJobState:
        async with self._lock:
            job = replace(
                self.jobs[job_id],
                status="failed",
                error_code=error_code,
                completed_at=_now(),
            )
            self.jobs[job_id] = job
            snapshot = self.snapshots[self._key(job.request)]
            if self._is_current(snapshot, job.request):
                self.snapshots[self._key(job.request)] = replace(
                    snapshot, status="failed", error_code=error_code
                )
            return job

    async def claim(self, job_id: UUID) -> bool:
        async with self._lock:
            job = self.jobs[job_id]
            snapshot = self.snapshots[self._key(job.request)]
            if job.status != "queued":
                return False
            if not self._is_current(snapshot, job.request):
                self.jobs[job_id] = replace(job, status="superseded", completed_at=_now())
                return False
            self.jobs[job_id] = replace(job, status="running")
            return True

    async def succeed(self, job_id: UUID, indexed_sha: str) -> IndexJobState:
        if not _FULL_SHA.fullmatch(indexed_sha):
            raise ValueError("indexed_sha must be a lowercase 40-character Git SHA")
        async with self._lock:
            current_job = self.jobs[job_id]
            if indexed_sha != current_job.request.requested_sha:
                raise ValueError("indexed_sha does not match requested_sha")
            job = replace(
                current_job,
                status="succeeded",
                indexed_sha=indexed_sha,
                error_code=None,
                completed_at=_now(),
            )
            self.jobs[job_id] = job
            snapshot = self.snapshots[self._key(job.request)]
            if self._is_current(snapshot, job.request):
                self.snapshots[self._key(job.request)] = replace(
                    snapshot,
                    indexed_sha=indexed_sha,
                    status="indexed",
                    error_code=None,
                )
            return job

    @staticmethod
    def _is_current(snapshot: RepositorySnapshotState, request: IndexRequest) -> bool:
        return (
            snapshot.requested_sha == request.requested_sha
            and snapshot.pipeline_version == request.pipeline_version
            and snapshot.extraction_version == request.extraction_version
        )

    def current(self, request: IndexRequest) -> RepositorySnapshotState:
        return self.snapshots[self._key(request)]


def _store_lock_key(request: IndexRequest) -> int:
    digest = hashlib.sha256(
        request.organization_id.bytes + request.github_repository_id.to_bytes(8, "big")
    ).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


def _operation_lock_key(namespace: str, identity: bytes) -> int:
    digest = hashlib.sha256(namespace.encode() + b":" + identity).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


@asynccontextmanager
async def weave_operation_lock(organization_id: UUID, github_repository_id: int | None = None):
    """Serialize tenant mutation, then repository mutation, across processes."""

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        if session.get_bind().dialect.name != "postgresql":
            yield
            return
        keys = [_operation_lock_key("weave-organization", organization_id.bytes)]
        if github_repository_id is not None:
            keys.append(
                _operation_lock_key(
                    "weave-repository",
                    organization_id.bytes + github_repository_id.to_bytes(8, "big"),
                )
            )
        try:
            for key in keys:
                await session.execute(text("SELECT pg_advisory_lock(:lock_key)"), {"lock_key": key})
            yield
        finally:
            for key in reversed(keys):
                await session.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"), {"lock_key": key}
                )


def _job_state(job: WeaveIndexJob, snapshot: WeaveRepositorySnapshot) -> IndexJobState:
    request = IndexRequest(
        organization_id=job.organization_id,
        github_repository_id=job.github_repository_id,
        repository_owner=snapshot.repository_owner,
        repository_name=snapshot.repository_name,
        default_branch=snapshot.default_branch,
        requested_sha=job.requested_sha,
        pipeline_version=job.pipeline_version,
        extraction_version=job.extraction_version,
    )
    return IndexJobState(
        id=job.id,
        request=request,
        status=job.status,
        error_code=job.error_code,
        indexed_sha=job.indexed_sha,
        attempt_count=job.attempt_count,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


def _orm_job_is_current(job: WeaveIndexJob, snapshot: WeaveRepositorySnapshot) -> bool:
    return (
        snapshot.deleted_at is None
        and snapshot.requested_sha == job.requested_sha
        and snapshot.pipeline_version == job.pipeline_version
        and snapshot.extraction_version == job.extraction_version
    )


class PostgresIndexStateStore:
    """Transactional exact-SHA ledger protected by organization RLS."""

    async def accept(
        self, request: IndexRequest, *, reclaim_running: bool = False
    ) -> IndexJobState:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, request.organization_id)
            if session.get_bind().dialect.name == "postgresql":
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": _store_lock_key(request)},
                )
            binding = await session.scalar(
                select(WeaveOrganizationBinding)
                .where(WeaveOrganizationBinding.organization_id == request.organization_id)
                .with_for_update()
            )
            if binding is None or binding.deleted_at is not None or binding.deletion_pending:
                raise LookupError("Organization not found")
            if request.lifecycle_generation < binding.lifecycle_generation:
                raise RepositoryDeletedError(
                    "Repository lifecycle predates the current organization lifecycle"
                )

            snapshot = await session.scalar(
                select(WeaveRepositorySnapshot).where(
                    WeaveRepositorySnapshot.organization_id == request.organization_id,
                    WeaveRepositorySnapshot.github_repository_id == request.github_repository_id,
                )
            )
            lifecycle = await session.scalar(
                select(WeaveRepositoryLifecycle)
                .where(
                    WeaveRepositoryLifecycle.organization_id == request.organization_id,
                    WeaveRepositoryLifecycle.github_repository_id == request.github_repository_id,
                )
                .with_for_update()
            )
            if lifecycle is not None and (
                not lifecycle.active
                or lifecycle.deletion_pending
                or request.lifecycle_generation < lifecycle.lifecycle_generation
            ):
                raise RepositoryDeletedError(
                    "Repository lifecycle is removed or newer than this index request"
                )
            if snapshot is not None and snapshot.deleted_at is not None:
                raise RepositoryDeletedError(
                    "Repository is removed; a verified installation event must reactivate it"
                )
            binding.observed_lifecycle_generation = max(
                binding.observed_lifecycle_generation, request.lifecycle_generation
            )
            if lifecycle is None:
                lifecycle = WeaveRepositoryLifecycle(
                    organization_id=request.organization_id,
                    github_repository_id=request.github_repository_id,
                    lifecycle_generation=request.lifecycle_generation,
                    active=True,
                )
                session.add(lifecycle)
            elif request.lifecycle_generation > lifecycle.lifecycle_generation:
                lifecycle.lifecycle_generation = request.lifecycle_generation
            job = await session.scalar(
                select(WeaveIndexJob).where(
                    WeaveIndexJob.organization_id == request.organization_id,
                    WeaveIndexJob.github_repository_id == request.github_repository_id,
                    WeaveIndexJob.requested_sha == request.requested_sha,
                    WeaveIndexJob.pipeline_version == request.pipeline_version,
                    WeaveIndexJob.extraction_version == request.extraction_version,
                )
            )
            if job is not None:
                if snapshot is None:
                    raise RuntimeError("Index job exists without repository snapshot")
                if (
                    job.status == "failed"
                    or (reclaim_running and job.status == "running")
                    or snapshot.status == "not_indexed"
                ):
                    job.status = "queued"
                    job.error_code = None
                    job.indexed_sha = None
                    job.completed_at = None
                    job.attempt_count += 1
                    snapshot.repository_owner = request.repository_owner
                    snapshot.repository_name = request.repository_name
                    snapshot.default_branch = request.default_branch
                    snapshot.requested_sha = request.requested_sha
                    snapshot.pipeline_version = request.pipeline_version
                    snapshot.extraction_version = request.extraction_version
                    snapshot.status = "queued"
                    snapshot.error_code = None
                    await session.commit()
                return _job_state(job, snapshot)

            if snapshot is None:
                snapshot = WeaveRepositorySnapshot(
                    organization_id=request.organization_id,
                    github_repository_id=request.github_repository_id,
                    repository_owner=request.repository_owner,
                    repository_name=request.repository_name,
                    default_branch=request.default_branch,
                )
                session.add(snapshot)
            snapshot.repository_owner = request.repository_owner
            snapshot.repository_name = request.repository_name
            snapshot.default_branch = request.default_branch
            snapshot.requested_sha = request.requested_sha
            snapshot.pipeline_version = request.pipeline_version
            snapshot.extraction_version = request.extraction_version
            snapshot.status = "queued"
            snapshot.error_code = None

            job = WeaveIndexJob(
                organization_id=request.organization_id,
                github_repository_id=request.github_repository_id,
                requested_sha=request.requested_sha,
                pipeline_version=request.pipeline_version,
                extraction_version=request.extraction_version,
                status="queued",
                attempt_count=1,
            )
            session.add(job)
            await session.commit()
            return _job_state(job, snapshot)

    async def is_current(self, organization_id: UUID, job_id: UUID) -> bool:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.scalar(
                select(WeaveIndexJob).where(
                    WeaveIndexJob.id == job_id,
                    WeaveIndexJob.organization_id == organization_id,
                )
            )
            if job is None:
                return False
            snapshot = await session.scalar(
                select(WeaveRepositorySnapshot).where(
                    WeaveRepositorySnapshot.organization_id == organization_id,
                    WeaveRepositorySnapshot.github_repository_id == job.github_repository_id,
                )
            )
            return snapshot is not None and _orm_job_is_current(job, snapshot)

    async def claim(self, organization_id: UUID, job_id: UUID) -> bool:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.scalar(
                select(WeaveIndexJob)
                .where(
                    WeaveIndexJob.id == job_id,
                    WeaveIndexJob.organization_id == organization_id,
                )
                .with_for_update()
            )
            if job is None or job.status != "queued":
                return False
            snapshot = await session.scalar(
                select(WeaveRepositorySnapshot)
                .where(
                    WeaveRepositorySnapshot.organization_id == organization_id,
                    WeaveRepositorySnapshot.github_repository_id == job.github_repository_id,
                )
                .with_for_update()
            )
            if snapshot is None:
                raise RuntimeError("Index job exists without repository snapshot")
            if not _orm_job_is_current(job, snapshot):
                job.status = "superseded"
                job.completed_at = _now()
                await session.commit()
                return False
            job.status = "running"
            snapshot.status = "running"
            await session.commit()
            return True

    async def fail(self, organization_id: UUID, job_id: UUID, error_code: str) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,64}", error_code):
            error_code = "indexing_failed"
        await self._finish(
            organization_id,
            job_id,
            status="failed",
            error_code=error_code,
        )

    async def succeed(self, organization_id: UUID, job_id: UUID, indexed_sha: str) -> None:
        if not _FULL_SHA.fullmatch(indexed_sha):
            raise ValueError("indexed_sha must be a lowercase 40-character Git SHA")
        await self._finish(
            organization_id,
            job_id,
            status="succeeded",
            indexed_sha=indexed_sha,
        )

    async def _finish(
        self,
        organization_id: UUID,
        job_id: UUID,
        *,
        status: str,
        error_code: Optional[str] = None,
        indexed_sha: Optional[str] = None,
    ) -> None:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await set_weave_organization_scope(session, organization_id)
            job = await session.scalar(
                select(WeaveIndexJob)
                .where(
                    WeaveIndexJob.id == job_id,
                    WeaveIndexJob.organization_id == organization_id,
                )
                .with_for_update()
            )
            if job is None:
                raise LookupError("Index job was not found")
            snapshot = await session.scalar(
                select(WeaveRepositorySnapshot)
                .where(
                    WeaveRepositorySnapshot.organization_id == organization_id,
                    WeaveRepositorySnapshot.github_repository_id == job.github_repository_id,
                )
                .with_for_update()
            )
            if snapshot is None:
                raise RuntimeError("Index job exists without repository snapshot")
            if indexed_sha is not None and indexed_sha != job.requested_sha:
                raise ValueError("indexed_sha does not match requested_sha")

            job.status = status
            job.error_code = error_code
            if indexed_sha is not None:
                job.indexed_sha = indexed_sha
            if status in ("failed", "succeeded", "superseded"):
                job.completed_at = _now()

            if _orm_job_is_current(job, snapshot):
                if status == "succeeded":
                    snapshot.indexed_sha = indexed_sha
                    snapshot.status = "indexed"
                    snapshot.error_code = None
                elif status == "failed":
                    snapshot.status = "failed"
                    snapshot.error_code = error_code
                elif status == "running":
                    snapshot.status = "running"
            await session.commit()


async def index_repository_archive(
    request: IndexRequest,
    archive_path,
    store: Optional[PostgresIndexStateStore] = None,
) -> IndexJobState:
    """Validate and index one exact GitHub default-branch archive."""

    from cognee.modules.weave.archive import UnsafeArchiveError, validated_archive
    from cognee.modules.weave.native_memory import NATIVE_PIPELINE_VERSION, remember_repository

    # Old parser-only receipts cannot satisfy native-memory indexing.
    request = replace(request, pipeline_version=NATIVE_PIPELINE_VERSION)
    store = store or PostgresIndexStateStore()
    async with weave_operation_lock(request.organization_id, request.github_repository_id):
        binding = await get_organization_binding(request.organization_id)
        if binding is None:
            raise LookupError("Organization is not provisioned")
        job = await store.accept(request, reclaim_running=True)
        if not await store.claim(request.organization_id, job.id):
            return job

        try:
            with validated_archive(archive_path) as repository:
                await remember_repository(binding, request, repository)
            await store.succeed(request.organization_id, job.id, request.requested_sha)
        except UnsafeArchiveError:
            await store.fail(request.organization_id, job.id, "archive_invalid")
            raise
        except Exception:
            await store.fail(request.organization_id, job.id, "indexing_failed")
            raise
        return job
