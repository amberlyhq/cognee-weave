import asyncio
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Protocol
from uuid import UUID, uuid4

from fastapi_users.exceptions import UserAlreadyExists
from sqlalchemy import select, text, update

from cognee.infrastructure.databases.postgres import dataset_schema_name
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.users.methods import create_user
from cognee.modules.users.models import DatasetDatabase, Tenant, User
from cognee.modules.users.tenants.methods import create_tenant

from .models import WeaveOrganizationBinding


@dataclass(frozen=True)
class OrganizationBinding:
    organization_id: UUID
    tenant_id: UUID
    service_user_id: UUID
    dataset_id: UUID
    graph_schema: str
    vector_schema: str


class OrganizationProvisioningBackend(Protocol):
    async def get(self, organization_id: UUID) -> Optional[OrganizationBinding]: ...

    async def provision(self, organization_id: UUID) -> OrganizationBinding: ...


class OrganizationProvisioner:
    """Serializes provisioning locally; the SQL backend adds a cross-process lock."""

    def __init__(self, backend: OrganizationProvisioningBackend):
        self.backend = backend
        self._locks: dict[UUID, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, organization_id: UUID) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(organization_id, asyncio.Lock())

    async def provision(self, organization_id: UUID) -> OrganizationBinding:
        lock = await self._lock_for(organization_id)
        async with lock:
            existing = await self.backend.get(organization_id)
            if existing is not None:
                return existing
            return await self.backend.provision(organization_id)


class InMemoryOrganizationProvisioningBackend:
    """Small deterministic backend used to prove the provisioning contract."""

    def __init__(self):
        self.bindings: dict[UUID, OrganizationBinding] = {}
        self.provision_count = 0

    async def get(self, organization_id: UUID) -> Optional[OrganizationBinding]:
        return self.bindings.get(organization_id)

    async def provision(self, organization_id: UUID) -> OrganizationBinding:
        self.provision_count += 1
        dataset_id = uuid4()
        schema = dataset_schema_name(dataset_id)
        binding = OrganizationBinding(
            organization_id=organization_id,
            tenant_id=uuid4(),
            service_user_id=uuid4(),
            dataset_id=dataset_id,
            graph_schema=schema,
            vector_schema=schema,
        )
        self.bindings[organization_id] = binding
        return binding


def _binding_result(binding: WeaveOrganizationBinding) -> OrganizationBinding:
    schema = dataset_schema_name(binding.primary_dataset_id)
    return OrganizationBinding(
        organization_id=binding.organization_id,
        tenant_id=binding.tenant_id,
        service_user_id=binding.service_user_id,
        dataset_id=binding.primary_dataset_id,
        graph_schema=schema,
        vector_schema=schema,
    )


def _advisory_lock_key(organization_id: UUID) -> int:
    raw = hashlib.sha256(organization_id.bytes).digest()[:8]
    return int.from_bytes(raw, byteorder="big", signed=True)


def _dialect_name(session) -> str:
    bind = session.get_bind() if hasattr(session, "get_bind") else session.bind
    return bind.dialect.name


async def set_weave_organization_scope(session, organization_id: UUID) -> None:
    """Pin RLS to an organization for the current transaction only."""

    if _dialect_name(session) != "postgresql":
        return
    await session.execute(
        text("SELECT set_config('app.weave_organization_id', :organization_id, true)"),
        {"organization_id": str(organization_id)},
    )


class CogneeOrganizationProvisioningBackend:
    """Create and persist Cognee resources for an Amberly organization."""

    async def get(self, organization_id: UUID) -> Optional[OrganizationBinding]:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            record = await session.scalar(
                select(WeaveOrganizationBinding).where(
                    WeaveOrganizationBinding.organization_id == organization_id,
                    WeaveOrganizationBinding.deleted_at.is_(None),
                )
            )
            return _binding_result(record) if record else None

    async def _service_user(self, organization_id: UUID) -> User:
        engine = get_relational_engine()
        email = f"weave+{organization_id.hex}@internal.amberly"
        async with engine.get_async_session() as session:
            user = await session.scalar(select(User).where(User.email == email))
        if user is not None:
            return user

        try:
            return await create_user(
                email=email,
                password=secrets.token_urlsafe(48),
                is_active=True,
                is_verified=True,
            )
        except UserAlreadyExists:
            async with engine.get_async_session() as session:
                user = await session.scalar(select(User).where(User.email == email))
                if user is None:
                    raise
                return user

    async def _tenant(self, organization_id: UUID, user: User) -> UUID:
        engine = get_relational_engine()
        name = f"weave-org-{organization_id}"
        async with engine.get_async_session() as session:
            tenant_id = await session.scalar(
                select(Tenant.id).where(Tenant.name == name, Tenant.owner_id == user.id)
            )
        if tenant_id is None:
            tenant_id = await create_tenant(name, user.id, set_as_active_tenant=True)
        elif user.tenant_id != tenant_id:
            async with engine.get_async_session() as session:
                await session.execute(
                    update(User).where(User.id == user.id).values(tenant_id=tenant_id)
                )
                await session.commit()
        return tenant_id

    async def provision(self, organization_id: UUID) -> OrganizationBinding:
        engine = get_relational_engine()
        async with engine.get_async_session() as lock_session:
            if _dialect_name(lock_session) == "postgresql":
                await lock_session.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": _advisory_lock_key(organization_id)},
                )

            existing = await lock_session.scalar(
                select(WeaveOrganizationBinding).where(
                    WeaveOrganizationBinding.organization_id == organization_id,
                    WeaveOrganizationBinding.deleted_at.is_(None),
                )
            )
            if existing is not None:
                return _binding_result(existing)

            user = await self._service_user(organization_id)
            tenant_id = await self._tenant(organization_id, user)

            async with engine.get_async_session() as session:
                service_user = await session.scalar(select(User).where(User.id == user.id))
            if service_user is None:
                raise RuntimeError("Provisioned Weave service user disappeared")

            dataset = await create_authorized_dataset(
                f"weave-primary-{organization_id.hex}", service_user
            )

            from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
                get_or_create_dataset_database,
            )

            dataset_database = await get_or_create_dataset_database(dataset.id, service_user)
            self._validate_shared_database(dataset_database, dataset.id)

            record = WeaveOrganizationBinding(
                organization_id=organization_id,
                tenant_id=tenant_id,
                service_user_id=service_user.id,
                primary_dataset_id=dataset.id,
            )
            lock_session.add(record)
            await lock_session.commit()
            return _binding_result(record)

    @staticmethod
    def _validate_shared_database(database: DatasetDatabase, dataset_id: UUID) -> None:
        expected = dataset_schema_name(dataset_id)
        graph_schema = database.graph_database_connection_info.get("graph_database_schema")
        vector_schema = database.vector_database_connection_info.get("schema")
        if (
            database.graph_dataset_database_handler != "postgres_graph_shared"
            or database.vector_dataset_database_handler != "pgvector_shared"
            or graph_schema != expected
            or vector_schema != expected
        ):
            raise RuntimeError(
                "Weave requires dataset-derived shared Postgres graph/vector schemas"
            )


_provisioner = OrganizationProvisioner(CogneeOrganizationProvisioningBackend())


async def provision_organization(organization_id: UUID) -> OrganizationBinding:
    return await _provisioner.provision(organization_id)
