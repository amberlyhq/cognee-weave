import asyncio
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_provisioning_is_idempotent_and_concurrency_safe():
    from cognee.modules.weave.organizations import InMemoryOrganizationProvisioningBackend
    from cognee.modules.weave.organizations import OrganizationProvisioner

    organization_id = uuid4()
    backend = InMemoryOrganizationProvisioningBackend()
    provisioner = OrganizationProvisioner(backend)

    bindings = await asyncio.gather(*(provisioner.provision(organization_id) for _ in range(12)))

    assert len({binding.tenant_id for binding in bindings}) == 1
    assert len({binding.service_user_id for binding in bindings}) == 1
    assert len({binding.dataset_id for binding in bindings}) == 1
    assert backend.provision_count == 1


@pytest.mark.asyncio
async def test_provisioning_derives_both_database_schemas_from_dataset_id():
    from cognee.infrastructure.databases.postgres import dataset_schema_name
    from cognee.modules.weave.organizations import InMemoryOrganizationProvisioningBackend
    from cognee.modules.weave.organizations import OrganizationProvisioner

    backend = InMemoryOrganizationProvisioningBackend()
    binding = await OrganizationProvisioner(backend).provision(uuid4())
    expected_schema = dataset_schema_name(binding.dataset_id)

    assert binding.graph_schema == expected_schema
    assert binding.vector_schema == expected_schema


def test_binding_identity_cannot_be_reassigned():
    from cognee.modules.weave.models import WeaveOrganizationBinding

    binding = WeaveOrganizationBinding(
        organization_id=uuid4(),
        tenant_id=uuid4(),
        service_user_id=uuid4(),
        primary_dataset_id=uuid4(),
    )

    with pytest.raises(ValueError, match="immutable"):
        binding.primary_dataset_id = uuid4()


def test_repository_and_job_tables_are_organization_scoped():
    from cognee.modules.weave.models import WeaveIndexJob
    from cognee.modules.weave.models import WeaveRepositorySnapshot

    assert WeaveRepositorySnapshot.__table__.c.organization_id.nullable is False
    assert WeaveIndexJob.__table__.c.organization_id.nullable is False
    assert {column.name for column in WeaveRepositorySnapshot.__table__.primary_key} == {"id"}
    assert {column.name for column in WeaveIndexJob.__table__.primary_key} == {"id"}


@pytest.mark.asyncio
async def test_runtime_scope_uses_transaction_local_postgres_setting():
    from cognee.modules.weave.organizations import set_weave_organization_scope

    class Bind:
        dialect = type("Dialect", (), {"name": "postgresql"})()

    class Session:
        bind = Bind()

        def __init__(self):
            self.statements = []

        async def execute(self, statement, values):
            self.statements.append((str(statement), values))

    organization_id = uuid4()
    session = Session()

    await set_weave_organization_scope(session, organization_id)

    assert session.statements == [
        (
            "SELECT set_config('app.weave_organization_id', :organization_id, true)",
            {"organization_id": str(organization_id)},
        )
    ]
