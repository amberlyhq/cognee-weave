"""Atomic consolidation preserves native storage without running models."""

import json
import os
from uuid import uuid4

import pytest
from sqlalchemy import select, text

pytestmark = pytest.mark.skipif(os.getenv("DB_PROVIDER") != "postgres", reason="requires Postgres")


async def _fixture():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name
    from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
        get_or_create_dataset_database,
    )
    from cognee.modules.data.methods import create_authorized_dataset
    from cognee.modules.users.methods import get_user
    from cognee.modules.weave.organizations import provision_organization
    from cognee.modules.weave.models import WeaveRepositorySnapshot, WeaveMemorySource
    from cognee.modules.data.models import Data
    from cognee.modules.session_lifecycle.models import SessionRecord

    binding = await provision_organization(uuid4())
    user = await get_user(binding.service_user_id)
    legacy = await create_authorized_dataset(
        f"weave-memory-{binding.organization_id.hex}-123-code-v1", user
    )
    await get_or_create_dataset_database(legacy.id, user)
    schema = dataset_schema_name(legacy.id)
    data_id, node_id, other_id, run_id = uuid4(), uuid4(), uuid4(), uuid4()
    source = f"source_ref:v1:{legacy.id}:{data_id}"
    source_run = f"source_run_ref:v1:{run_id}:{source}"
    async with get_relational_engine().get_async_session() as session:
        session.add(
            WeaveRepositorySnapshot(
                organization_id=binding.organization_id,
                github_repository_id=123,
                repository_owner="fixture",
                repository_name="fixture",
                default_branch="main",
            )
        )
        session.add(
            Data(
                id=data_id,
                name="Historical knowledge",
                dataset_id=legacy.id,
                owner_id=user.id,
                tenant_id=binding.tenant_id,
                content_hash="a" * 64,
                external_metadata={"preserved": True},
            )
        )
        session.add(
            WeaveMemorySource(
                organization_id=binding.organization_id,
                github_repository_id=123,
                source_key="review:old:session:one",
                data_id=data_id,
                dataset_id=legacy.id,
                session_id="session-one",
                content_hash="a" * 64,
                status="completed",
            )
        )
        session.add(SessionRecord(session_id="session-one", user_id=user.id, dataset_id=legacy.id))
        for identity, kind in [(node_id, "DocumentChunk"), (other_id, "Entity")]:
            await session.execute(
                text(f'''INSERT INTO "{schema}".graph_node
                (id,name,type,properties,source_ref_keys,source_dataset_ids,source_run_ids,source_run_refs)
                VALUES (:id,'knowledge',:kind,CAST(:props AS jsonb),:refs,:datasets,:runs,:runrefs)'''),
                dict(
                    id=str(identity),
                    kind=kind,
                    props=json.dumps({"type": kind, "text": "keep me"}),
                    refs=[source],
                    datasets=[str(legacy.id)],
                    runs=[str(run_id)],
                    runrefs=[source_run],
                ),
            )
        await session.execute(
            text(f'''INSERT INTO "{schema}".graph_edge
            (source_id,target_id,relationship_name,properties,source_ref_keys,source_dataset_ids,source_run_ids,source_run_refs)
            VALUES (:a,:b,'contains','{{"confidence":"reported"}}',:refs,:datasets,:runs,:runrefs)'''),
            dict(
                a=str(node_id),
                b=str(other_id),
                refs=[source],
                datasets=[str(legacy.id)],
                runs=[str(run_id)],
                runrefs=[source_run],
            ),
        )
        await session.execute(
            text(f'''CREATE TABLE "{schema}"."DocumentChunk_text"
            (id uuid PRIMARY KEY, payload json, vector public.vector(3))''')
        )
        await session.execute(
            text(f'''INSERT INTO "{schema}"."DocumentChunk_text"
            VALUES (:id, CAST(:payload AS json), '[0.1,0.2,0.3]')'''),
            dict(id=node_id, payload=json.dumps({"text": "keep me", "belongs_to_set": ["old"]})),
        )
        await session.commit()
    return binding, legacy, data_id, node_id, other_id, run_id


async def _snapshot(binding):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name
    from cognee.modules.data.models import Data, Dataset
    from cognee.modules.weave.models import WeaveMemorySource
    from cognee.modules.session_lifecycle.models import SessionRecord

    async with get_relational_engine().get_async_session() as session:
        datasets = list(
            await session.scalars(
                select(Dataset).where(Dataset.owner_id == binding.service_user_id)
            )
        )
        result = {}
        for dataset in datasets:
            schema = dataset_schema_name(dataset.id)
            tables = list(
                await session.scalars(
                    text("SELECT tablename FROM pg_tables WHERE schemaname=:s ORDER BY tablename"),
                    {"s": schema},
                )
            )
            result[str(dataset.id)] = {}
            for table in tables:
                result[str(dataset.id)][table] = sorted(
                    list(
                        await session.scalars(
                            text(f'SELECT row_to_json(t)::text FROM "{schema}"."{table}" t')
                        )
                    )
                )
        result["data"] = [
            (str(d.id), str(d.dataset_id), d.content_hash)
            for d in await session.scalars(
                select(Data).where(Data.owner_id == binding.service_user_id)
            )
        ]
        result["receipts"] = [
            (r.source_key, str(r.dataset_id), r.status)
            for r in await session.scalars(
                select(WeaveMemorySource).where(
                    WeaveMemorySource.organization_id == binding.organization_id
                )
            )
        ]
        result["sessions"] = [
            (r.session_id, str(r.dataset_id))
            for r in await session.scalars(
                select(SessionRecord).where(SessionRecord.user_id == binding.service_user_id)
            )
        ]
        return result


@pytest.mark.asyncio
async def test_migration_preserves_native_memory_vectors_and_other_customer(monkeypatch):
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    async def forbidden(*args, **kwargs):
        raise AssertionError("Migration must never call a model")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", forbidden)
    binding, legacy, data_id, node_id, other_id, run_id = await _fixture()
    canary = (await _fixture())[0]
    before = await _snapshot(canary)
    receipt = await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert receipt["migrated_datasets"] == [str(legacy.id)]
    after = await _snapshot(binding)
    assert str(legacy.id) not in after
    graph = after[str(binding.dataset_id)]
    node = next(json.loads(s) for s in graph["graph_node"] if json.loads(s)["id"] == str(node_id))
    assert node["properties"] == {"type": "DocumentChunk", "text": "keep me"}
    assert node["source_ref_keys"] == [f"source_ref:v1:{binding.dataset_id}:{data_id}"]
    assert node["source_run_refs"] == [
        f"source_run_ref:v1:{run_id}:source_ref:v1:{binding.dataset_id}:{data_id}"
    ]
    assert len(graph["graph_edge"]) == 1
    point = json.loads(graph["DocumentChunk_text"][0])
    assert point["vector"] == "[0.1,0.2,0.3]"
    assert point["payload"]["text"] == "keep me"
    assert after["data"] == [(str(data_id), str(binding.dataset_id), "a" * 64)]
    assert after["receipts"] == [("review:old:session:one", str(binding.dataset_id), "completed")]
    assert after["sessions"] == [("session-one", str(binding.dataset_id))]
    assert await _snapshot(canary) == before
    assert (await dataset_migration.migrate_customer_datasets(binding.organization_id))[
        "migrated_datasets"
    ] == []
    assert await _snapshot(binding) == after


@pytest.mark.asyncio
async def test_migration_rolls_back_when_retirement_fails(monkeypatch):
    from cognee.modules.weave import dataset_migration

    binding = (await _fixture())[0]
    before = await _snapshot(binding)

    async def fail(*args, **kwargs):
        raise RuntimeError("injected retirement failure")

    monkeypatch.setattr(dataset_migration, "_retire_dataset", fail)
    with pytest.raises(RuntimeError, match="injected retirement"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before


@pytest.mark.asyncio
async def test_migration_rejects_foreign_dataset_owner_without_writes():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from sqlalchemy import update

    binding, legacy, *_ = await _fixture()
    canary = (await _fixture())[0]
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(Dataset).where(Dataset.id == legacy.id).values(owner_id=canary.service_user_id)
        )
        await session.commit()
    before, foreign = await _snapshot(binding), await _snapshot(canary)
    with pytest.raises(ValueError, match="ownership"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before
    assert await _snapshot(canary) == foreign


@pytest.mark.asyncio
async def test_migration_stamps_only_code_with_verified_repository_identity():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name

    binding, legacy, *_ = await _fixture()
    schema = dataset_schema_name(legacy.id)
    identity = str(uuid4())
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            text(f'''INSERT INTO "{schema}".graph_node (id,name,type,properties)
            VALUES (:id,'runRelay','CodeSymbol','{{"type":"CodeSymbol","repo":"old-repo","name":"runRelay"}}')'''),
            {"id": identity},
        )
        await session.commit()
    await dataset_migration.migrate_customer_datasets(binding.organization_id)
    graph = (await _snapshot(binding))[str(binding.dataset_id)]["graph_node"]
    node = next(json.loads(s) for s in graph if json.loads(s)["id"] == identity)
    assert node["properties"]["organization_id"] == str(binding.organization_id)
    assert node["properties"]["github_repository_id"] == 123
    assert node["properties"]["repo"] == "old-repo"
    entity = next(json.loads(s) for s in graph if json.loads(s)["type"] == "Entity")
    assert "organization_id" not in entity["properties"]


@pytest.mark.asyncio
async def test_migration_preserves_shared_entity_descriptions_and_both_source_refs():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name

    binding, legacy, data_id, _, shared_id, _ = await _fixture()
    schema = dataset_schema_name(binding.dataset_id)
    original_ref = f"source_ref:v1:{binding.dataset_id}:{uuid4()}"
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            text(f'''INSERT INTO "{schema}".graph_node
            (id,name,type,properties,source_ref_keys,source_dataset_ids)
            VALUES (:id,'knowledge','Entity','{{"type":"Entity","text":"primary description"}}',:refs,:datasets)'''),
            {"id": str(shared_id), "refs": [original_ref], "datasets": [str(binding.dataset_id)]},
        )
        await session.commit()
    await dataset_migration.migrate_customer_datasets(binding.organization_id)
    graph = (await _snapshot(binding))[str(binding.dataset_id)]["graph_node"]
    node = next(json.loads(s) for s in graph if json.loads(s)["id"] == str(shared_id))
    assert node["properties"]["text"] == "primary description"
    assert node["properties"]["_weave_migration_properties"][str(legacy.id)]["text"] == "keep me"
    assert set(node["source_ref_keys"]) == {
        original_ref,
        f"source_ref:v1:{binding.dataset_id}:{data_id}",
    }


@pytest.mark.asyncio
async def test_migration_rejects_foreign_graph_provenance_and_rolls_back():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name

    binding, legacy, *_ = await _fixture()
    schema = dataset_schema_name(legacy.id)
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            text(f'UPDATE "{schema}".graph_node SET source_dataset_ids=:ids'),
            {"ids": [str(uuid4())]},
        )
        await session.commit()
    before = await _snapshot(binding)
    with pytest.raises(ValueError, match="Foreign dataset"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["owner_id", "tenant_id"])
async def test_migration_rejects_data_with_unverifiable_ownership(field):
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from sqlalchemy import update

    binding, _, data_id, *_ = await _fixture()
    async with get_relational_engine().get_async_session() as session:
        await session.execute(update(Data).where(Data.id == data_id).values(**{field: None}))
        await session.commit()
    before = await _snapshot(binding)
    with pytest.raises(ValueError, match="ownership"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before


@pytest.mark.asyncio
async def test_migration_rejects_unregistered_legacy_storage():
    from cognee.modules.weave import dataset_migration
    from cognee.modules.data.methods import create_authorized_dataset
    from cognee.modules.users.methods import get_user

    binding = (await _fixture())[0]
    await create_authorized_dataset(
        f"weave-memory-{binding.organization_id.hex}-999-code-v1",
        await get_user(binding.service_user_id),
    )
    before = await _snapshot(binding)
    with pytest.raises(ValueError, match="Unregistered legacy"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before


@pytest.mark.asyncio
async def test_migration_rejects_migration_revision_drift():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import DatasetDatabase
    from sqlalchemy import update

    binding, legacy, *_ = await _fixture()
    async with get_relational_engine().get_async_session() as session:
        previous = (await session.get(DatasetDatabase, legacy.id)).migration_revision
        await session.execute(
            update(DatasetDatabase)
            .where(DatasetDatabase.dataset_id == legacy.id)
            .values(migration_revision="unknown-revision")
        )
        await session.commit()
    try:
        before = await _snapshot(binding)
        with pytest.raises(ValueError, match="migration revision"):
            await dataset_migration.migrate_customer_datasets(binding.organization_id)
        assert await _snapshot(binding) == before
    finally:
        async with get_relational_engine().get_async_session() as session:
            await session.execute(
                update(DatasetDatabase)
                .where(DatasetDatabase.dataset_id == legacy.id)
                .values(migration_revision=previous)
            )
            await session.commit()


@pytest.mark.asyncio
async def test_migration_checks_exact_stored_content_before_retirement():
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name

    binding = (await _fixture())[0]
    schema = dataset_schema_name(binding.dataset_id)
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            text(f'''CREATE FUNCTION "{schema}".corrupt_copy() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN NEW.properties = '{{}}'::jsonb; RETURN NEW; END $$''')
        )
        await session.execute(
            text(f'''CREATE TRIGGER corrupt_copy BEFORE INSERT OR UPDATE
            ON "{schema}".graph_node FOR EACH ROW EXECUTE FUNCTION "{schema}".corrupt_copy()''')
        )
        await session.commit()
    before = await _snapshot(binding)
    with pytest.raises(RuntimeError, match="readback"):
        await dataset_migration.migrate_customer_datasets(binding.organization_id)
    assert await _snapshot(binding) == before


@pytest.mark.asyncio
async def test_native_remember_migrates_and_dry_run_preserves_every_row(monkeypatch):
    import cognee
    from cognee.modules.weave import dataset_migration
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.context_global_variables import scoped_database_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.postgres import dataset_schema_name
    from cognee.modules.users.methods import get_user
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.modules.weave.memory_sources import assert_native_completed
    from cognee.modules.weave.config import get_weave_llm_config, get_weave_embedding_config
    from cognee.modules.data.models import Data

    binding, legacy, synthetic_data_id, *_ = await _fixture()
    native_id = uuid4()
    schema = dataset_schema_name(legacy.id)
    # The manual fixture uses a tiny vector. This test uses native collections
    # with the configured dimensions and mocked embedding responses instead.
    async with get_relational_engine().get_async_session() as session:
        await session.execute(text(f'DROP TABLE "{schema}"."DocumentChunk_text"'))
        # A synthetic storage-only Data row has no raw file for the native
        # cognify pipeline. The native remember below creates the real row.
        from sqlalchemy import delete

        await session.execute(delete(Data).where(Data.id == synthetic_data_id))
        await session.commit()
    calls = []

    async def model(*args, **kwargs):
        cls = kwargs.get("response_model") or args[2]
        calls.append(cls.__name__)
        if cls.__name__ == "KnowledgeGraph":
            return cls(
                nodes=[
                    dict(
                        id="MessagePayment",
                        name="MessagePayment",
                        type="function",
                        description="main.go returns a payment message",
                    )
                ],
                edges=[],
            )
        if cls.__name__ == "SummarizedContent":
            return cls(summary="Historical main.go MessagePayment returns a payment message.")
        raise AssertionError(f"Unexpected model stage {cls.__name__}")

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", model)
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    user = await get_user(binding.service_user_id)
    llm, embedding = get_weave_llm_config(), get_weave_embedding_config()
    async with scoped_database_context_variables(
        legacy.id, user.id, llm_config=llm, embedding_config=embedding
    ):
        assert_native_completed(
            await cognee.remember(
                DataItem(
                    data="Historical main.go MessagePayment returns a payment message.",
                    label="native-history.txt",
                    data_id=native_id,
                ),
                dataset_id=legacy.id,
                user=user,
                self_improvement=False,
                llm_config=llm,
                embedding_config=embedding,
            )
        )
    assert "KnowledgeGraph" in calls
    # Initialize the primary through the same native migration gate.
    async with scoped_database_context_variables(binding.dataset_id, user.id):
        await get_graph_engine()
    before = await _snapshot(binding)
    native_count = len(before[str(legacy.id)]["graph_node"])
    async with get_relational_engine().get_async_session() as session:
        assert (await session.get(Data, native_id)).dataset_id == legacy.id
    calls_before = list(calls)
    dry = await dataset_migration.migrate_customer_datasets(binding.organization_id, dry_run=True)
    assert dry["dry_run"] is True
    assert dry["migrated_datasets"] == [str(legacy.id)]
    assert await _snapshot(binding) == before
    await dataset_migration.migrate_customer_datasets(binding.organization_id)
    after = await _snapshot(binding)
    assert len(after[str(binding.dataset_id)]["graph_node"]) >= native_count
    assert calls == calls_before
    async with get_relational_engine().get_async_session() as session:
        assert (await session.get(Data, native_id)).dataset_id == binding.dataset_id
    async with scoped_database_context_variables(
        binding.dataset_id, user.id, llm_config=llm, embedding_config=embedding
    ):
        await cognee.forget(data_id=native_id, dataset_id=binding.dataset_id, user=user)
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    assert not any("MessagePayment" in str(properties) for _, properties in nodes)
    assert any(properties.get("text") == "keep me" for _, properties in nodes)
