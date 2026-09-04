import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.context_global_variables as context_variables

from cognee.context_global_variables import (
    current_dataset_id,
    embedding_config,
    graph_db_config,
    llm_config,
    set_database_global_context_variables,
    strict_database_scope,
    vector_db_config,
)
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.infrastructure.llm.config import LLMConfig


def _install_dataset_context_fakes(monkeypatch, databases):
    users = {
        database.owner_id: SimpleNamespace(id=database.owner_id, tenant_id=None)
        for database in databases.values()
    }

    async def fake_get_user(user_id):
        return users[user_id]

    async def fake_get_or_create_dataset_database(dataset_id, _user):
        return databases[dataset_id]

    async def fake_resolve_connection_info(dataset_database):
        return dataset_database

    class FakeDatasetQueue:
        async def ensure_slot(self, dataset):
            pass

        async def release_slot_for(self, dataset):
            pass

    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: True
    )
    monkeypatch.setattr("cognee.context_global_variables.get_user", fake_get_user)
    monkeypatch.setattr(
        "cognee.context_global_variables.get_or_create_dataset_database",
        fake_get_or_create_dataset_database,
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.resolve_dataset_database_connection_info",
        fake_resolve_connection_info,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue", FakeDatasetQueue
    )


def _dataset_database(dataset_id, user_id, schema):
    return SimpleNamespace(
        owner_id=user_id,
        vector_database_provider="pgvector",
        vector_database_url="",
        vector_database_key="",
        vector_database_name="cognee_db",
        vector_database_connection_info={
            "host": "postgres",
            "port": "5432",
            "username": "cognee",
            "password": "cognee",
            "schema": schema,
        },
        graph_database_provider="postgres_demo",
        graph_database_url="",
        graph_database_key="",
        graph_database_name="cognee_db",
        graph_database_connection_info={
            "graph_database_host": "postgres",
            "graph_database_port": "5432",
            "graph_database_username": "cognee",
            "graph_database_password": "cognee",
            "graph_database_schema": schema,
        },
        graph_dataset_database_handler="postgres_graph_shared",
    )


@pytest.mark.asyncio
async def test_database_context_sets_and_resets_current_dataset_id(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    current_dataset_id.set("outer")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    async with set_database_global_context_variables(dataset_id, user_id):
        assert current_dataset_id.get() == dataset_id

    assert current_dataset_id.get() == "outer"


@pytest.mark.asyncio
async def test_llm_and_embedding_config_reset_on_exit(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    outer_llm_config = LLMConfig(llm_model="outer-model")
    llm_config.set(outer_llm_config)
    inner_llm_config = LLMConfig(llm_model="inner-model")
    inner_embedding_config = EmbeddingConfig()

    async with set_database_global_context_variables(
        uuid4(),
        uuid4(),
        llm_config=inner_llm_config,
        embedding_config=inner_embedding_config,
    ):
        assert llm_config.get() is inner_llm_config
        assert embedding_config.get() is inner_embedding_config

    assert llm_config.get() is outer_llm_config
    assert embedding_config.get() is None


@pytest.mark.asyncio
async def test_llm_and_embedding_config_reset_on_exception(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with pytest.raises(RuntimeError):
        async with set_database_global_context_variables(
            uuid4(),
            uuid4(),
            llm_config=LLMConfig(llm_model="inner-model"),
            embedding_config=EmbeddingConfig(),
        ):
            raise RuntimeError("boom")

    assert llm_config.get() is None
    assert embedding_config.get() is None


@pytest.mark.asyncio
async def test_dataset_database_configs_persist_after_exit(monkeypatch):
    """graph/vector/file-storage configs intentionally persist after exit.

    Callers (and integration tests) read the per-dataset databases right after
    a pipeline run, outside the ``async with`` block; only the LLM/embedding
    overrides and the dataset id are restored on exit.
    """
    dataset_id = uuid4()
    user_id = uuid4()
    fake_user = SimpleNamespace(id=user_id, tenant_id=None)
    fake_dataset_database = SimpleNamespace(
        vector_database_provider="lancedb",
        vector_database_url="",
        vector_database_key="",
        vector_database_name="test_vector_db",
        vector_database_connection_info={},
        graph_database_provider="ladybug",
        graph_database_url="",
        graph_database_key="",
        graph_database_name="test_graph_db",
        graph_database_connection_info={},
        graph_dataset_database_handler="ladybug",
    )

    async def fake_get_user(_user_id):
        return fake_user

    async def fake_get_or_create_dataset_database(_dataset, _user):
        return fake_dataset_database

    async def fake_resolve_connection_info(dataset_database):
        return dataset_database

    class FakeDatasetQueue:
        async def ensure_slot(self, dataset):
            pass

        async def release_slot_for(self, dataset):
            pass

    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: True
    )
    monkeypatch.setattr("cognee.context_global_variables.get_user", fake_get_user)
    monkeypatch.setattr(
        "cognee.context_global_variables.get_or_create_dataset_database",
        fake_get_or_create_dataset_database,
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.resolve_dataset_database_connection_info",
        fake_resolve_connection_info,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue", FakeDatasetQueue
    )

    async with set_database_global_context_variables(dataset_id, user_id):
        assert graph_db_config.get()["graph_database_name"] == "test_graph_db"
        assert vector_db_config.get()["vector_db_name"] == "test_vector_db"
        assert file_storage_config.get() is not None

    assert graph_db_config.get()["graph_database_name"] == "test_graph_db"
    assert vector_db_config.get()["vector_db_name"] == "test_vector_db"
    assert file_storage_config.get() is not None


@pytest.mark.asyncio
async def test_strict_database_context_restores_outer_database_configs(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    schema = f"ds_{dataset_id.hex}"
    _install_dataset_context_fakes(
        monkeypatch,
        {dataset_id: _dataset_database(dataset_id, user_id, schema)},
    )
    outer_graph = {"graph_database_name": "outer_graph"}
    outer_vector = {"vector_db_name": "outer_vector"}
    outer_storage = {"data_root_directory": "/outer"}
    graph_db_config.set(outer_graph)
    vector_db_config.set(outer_vector)
    file_storage_config.set(outer_storage)

    if not hasattr(context_variables, "scoped_database_context_variables"):
        pytest.fail("strict database context API is missing")

    async with context_variables.scoped_database_context_variables(dataset_id, user_id):
        assert strict_database_scope.get() is True
        assert graph_db_config.get()["graph_database_schema"] == schema
        assert vector_db_config.get()["vector_db_schema"] == schema
        assert file_storage_config.get() is not outer_storage

    assert graph_db_config.get() is outer_graph
    assert vector_db_config.get() is outer_vector
    assert file_storage_config.get() is outer_storage
    assert strict_database_scope.get() is False


@pytest.mark.asyncio
async def test_strict_database_context_cleans_up_when_entry_fails(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    released = []

    async def failing_get_user(_user_id):
        raise RuntimeError("user lookup failed")

    class FakeDatasetQueue:
        async def ensure_slot(self, _dataset):
            pass

        async def release_slot_for(self, dataset):
            released.append(dataset)

    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: True
    )
    monkeypatch.setattr("cognee.context_global_variables.get_user", failing_get_user)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue", FakeDatasetQueue
    )
    current_dataset_id.set("outer")

    with pytest.raises(RuntimeError, match="user lookup failed"):
        async with context_variables.scoped_database_context_variables(dataset_id, user_id):
            pass

    assert current_dataset_id.get() == "outer"
    assert released == [dataset_id]


@pytest.mark.asyncio
async def test_strict_database_context_restores_nested_scope(monkeypatch):
    outer_dataset = uuid4()
    inner_dataset = uuid4()
    outer_user = uuid4()
    inner_user = uuid4()
    outer_schema = f"ds_{outer_dataset.hex}"
    inner_schema = f"ds_{inner_dataset.hex}"
    _install_dataset_context_fakes(
        monkeypatch,
        {
            outer_dataset: _dataset_database(outer_dataset, outer_user, outer_schema),
            inner_dataset: _dataset_database(inner_dataset, inner_user, inner_schema),
        },
    )

    async with context_variables.scoped_database_context_variables(outer_dataset, outer_user):
        outer_graph = graph_db_config.get()
        outer_vector = vector_db_config.get()
        async with context_variables.scoped_database_context_variables(inner_dataset, inner_user):
            assert graph_db_config.get()["graph_database_schema"] == inner_schema
            assert vector_db_config.get()["vector_db_schema"] == inner_schema
        assert graph_db_config.get() is outer_graph
        assert vector_db_config.get() is outer_vector
        assert current_dataset_id.get() == outer_dataset


@pytest.mark.asyncio
async def test_strict_database_context_restores_after_cancellation(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    schema = f"ds_{dataset_id.hex}"
    _install_dataset_context_fakes(
        monkeypatch,
        {dataset_id: _dataset_database(dataset_id, user_id, schema)},
    )
    outer_graph = {"graph_database_name": "outer_graph"}
    graph_db_config.set(outer_graph)
    current_dataset_id.set("outer")

    with pytest.raises(asyncio.CancelledError):
        async with context_variables.scoped_database_context_variables(dataset_id, user_id):
            raise asyncio.CancelledError()

    assert graph_db_config.get() is outer_graph
    assert current_dataset_id.get() == "outer"


@pytest.mark.asyncio
async def test_concurrent_strict_database_contexts_keep_distinct_schemas(monkeypatch):
    first_dataset = uuid4()
    second_dataset = uuid4()
    first_user = uuid4()
    second_user = uuid4()
    first_schema = f"ds_{first_dataset.hex}"
    second_schema = f"ds_{second_dataset.hex}"
    _install_dataset_context_fakes(
        monkeypatch,
        {
            first_dataset: _dataset_database(first_dataset, first_user, first_schema),
            second_dataset: _dataset_database(second_dataset, second_user, second_schema),
        },
    )
    both_entered = asyncio.Barrier(2)

    async def observe(dataset_id, user_id):
        async with context_variables.scoped_database_context_variables(dataset_id, user_id):
            await both_entered.wait()
            return (
                graph_db_config.get()["graph_database_schema"],
                vector_db_config.get()["vector_db_schema"],
                current_dataset_id.get(),
            )

    first, second = await asyncio.gather(
        observe(first_dataset, first_user),
        observe(second_dataset, second_user),
    )

    assert first == (first_schema, first_schema, first_dataset)
    assert second == (second_schema, second_schema, second_dataset)


@pytest.mark.asyncio
async def test_dataset_name_is_rejected(monkeypatch):
    """Only a dataset id (UUID or UUID string) may enter the database context.
    Names must be resolved by the caller first; the context variable is left
    untouched when entry is refused."""
    from cognee.exceptions import CogneeValidationError

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    current_dataset_id.set(None)

    with pytest.raises(CogneeValidationError, match="main_dataset"):
        async with set_database_global_context_variables("main_dataset", uuid4()):
            pass

    assert current_dataset_id.get() is None


@pytest.mark.asyncio
async def test_uuid_string_dataset_is_rejected(monkeypatch):
    """One input type: even a valid UUID in string form is refused — callers
    hold real UUID objects, the boundary does no coercion."""
    from cognee.exceptions import CogneeValidationError

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with pytest.raises(CogneeValidationError, match="must be a dataset id"):
        async with set_database_global_context_variables(str(uuid4()), uuid4()):
            pass
