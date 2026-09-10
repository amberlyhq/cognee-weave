"""The repository boundary hands the intact directory to native Cognee."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_repository_uses_one_native_directory_operation(tmp_path, monkeypatch, status):
    import cognee
    from cognee.modules.weave import native_memory, memory_sources, repository_sources
    import importlib

    creation = importlib.import_module("cognee.modules.data.methods.create_authorized_dataset")

    binding = SimpleNamespace(organization_id=uuid4(), dataset_id=uuid4(), service_user_id=uuid4())
    user = SimpleNamespace(id=binding.service_user_id, tenant_id=uuid4())
    dataset = SimpleNamespace(id=uuid4())
    request = SimpleNamespace(
        repository_owner="amberlyhq", repository_name="test", github_repository_id=42
    )
    repository = tmp_path / "archive-sha"
    repository.mkdir()
    (repository / "package.json").write_text('{"name":"test"}')
    (repository / "index.js").write_text("export const value = 1;")
    (repository / "README.md").write_text("Repository documentation")
    calls = []

    async def get_user(*args):
        return user

    async def get_dataset(*args):
        return dataset

    async def no_records(*args):
        return []

    async def forget(*args):
        calls.append("forget")

    async def prepared(*args, **kwargs):
        return [("code", SimpleNamespace(), "hash")]

    async def forbidden(*args, **kwargs):
        pytest.fail("Repository ingestion must not use per-file sync_source")

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield

    async def remember(data, **kwargs):
        from pathlib import Path

        directory = Path(data)
        assert directory.is_dir()
        assert (directory / "index.js").exists()
        assert (directory / "README.md").exists()
        assert kwargs == dict(
            dataset_id=dataset.id, user=user, llm_config="llm", embedding_config="embedding"
        )
        calls.append("remember")
        return SimpleNamespace(status=status)

    monkeypatch.setattr(native_memory, "get_user", get_user)
    monkeypatch.setattr(native_memory, "customer_dataset", get_dataset)
    monkeypatch.setattr(native_memory, "repository_dataset", get_dataset)
    monkeypatch.setattr(native_memory, "_forget_dataset", forget)
    monkeypatch.setattr(native_memory, "scoped_database_context_variables", scope)
    monkeypatch.setattr(native_memory, "get_weave_llm_config", lambda: "llm")
    monkeypatch.setattr(native_memory, "get_weave_embedding_config", lambda: "embedding")
    monkeypatch.setattr(creation, "create_authorized_dataset", get_dataset)
    monkeypatch.setattr(repository_sources, "prepare_repository_sources", prepared, raising=False)
    monkeypatch.setattr(memory_sources, "source_records", no_records)
    monkeypatch.setattr(memory_sources, "sync_source", forbidden)
    monkeypatch.setattr(cognee, "remember", remember)
    if status == "failed":
        with pytest.raises(RuntimeError, match="did not complete"):
            await native_memory.remember_repository(binding, request, repository)
    else:
        await native_memory.remember_repository(binding, request, repository)
    assert calls == ["forget", "remember"]
