from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest


@asynccontextmanager
async def unlocked(*args):
    yield


@pytest.mark.asyncio
async def test_repository_is_marked_deleting_before_native_cleanup_can_fail(monkeypatch):
    from cognee.modules.weave import deletion, native_memory

    calls = []

    async def newer(*args, **kwargs):
        return True

    async def binding(*args):
        return SimpleNamespace()

    async def pending(*args):
        calls.append("pending")

    async def forget(*args):
        calls.append("forget")
        raise RuntimeError("failure after last native source deletion")

    monkeypatch.setattr(deletion, "weave_operation_lock", unlocked)
    monkeypatch.setattr(deletion, "_repository_transition_is_newer", newer)
    monkeypatch.setattr(deletion, "get_organization_binding", binding)
    monkeypatch.setattr(deletion, "_mark_repository_deleting", pending, raising=False)
    monkeypatch.setattr(native_memory, "forget_repository", forget)
    with pytest.raises(RuntimeError, match="last native source"):
        await deletion.delete_repository(uuid4(), 123, 2)
    assert calls == ["pending", "forget"]


@pytest.mark.asyncio
async def test_filtered_legacy_export_still_rejects_a_mixed_customer(monkeypatch):
    from cognee.modules.weave import deletion, native_memory

    legacy = SimpleNamespace(pipeline_version="weave-native-memory.v1")
    current = SimpleNamespace(
        pipeline_version=native_memory.NATIVE_PIPELINE_VERSION, status="running"
    )

    async def binding(*args):
        return SimpleNamespace()

    async def selected(*args):
        return [legacy]

    async def all_records(*args):
        return [legacy, current]

    monkeypatch.setattr(deletion, "get_organization_binding", binding)
    monkeypatch.setattr(deletion, "_surface_snapshots", selected)
    monkeypatch.setattr(native_memory, "customer_snapshots", all_records)
    with pytest.raises(deletion.SurfaceNotFound):
        await deletion._read_locked_surface(uuid4(), [123], "export")
