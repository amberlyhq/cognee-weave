from uuid import uuid4
from types import SimpleNamespace

import pytest


def test_storage_exposes_host_selected_operations_without_agent_endpoints():
    from cognee.api.v1.weave.routers.get_weave_router import get_weave_router

    paths = {route.path for route in get_weave_router().routes}
    assert (
        "/organizations/{organization_id}/repositories/{github_repository_id}/memory-notes/apply"
        in paths
    )
    assert "/organizations/{organization_id}/reviews/remember" not in paths
    assert "/organizations/{organization_id}/repositories/merge-memory" not in paths


def test_note_operations_accept_full_content_and_arbitrary_semantics():
    from cognee.modules.weave.agent_memory_contracts import MemoryApplyRequest

    req = MemoryApplyRequest(
        job_id=uuid4(),
        lifecycle_generation=1,
        source_kind="review",
        source_id="review:1",
        source_sha="a" * 40,
        operations=[
            dict(
                operation_id=uuid4(),
                note_id=uuid4(),
                action="add",
                expected_version=0,
                content="Even a wrong claim is an agent decision. " * 10000,
                source_paths=["src/pay.ts"],
                reason="qualified by agent",
            )
        ],
    )
    assert len(req.operations[0].content) > 28000


def test_note_eligibility_uses_scoped_completed_native_source():
    from cognee.modules.weave.memory_retrieval import eligible_memory_sources, native_source_tag

    org, dataset, data = uuid4(), uuid4(), uuid4()
    binding = SimpleNamespace(organization_id=org, dataset_id=dataset)
    note = SimpleNamespace(
        organization_id=org,
        dataset_id=dataset,
        data_id=data,
        content_hash="abc",
        status="completed",
        session_id=None,
        github_repository_id=12,
        qualification={"memory_note": True, "native_source_tag": native_source_tag(data, "abc")},
    )
    assert eligible_memory_sources(binding, [note], [12]) == [note]
    assert eligible_memory_sources(binding, [note], [13]) == []
    note.status = "processing"
    assert eligible_memory_sources(binding, [note], [12]) == []


def test_fresh_schema_bootstrap_enforces_rls_for_all_note_receipts():
    from cognee.modules.weave.rls import RLS_TABLES

    assert {"weave_memory_notes", "weave_memory_jobs", "weave_memory_operations"} <= set(RLS_TABLES)
