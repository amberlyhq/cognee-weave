from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.source_refs import (
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.databases.provenance.source_ref_state import provenance_after_attach
from cognee.modules.graph.methods.transfer_consolidated_ownership import _transfer_graph_ownership


@pytest.mark.asyncio
async def test_graph_ownership_transfer_preserves_original_run_and_is_repeatable():
    dataset, data, run, other_run = uuid4(), uuid4(), uuid4(), uuid4()
    key = make_source_ref_key(dataset, data)
    record = SimpleNamespace(
        source_ref_keys=[key],
        source_run_refs=[make_source_run_ref(run, key), make_source_run_ref(other_run, key)],
    )
    old, canonical, neighbor = map(str, [uuid4(), uuid4(), uuid4()])
    old_edge, new_edge = (
        EdgeIdentity(old, neighbor, "uses"),
        EdgeIdentity(canonical, neighbor, "uses"),
    )
    graph = AsyncMock()
    graph.get_node_delete_data.return_value = {old: record}
    graph.get_edge_delete_data.return_value = {old_edge: record}
    state = {}

    async def attach(targets, keys, pipeline_run_id=None, source_run_refs=None):
        for target in targets:
            previous = state.get(target)
            state[target] = provenance_after_attach(
                previous.source_ref_keys if previous else [],
                previous.source_run_refs if previous else [],
                keys,
                pipeline_run_id,
                source_run_refs,
            )

    graph.attach_node_source_refs.side_effect = attach
    graph.attach_edge_source_refs.side_effect = attach
    for _ in range(2):
        await _transfer_graph_ownership(
            graph, dataset, {old: canonical}, [(old, neighbor, "uses", {})]
        )
        for target in (canonical, new_edge):
            assert state[target].source_ref_keys == [key]
            assert state[target].source_run_refs == [
                make_source_run_ref(run, key),
                make_source_run_ref(other_run, key),
            ]
    assert old not in state and old_edge not in state


@pytest.mark.asyncio
async def test_graph_transfer_rejects_foreign_ownership_before_attach():
    graph = AsyncMock()
    old, canonical = str(uuid4()), str(uuid4())
    graph.get_node_delete_data.return_value = {
        old: SimpleNamespace(
            source_ref_keys=[make_source_ref_key(uuid4(), uuid4())],
            source_run_refs=[],
        )
    }
    with pytest.raises(ValueError, match="another dataset"):
        await _transfer_graph_ownership(graph, uuid4(), {old: canonical}, [])
    graph.attach_node_source_refs.assert_not_called()


def test_explicit_history_unions_runs_but_ordinary_attach_keeps_first_introduction():
    dataset, data, other_data = uuid4(), uuid4(), uuid4()
    key, other_key = make_source_ref_key(dataset, data), make_source_ref_key(dataset, other_data)
    first, second, third = uuid4(), uuid4(), uuid4()
    original = make_source_run_ref(first, key)
    inherited = make_source_run_ref(second, key)
    new = make_source_run_ref(third, other_key)
    ordinary = provenance_after_attach([key], [original], [key], str(second))
    assert ordinary.source_run_refs == [original]
    transferred = provenance_after_attach(
        [key], [original], [key, other_key], None, [inherited, new]
    )
    assert transferred.source_ref_keys == [key, other_key]
    assert transferred.source_run_refs == [original, inherited, new]
    assert (
        provenance_after_attach(
            transferred.source_ref_keys,
            transferred.source_run_refs,
            [key, other_key],
            None,
            [inherited, new],
        )
        == transferred
    )
    with pytest.raises(ValueError, match="supplied source key"):
        provenance_after_attach([key], [original], [key], None, [new])
