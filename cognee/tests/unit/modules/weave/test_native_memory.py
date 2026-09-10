"""The Weave boundary delegates memory semantics to the upstream public API."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


def test_archive_indexing_uses_native_remember_not_custom_tasks():
    from cognee.modules.weave import indexing
    import inspect

    source = inspect.getsource(indexing.index_repository_archive)
    assert "remember_repository" in source
    assert "run_custom_pipeline" not in source
    assert "get_code_graph_tasks" not in source


def test_native_runtime_configures_the_internal_llm():
    root = Path(__file__).resolve().parents[5]
    compose = (root / "deployment/docker-compose.weave.yml").read_text()
    assert "LLM_MODEL: openrouter/openai/gpt-oss-120b" in compose
    assert "LLM_API_KEY:" in compose


def test_visualization_can_return_native_memory_graph_without_symbol_conversion():
    from cognee.modules.weave.contracts import SurfaceResponse

    assert "native_graph" in SurfaceResponse.model_fields


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "running"},
        {"status": "failed"},
        {"status": "queued"},
        {"pipeline_version": "weave-code.v1"},
        {"indexed_sha": None},
        {"requested_sha": "b" * 40},
    ],
)
def test_partial_stale_and_legacy_indexes_cannot_be_recalled(changes):
    from cognee.modules.weave.native_memory import NATIVE_PIPELINE_VERSION, snapshots_ready

    values = dict(
        status="indexed",
        pipeline_version=NATIVE_PIPELINE_VERSION,
        indexed_sha="a" * 40,
        requested_sha="a" * 40,
    )
    assert snapshots_ready([SimpleNamespace(**values)])
    assert not snapshots_ready([SimpleNamespace(**{**values, **changes})])
    assert not snapshots_ready([])


def test_repository_dataset_names_separate_organizations_and_repositories():
    from cognee.modules.weave.native_memory import repository_dataset_name

    a, b = uuid4(), uuid4()
    assert (
        len(
            {
                repository_dataset_name(a, 1),
                repository_dataset_name(a, 2),
                repository_dataset_name(b, 1),
            }
        )
        == 3
    )


def test_internal_llm_requires_zdr(monkeypatch):
    from cognee.modules.weave.config import get_weave_llm_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    assert get_weave_llm_config().llm_args["extra_body"]["provider"]["zdr"] is True


@pytest.mark.parametrize("stage", ["extraction", "summarization", "query"])
def test_stage_environment_cannot_redirect_native_weave(stage, monkeypatch):
    from cognee.modules.weave.config import get_weave_llm_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv(f"LLM_{stage.upper()}_MODEL", "openai/other")
    monkeypatch.setenv(f"LLM_{stage.upper()}_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv(f"LLM_{stage.upper()}_API_KEY", "wrong-key")
    config = get_weave_llm_config().stage_config(stage)
    assert config.llm_model == "openrouter/openai/gpt-oss-120b"
    assert config.llm_endpoint == "https://openrouter.ai/api/v1"
    assert config.llm_api_key != "wrong-key"


def test_admin_parity_storage_leg_is_offline():
    root = Path(__file__).resolve().parents[5]
    script = (root / "scripts/weave-parity.sh").read_text()
    admin_leg = script.split("export WEAVE_STRICT_MODE=false", 1)[1].split("if [ -x", 1)[0]
    assert "export MOCK_EMBEDDING=true" in admin_leg


@pytest.mark.asyncio
async def test_strict_schema_deletion_requires_a_native_organization_before_connecting(monkeypatch):
    from cognee.infrastructure.databases.postgres.admin import drop_pg_schema_if_exists

    monkeypatch.setenv("WEAVE_STRICT_MODE", "true")
    with pytest.raises(ValueError, match="organization scope"):
        await drop_pg_schema_if_exists(
            "unused", "ds_" + "a" * 32, "unused", 5432, "unused", "unused"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["extraction", "summarization", "query"])
async def test_native_memory_requests_prefer_cerebras_with_groq_fallback(monkeypatch, stage):
    from unittest.mock import AsyncMock, patch

    from cognee.context_global_variables import llm_config
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.weave.config import get_weave_llm_config
    from cognee.shared.data_models import SummarizedContent

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    config = get_weave_llm_config().stage_config(stage)
    token = llm_config.set(config)
    fake = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"summary":"Synthetic test."}'))
            ]
        )
    )
    try:
        with patch("litellm.acompletion", fake):
            result = await LLMGateway.acreate_structured_output(
                "Synthetic test.", "Summarize the input.", SummarizedContent
            )
    finally:
        llm_config.reset(token)
    assert result.summary == "Synthetic test."
    request = fake.call_args.kwargs
    assert request["model"] == "openrouter/openai/gpt-oss-120b"
    assert request["response_format"] is SummarizedContent
    assert request["extra_body"]["provider"] == {
        "order": ["cerebras", "groq"],
        "only": ["cerebras", "groq"],
        "allow_fallbacks": True,
        "require_parameters": True,
        "zdr": True,
    }
