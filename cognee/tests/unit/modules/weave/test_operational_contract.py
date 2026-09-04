from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[5]


def strict_environment():
    environment = {
        "WEAVE_STRICT_MODE": "true",
        "WEAVE_INTERNAL_TOKEN": "a" * 32,
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "DB_PROVIDER": "postgres",
        "DB_HOST": "postgres",
        "DB_PORT": "5432",
        "DB_USERNAME": "cognee",
        "DB_PASSWORD": "secret",
        "DB_NAME": "cognee_db",
        "VECTOR_DB_PROVIDER": "pgvector",
        "VECTOR_DATASET_DATABASE_HANDLER": "pgvector_shared",
        "GRAPH_DATABASE_PROVIDER": "postgres_demo",
        "GRAPH_DATASET_DATABASE_HANDLER": "postgres_graph_shared",
    }
    for prefix in ("VECTOR_DB", "GRAPH_DATABASE"):
        environment.update(
            {
                f"{prefix}_HOST": environment["DB_HOST"],
                f"{prefix}_PORT": environment["DB_PORT"],
                f"{prefix}_USERNAME": environment["DB_USERNAME"],
                f"{prefix}_PASSWORD": environment["DB_PASSWORD"],
                f"{prefix}_NAME": environment["DB_NAME"],
            }
        )
    return environment


def test_strict_weave_runtime_accepts_only_the_all_postgres_tenant_boundary():
    from cognee.modules.weave.config import validate_weave_runtime_environment

    validate_weave_runtime_environment(strict_environment())
    for key, invalid in (
        ("WEAVE_INTERNAL_TOKEN", "short"),
        ("ENABLE_BACKEND_ACCESS_CONTROL", "false"),
        ("DB_PROVIDER", "sqlite"),
        ("VECTOR_DB_PROVIDER", "lancedb"),
        ("VECTOR_DATASET_DATABASE_HANDLER", "pgvector"),
        ("GRAPH_DATABASE_PROVIDER", "neo4j"),
        ("GRAPH_DATASET_DATABASE_HANDLER", "neo4j_community"),
        ("GRAPH_DATABASE_HOST", "another-postgres"),
        ("VECTOR_DB_NAME", "another_database"),
    ):
        with pytest.raises(ValueError, match=key):
            validate_weave_runtime_environment({**strict_environment(), key: invalid})


def test_container_startup_enforces_strict_mode_before_migrations():
    entrypoint = (ROOT / "entrypoint.sh").read_text()
    validation = "validate_weave_runtime_environment"
    assert validation in entrypoint
    assert entrypoint.index(validation) < entrypoint.index("run_migrations")


def test_fresh_database_registers_weave_models_before_create_all():
    startup = (ROOT / "cognee/modules/migrations/startup.py").read_text()
    model_import = "import cognee.modules.weave.models"
    fresh_create = "await get_relational_engine().create_database()"
    assert model_import in startup
    assert startup.index(model_import) < startup.index(fresh_create)


def test_weave_ci_runs_every_fork_specific_postgres_gate():
    workflow = (ROOT / ".github/workflows/weave-gate.yml").read_text()
    for path in (
        "cognee/tests/unit/modules/weave",
        "cognee/tests/unit/api/v1/weave/test_internal_auth.py",
        "cognee/tests/e2e/postgres/test_shared_schema_isolation.py",
        "cognee/tests/e2e/postgres/test_shared_schema_concurrency.py",
        "cognee/tests/e2e/postgres/test_tenant_graph_retrieval.py",
        "cognee/tests/e2e/postgres/test_pgvector_hnsw_plan.py",
        "cognee/tests/e2e/postgres/test_weave_exact_sha_indexing.py",
        "cognee/tests/e2e/postgres/test_weave_hybrid_recall.py",
        "cognee/tests/e2e/postgres/test_weave_surface_isolation.py",
    ):
        assert path in workflow


def test_weekly_upstream_sync_opens_a_manual_review_pr_without_auto_merge():
    workflow = (ROOT / ".github/workflows/upstream-sync.yml").read_text()
    assert "schedule:" in workflow
    assert "https://github.com/topoteretes/cognee.git" in workflow
    assert "git rev-parse upstream/main" in workflow
    assert 'git switch --create "${branch}" origin/main' in workflow
    assert 'git merge --no-edit "${UPSTREAM_SHA}"' in workflow
    assert "gh pr create" in workflow
    assert "gh pr merge" not in workflow
    assert "--auto" not in workflow


def test_operations_and_parity_docs_keep_neo4j_out_of_the_runtime():
    operations = (ROOT / "docs/weave/operations.md").read_text()
    parity = (ROOT / "docs/weave/parity.md").read_text()
    compose = (ROOT / "deployment/docker-compose.weave.yml").read_text()
    assert "GRAPH_DATABASE_PROVIDER=postgres_demo" in operations
    assert "VECTOR_DB_PROVIDER=pgvector" in operations
    assert "Neo4j is not deployed" in operations
    assert "Production parity is not claimed" in parity
    assert "zero cross-organization candidates" in parity
    assert "pgvector/pgvector:0.8.6-pg17-bookworm" in compose
    assert "image: cognee-weave-parity:local" in compose
    assert "WEAVE_PARITY_SKIP_BUILD" in parity
    assert "WEAVE_STRICT_MODE: \"true\"" in compose
    for prefix in ("VECTOR_DB", "GRAPH_DATABASE"):
        for suffix in ("HOST", "PORT", "USERNAME", "PASSWORD", "NAME"):
            assert f"{prefix}_{suffix}:" in compose
    assert "actions/checkout@v" not in (ROOT / ".github/workflows/weave-gate.yml").read_text()
    assert "astral-sh/setup-uv@v" not in (ROOT / ".github/workflows/weave-gate.yml").read_text()


def test_parity_backup_and_restore_scripts_are_fail_closed():
    parity = (ROOT / "scripts/weave-parity.sh").read_text()
    backup = (ROOT / "scripts/weave-backup.sh").read_text()
    restore = (ROOT / "scripts/weave-restore-drill.sh").read_text()
    assert "set -euo pipefail" in parity
    assert "test_weave_surface_isolation.py" in parity
    assert "fixture_index_seconds_a" in parity
    assert "fixture_index_seconds_b" in parity
    assert "math.ceil" in parity
    assert "262144000" in parity
    assert "logs --no-color" in parity
    assert 'organization_a="7e1a7b9d-08c2-4f57-9884-623e01b68b01"' in parity
    assert 'organization_b="7e1a7b9d-08c2-4f57-9884-623e01b68b02"' in parity
    assert 'go.mod' in parity
    assert 'func %s() string' in parity
    postgres_conftest = (ROOT / "cognee/tests/e2e/postgres/conftest.py").read_text()
    assert '@pytest.fixture(scope="session")' in postgres_conftest
    assert "asyncio.new_event_loop()" in postgres_conftest
    assert "pg_dump" in backup
    assert "umask 077" in backup
    assert "cognee-weave-restore-" in restore
    assert "pg_restore" in restore
    assert "RESTORE_ORGANIZATION_ID" in restore
    assert 'assert value["edges"]' in restore
