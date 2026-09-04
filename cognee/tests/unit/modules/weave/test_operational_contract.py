from pathlib import Path
from unittest.mock import AsyncMock

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
    assert "ensure_weave_rls_policies" in startup
    assert startup.index(fresh_create) < startup.index("await ensure_weave_rls_policies()")


@pytest.mark.asyncio
async def test_strict_runtime_pipeline_does_not_attempt_global_relational_ddl(monkeypatch):
    from cognee.modules.pipelines.layers import setup_and_check_environment as setup_module

    relational_create = AsyncMock()
    vector_create = AsyncMock()
    monkeypatch.setenv("WEAVE_STRICT_MODE", "true")
    monkeypatch.setattr(setup_module, "create_relational_db_and_tables", relational_create)
    monkeypatch.setattr(setup_module, "create_pgvector_db_and_tables", vector_create)
    monkeypatch.setattr(setup_module, "_first_run_done", True)

    await setup_module.setup_and_check_environment(skip_connection_test=True)

    relational_create.assert_not_awaited()
    vector_create.assert_awaited_once_with()


def test_rls_covers_every_shared_control_plane_table_on_fresh_and_existing_databases():
    original_migration = (
        ROOT / "cognee/alembic/versions/d1e3f5a7b9c2_add_weave_organization_control_plane.py"
    ).read_text()
    forward_migration_path = (
        ROOT / "cognee/alembic/versions/f3a5c7e9b1d4_secure_weave_runtime_role.py"
    )
    assert forward_migration_path.exists()
    forward_migration = forward_migration_path.read_text()
    deletion_migration_path = (
        ROOT
        / "cognee/alembic/versions/a5c7e9b1d3f6_add_scoped_weave_schema_deletion.py"
    )
    assert deletion_migration_path.exists()
    deletion_migration = deletion_migration_path.read_text()
    runtime_security = (ROOT / "cognee/modules/weave/rls.py").read_text()
    postgres_admin = (ROOT / "cognee/infrastructure/databases/postgres/admin.py").read_text()
    compose = (ROOT / "deployment/docker-compose.weave.yml").read_text()
    init = (ROOT / "deployment/init-weave-postgres.sh").read_text()

    assert '"weave_organization_bindings"' not in original_migration.partition("def upgrade")[0]
    assert 'down_revision: Union[str, None] = "e2f4a6b8c0d3"' in forward_migration
    assert 'table_name = "weave_organization_bindings"' in forward_migration
    assert "ENABLE ROW LEVEL SECURITY" in forward_migration
    assert "FORCE ROW LEVEL SECURITY" in forward_migration
    assert "weave_organization_bindings_organization_isolation" in forward_migration
    assert "ALTER DATABASE" in forward_migration
    assert "current_database()" in forward_migration
    assert "op.get_context().autocommit_block()" in forward_migration
    assert "POSTGRES_USER: cognee_admin" in compose
    assert "DB_USERNAME: cognee" in compose
    assert 'ENABLE_AUTO_MIGRATIONS: "false"' in compose
    assert "service_completed_successfully" in compose
    assert "init-weave-postgres.sh" in compose
    assert "ALTER DATABASE cognee_db OWNER TO cognee" not in init
    assert "weave_create_dataset_schema" in init
    assert "weave_create_dataset_schema" in postgres_admin
    assert "weave_drop_organization_dataset_schema" in init
    assert "weave_drop_organization_dataset_schema" not in forward_migration
    assert 'down_revision: Union[str, None] = "f3a5c7e9b1d4"' in deletion_migration
    assert "weave_drop_organization_dataset_schema" in deletion_migration
    assert "weave_drop_organization_dataset_schema" in runtime_security
    for source in (init, deletion_migration, runtime_security):
        assert "primary_dataset_id" in source
        assert "app.weave_organization_id" in source
    assert 'os.getenv("WEAVE_STRICT_MODE") == "true"' in postgres_admin
    assert "datdba" in runtime_security
    assert "relowner" in runtime_security
    assert "rolbypassrls" in runtime_security
    assert runtime_security.count("FROM pg_class c CROSS JOIN pg_roles r ") == 1


def test_strict_mode_exposes_only_health_root_and_weave_routes():
    client = (ROOT / "cognee/api/client.py").read_text()
    assert "_restrict_to_weave_routes" in client
    assert '{"/", "/health", "/api/v1/weave"}' in client


def test_weave_index_api_pins_the_actual_extractor_version():
    router = (ROOT / "cognee/api/v1/weave/routers/get_weave_router.py").read_text()
    assert "ENOLA_PINNED_VERSION" in router
    assert 'extraction_version != f"enola-{ENOLA_PINNED_VERSION}"' in router


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
    assert "scripts/weave-parity.sh" in workflow
    assert "scripts/weave-secret-scan.sh" in workflow
    assert "ruff check --select E4,E7,E9,F --ignore F401 cognee" in workflow


def test_parity_keeps_admin_only_test_cleanup_out_of_the_runtime_service():
    parity = (ROOT / "scripts/weave-parity.sh").read_text()
    backup = (ROOT / "scripts/weave-backup.sh").read_text()
    restore = (ROOT / "scripts/weave-restore-drill.sh").read_text()
    compose = (ROOT / "deployment/docker-compose.weave.yml").read_text()
    assert "export WEAVE_STRICT_MODE=false" in parity
    assert "export DB_USERNAME=cognee_admin" in parity
    assert 'WEAVE_STRICT_MODE: "true"' in compose
    assert "DB_USERNAME: cognee" in compose
    assert "cross-organization schema deletion unexpectedly succeeded" in parity
    assert "weave_drop_organization_dataset_schema" in parity
    assert "pg_restore --username=cognee_admin" in restore
    assert "--no-acl" not in backup
    assert "--no-acl" not in restore
    assert "--no-owner" not in backup
    assert "--no-owner" not in restore


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
    assert 'WEAVE_STRICT_MODE: "true"' in compose
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "install_enola()" in dockerfile
    assert "AutoTokenizer.from_pretrained" in dockerfile
    assert "TextEmbedding" in dockerfile
    assert "weave image warmup" in dockerfile
    assert "ENOLA_AUTO_INSTALL=false" in dockerfile
    assert "ENOLA_PATH=/app/.cognee/bin/enola-0.3.13-linux-arm64" in dockerfile
    assert "ENOLA_PATH: /app/.cognee/bin/enola-0.3.13-linux-arm64" in compose
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
    assert "go.mod" in parity
    assert "func %s() string" in parity
    postgres_conftest = (ROOT / "cognee/tests/e2e/postgres/conftest.py").read_text()
    assert '@pytest.fixture(scope="session")' in postgres_conftest
    assert "asyncio.new_event_loop()" in postgres_conftest
    assert "pg_dump" in backup
    assert "umask 077" in backup
    assert "cognee-weave-restore-" in restore
    assert "pg_restore" in restore
    assert "RESTORE_ORGANIZATION_ID" in restore
    assert "RESTORE_EXPECTED_EXPORT" in restore
    assert "restored export does not match the backup source" in restore
