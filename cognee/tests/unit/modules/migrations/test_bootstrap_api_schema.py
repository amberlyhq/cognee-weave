"""Fresh CLI bootstrap must register the same relational models as the API."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_fresh_bootstrap_creates_api_tables_before_stamping_head():
    # A fresh interpreter matters: another test importing API routers must not
    # accidentally register missing models before the bootstrap under test.
    script = textwrap.dedent(
        """
        import asyncio
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, patch
        from sqlalchemy import create_mock_engine
        from cognee.infrastructure.databases.relational import Base
        from cognee.modules.migrations import startup

        statements = []
        engine = create_mock_engine(
            'postgresql://',
            lambda sql, *args, **kwargs: statements.append(str(sql.compile(dialect=engine.dialect))),
        )
        created_tables = set()

        class BootstrapEngine:
            async def create_database(self):
                Base.metadata.create_all(engine)
                created_tables.update(Base.metadata.tables)

        @asynccontextmanager
        async def no_database_lock():
            yield

        async def verify_stamp(*args):
            # Match what the real server imports, AFTER capturing the schema
            # created by the standalone CLI bootstrap.
            import cognee.api.client
            missing = set(Base.metadata.tables) - created_tables
            assert not missing, f'API tables absent before stamp head: {sorted(missing)}'
            assert any('CREATE TYPE syncstatus' in ddl for ddl in statements), statements

        with (
            patch.object(startup, '_relational_schema_exists', AsyncMock(return_value=False)),
            patch.object(startup, 'run_relational_stamp', verify_stamp),
            patch('cognee.infrastructure.databases.relational.get_relational_engine', return_value=BootstrapEngine()),
            patch('cognee.modules.migrations.runner.migration_lock', no_database_lock),
            patch('cognee.modules.migrations.runner.run_database_migrations', AsyncMock(return_value=[])),
            patch('cognee.modules.weave.rls.ensure_weave_rls_policies', AsyncMock()),
        ):
            asyncio.run(startup.apply_all_migrations())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[5],
        env={**os.environ, "ENV": "test", "COGNEE_SKIP_CONNECTION_TEST": "true"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
