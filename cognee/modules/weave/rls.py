from sqlalchemy import text

from cognee.infrastructure.databases.relational import get_relational_engine

RLS_TABLES = (
    "weave_organization_bindings",
    "weave_repository_snapshots",
    "weave_index_jobs",
)

_ORGANIZATION_PREDICATE = (
    "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
)


async def ensure_weave_rls_policies() -> None:
    """Install policies create_all cannot express on a fresh Postgres database."""

    engine = get_relational_engine()
    if engine.engine.dialect.name != "postgresql":
        return
    async with engine.get_async_session() as session:
        for table_name in RLS_TABLES:
            await session.execute(text(f'ALTER TABLE "{table_name}" ENABLE ROW LEVEL SECURITY'))
            await session.execute(text(f'ALTER TABLE "{table_name}" FORCE ROW LEVEL SECURITY'))
            await session.execute(
                text(
                    f'DROP POLICY IF EXISTS "{table_name}_organization_isolation" '
                    f'ON "{table_name}"'
                )
            )
            await session.execute(
                text(
                    f'CREATE POLICY "{table_name}_organization_isolation" ON "{table_name}" '
                    f"USING ({_ORGANIZATION_PREDICATE}) WITH CHECK ({_ORGANIZATION_PREDICATE})"
                )
            )
        await session.commit()


async def assert_weave_runtime_database_security() -> None:
    """Fail strict startup if the runtime can bypass the tenant policies."""

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        is_superuser = await session.scalar(
            text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        )
        if is_superuser:
            raise RuntimeError("Strict Weave runtime database role must not be a superuser")
        result = await session.execute(
            text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, "
                "EXISTS (SELECT 1 FROM pg_policies p WHERE p.tablename = c.relname) AS has_policy "
                "FROM pg_class c WHERE c.relname = ANY(:tables)"
            ),
            {"tables": list(RLS_TABLES)},
        )
        state = {row.relname: row for row in result}
        invalid = [
            table
            for table in RLS_TABLES
            if table not in state
            or not state[table].relrowsecurity
            or not state[table].relforcerowsecurity
            or not state[table].has_policy
        ]
        if invalid:
            raise RuntimeError(f"Strict Weave RLS is incomplete: {', '.join(invalid)}")
