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

_CREATE_SCHEMA_FUNCTION = """
CREATE OR REPLACE FUNCTION public.weave_create_dataset_schema(
    schema_name text,
    include_vector boolean DEFAULT false
) RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF schema_name !~ '^ds_[0-9a-f]{32}$' THEN
        RAISE EXCEPTION 'Invalid Weave dataset schema';
    END IF;
    IF include_vector AND NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'The vector extension is not installed';
    END IF;
    EXECUTE format(
        'CREATE SCHEMA IF NOT EXISTS %I AUTHORIZATION %I', schema_name, current_user
    );
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
        EXECUTE format('GRANT USAGE, CREATE ON SCHEMA %I TO cognee', schema_name);
    END IF;
END;
$function$
"""

_DROP_ORGANIZATION_SCHEMA_FUNCTION = """
CREATE OR REPLACE FUNCTION public.weave_drop_organization_dataset_schema(
    target_organization_id uuid
) RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    bound_dataset_id uuid;
    schema_name text;
BEGIN
    IF NULLIF(current_setting('app.weave_organization_id', true), '')::uuid
        IS DISTINCT FROM target_organization_id THEN
        RAISE EXCEPTION 'Weave organization scope mismatch';
    END IF;
    SELECT primary_dataset_id
      INTO bound_dataset_id
      FROM public.weave_organization_bindings
     WHERE organization_id = target_organization_id
       AND deleted_at IS NULL;
    IF bound_dataset_id IS NULL THEN
        RAISE EXCEPTION 'Active Weave organization binding not found';
    END IF;
    schema_name := 'ds_' || replace(bound_dataset_id::text, '-', '');
    EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', schema_name);
END;
$function$
"""

_RUNTIME_GRANTS = """
DO $grant$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
        GRANT EXECUTE ON FUNCTION public.weave_create_dataset_schema(text, boolean) TO cognee;
        GRANT EXECUTE ON FUNCTION public.weave_drop_organization_dataset_schema(uuid) TO cognee;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cognee;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cognee;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cognee;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT ON SEQUENCES TO cognee;
    END IF;
END
$grant$
"""


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
                    f'DROP POLICY IF EXISTS "{table_name}_organization_isolation" ON "{table_name}"'
                )
            )
            await session.execute(
                text(
                    f'CREATE POLICY "{table_name}_organization_isolation" ON "{table_name}" '
                    f"USING ({_ORGANIZATION_PREDICATE}) WITH CHECK ({_ORGANIZATION_PREDICATE})"
                )
            )
        await session.execute(text(_CREATE_SCHEMA_FUNCTION))
        await session.execute(text(_DROP_ORGANIZATION_SCHEMA_FUNCTION))
        await session.execute(
            text(
                "REVOKE ALL ON FUNCTION "
                "public.weave_create_dataset_schema(text, boolean) FROM PUBLIC"
            )
        )
        await session.execute(
            text(
                "REVOKE ALL ON FUNCTION "
                "public.weave_drop_organization_dataset_schema(uuid) FROM PUBLIC"
            )
        )
        await session.execute(text(_RUNTIME_GRANTS))
        await session.commit()


async def assert_weave_runtime_database_security() -> None:
    """Fail strict startup if the runtime can bypass the tenant policies."""

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        role = (
            (
                await session.execute(
                    text(
                        "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
                        "FROM pg_roles WHERE rolname = current_user"
                    )
                )
            )
            .mappings()
            .one()
        )
        if any(role.values()):
            raise RuntimeError("Strict Weave runtime database role has administrative privileges")
        owns_database = await session.scalar(
            text(
                "SELECT d.datdba = r.oid FROM pg_database d "
                "JOIN pg_roles r ON r.rolname = current_user "
                "WHERE d.datname = current_database()"
            )
        )
        if owns_database:
            raise RuntimeError("Strict Weave runtime database role must not own the database")
        result = await session.execute(
            text(
                "SELECT c.relname, c.relowner, r.oid AS runtime_oid, "
                "c.relrowsecurity, c.relforcerowsecurity, "
                "EXISTS (SELECT 1 FROM pg_policies p WHERE p.tablename = c.relname) AS has_policy "
                "FROM pg_class c CROSS JOIN pg_roles r "
                "WHERE r.rolname = current_user AND c.relname = ANY(:tables)"
            ),
            {"tables": list(RLS_TABLES)},
        )
        state = {row.relname: row for row in result}
        invalid = [
            table
            for table in RLS_TABLES
            if table not in state
            or state[table].relowner == state[table].runtime_oid
            or not state[table].relrowsecurity
            or not state[table].relforcerowsecurity
            or not state[table].has_policy
        ]
        if invalid:
            raise RuntimeError(f"Strict Weave RLS is incomplete: {', '.join(invalid)}")
