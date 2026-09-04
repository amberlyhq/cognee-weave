"""secure_weave_runtime_role

Revision ID: f3a5c7e9b1d4
Revises: e2f4a6b8c0d3
Create Date: 2026-09-04 16:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f3a5c7e9b1d4"
down_revision: Union[str, None] = "e2f4a6b8c0d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ORGANIZATION_PREDICATE = (
    "organization_id = NULLIF(current_setting('app.weave_organization_id', true), '')::uuid"
)
_CONTROL_TABLES = (
    "weave_organization_bindings",
    "weave_repository_snapshots",
    "weave_index_jobs",
)


def _install_schema_functions() -> None:
    op.execute(
        """
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
                'CREATE SCHEMA IF NOT EXISTS %I AUTHORIZATION %I',
                schema_name,
                current_user
            );
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
                EXECUTE format('GRANT USAGE, CREATE ON SCHEMA %I TO cognee', schema_name);
            END IF;
        END;
        $function$
        """
    )
    op.execute(
        "REVOKE ALL ON FUNCTION public.weave_create_dataset_schema(text, boolean) FROM PUBLIC"
    )
    op.execute(
        """
        DO $grant$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
                GRANT EXECUTE ON FUNCTION public.weave_create_dataset_schema(text, boolean)
                    TO cognee;
                GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public
                    TO cognee;
                GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cognee;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cognee;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    GRANT USAGE, SELECT ON SEQUENCES TO cognee;
            END IF;
        END
        $grant$
        """
    )


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    with op.get_context().autocommit_block():
        op.execute(
            """
            DO $owner$
            BEGIN
                EXECUTE format(
                    'ALTER DATABASE %I OWNER TO %I',
                    current_database(),
                    current_user
                );
            END
            $owner$
            """
        )
    for table_name in _CONTROL_TABLES:
        op.execute(f'ALTER TABLE "{table_name}" OWNER TO CURRENT_USER')

    table_name = "weave_organization_bindings"
    policy_name = f"{table_name}_organization_isolation"
    op.execute(f'ALTER TABLE "{table_name}" ENABLE ROW LEVEL SECURITY')
    op.execute(f'ALTER TABLE "{table_name}" FORCE ROW LEVEL SECURITY')
    op.execute(f'DROP POLICY IF EXISTS "{policy_name}" ON "{table_name}"')
    op.execute(
        f'CREATE POLICY "{policy_name}" ON "{table_name}" '
        f"USING ({_ORGANIZATION_PREDICATE}) WITH CHECK ({_ORGANIZATION_PREDICATE})"
    )
    _install_schema_functions()


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute("DROP FUNCTION IF EXISTS public.weave_create_dataset_schema(text, boolean)")
    op.execute(
        'DROP POLICY IF EXISTS "weave_organization_bindings_organization_isolation" '
        'ON "weave_organization_bindings"'
    )
    op.execute('ALTER TABLE "weave_organization_bindings" NO FORCE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE "weave_organization_bindings" DISABLE ROW LEVEL SECURITY')
