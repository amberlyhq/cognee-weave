"""add_scoped_weave_schema_deletion

Revision ID: a5c7e9b1d3f6
Revises: f3a5c7e9b1d4
Create Date: 2026-09-04 19:45:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a5c7e9b1d3f6"
down_revision: Union[str, None] = "f3a5c7e9b1d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(
        """
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
    )
    op.execute(
        "REVOKE ALL ON FUNCTION "
        "public.weave_drop_organization_dataset_schema(uuid) FROM PUBLIC"
    )
    op.execute(
        """
        DO $grant$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
                GRANT EXECUTE ON FUNCTION public.weave_drop_organization_dataset_schema(uuid)
                    TO cognee;
            END IF;
        END
        $grant$
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute("DROP FUNCTION IF EXISTS public.weave_drop_organization_dataset_schema(uuid)")
