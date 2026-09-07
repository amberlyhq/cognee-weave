"""Permit native forget of a tenant-owned repository dataset, not arbitrary DDL."""

from alembic import op

revision = "c7e9f1a3b5d8"
down_revision = "b6d8f0a2c4e7"
branch_labels = None
depends_on = None

DROP_NATIVE_SCHEMA_FUNCTION = """
CREATE OR REPLACE FUNCTION public.weave_drop_native_dataset_schema(
    target_organization_id uuid, target_dataset_id uuid
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF NULLIF(current_setting('app.weave_organization_id', true), '')::uuid
        IS DISTINCT FROM target_organization_id THEN
        RAISE EXCEPTION 'Weave organization scope mismatch';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM public.weave_organization_bindings b
        JOIN public.datasets d ON d.owner_id = b.service_user_id AND d.tenant_id = b.tenant_id
        WHERE b.organization_id = target_organization_id AND b.deleted_at IS NULL
        AND d.id = target_dataset_id
        AND d.name ~ ('^weave-memory-' || replace(target_organization_id::text, '-', '') || '-[0-9]+$')
    ) THEN
        RAISE EXCEPTION 'Bound native Weave dataset not found';
    END IF;
    EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE',
        'ds_' || replace(target_dataset_id::text, '-', ''));
END;
$function$
"""

NATIVE_SCHEMA_GRANT = """
DO $grant$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'cognee') THEN
        GRANT EXECUTE ON FUNCTION public.weave_drop_native_dataset_schema(uuid, uuid) TO cognee;
    END IF;
END
$grant$
"""


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(DROP_NATIVE_SCHEMA_FUNCTION)
        op.execute(
            "REVOKE ALL ON FUNCTION public.weave_drop_native_dataset_schema(uuid, uuid) FROM PUBLIC"
        )
        op.execute(NATIVE_SCHEMA_GRANT)


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS public.weave_drop_native_dataset_schema(uuid, uuid)")
