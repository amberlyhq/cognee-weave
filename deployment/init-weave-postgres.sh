#!/usr/bin/env bash
set -euo pipefail

if [ "${WEAVE_DB_PASSWORD}" = "${POSTGRES_PASSWORD}" ]; then
  echo "runtime and admin database passwords must differ" >&2
  exit 2
fi

psql --set ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  --set runtime_password="${WEAVE_DB_PASSWORD}" <<'SQL'
SELECT format('CREATE ROLE cognee LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION', :'runtime_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cognee')\gexec
CREATE EXTENSION IF NOT EXISTS vector;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE cognee_db TO cognee;
GRANT USAGE ON SCHEMA public TO cognee;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cognee;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO cognee;

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
  EXECUTE format('GRANT USAGE, CREATE ON SCHEMA %I TO cognee', schema_name);
END;
$function$;

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
$function$;

REVOKE ALL ON FUNCTION public.weave_create_dataset_schema(text, boolean) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.weave_drop_organization_dataset_schema(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.weave_create_dataset_schema(text, boolean) TO cognee;
GRANT EXECUTE ON FUNCTION public.weave_drop_organization_dataset_schema(uuid) TO cognee;
SQL
