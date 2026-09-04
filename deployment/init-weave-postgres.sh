#!/usr/bin/env bash
set -euo pipefail

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

REVOKE ALL ON FUNCTION public.weave_create_dataset_schema(text, boolean) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.weave_create_dataset_schema(text, boolean) TO cognee;
SQL
