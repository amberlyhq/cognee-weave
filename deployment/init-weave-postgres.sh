#!/usr/bin/env bash
set -euo pipefail

psql --set ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  --set runtime_password="${WEAVE_DB_PASSWORD}" <<'SQL'
SELECT format('CREATE ROLE cognee LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION', :'runtime_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cognee')\gexec
CREATE EXTENSION IF NOT EXISTS vector;
ALTER DATABASE cognee_db OWNER TO cognee;
GRANT ALL ON SCHEMA public TO cognee;
SQL
