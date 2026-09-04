#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$root/deployment/docker-compose.weave.yml"
project="cognee-weave-parity-${RANDOM}-$$"
temporary="$(mktemp -d)"
report="${WEAVE_PARITY_REPORT:-/tmp/cognee-weave-parity-report.json}"

export WEAVE_COMPOSE_FILE="$compose_file"
export WEAVE_COMPOSE_PROJECT="$project"
export WEAVE_ADMIN_DB_PASSWORD="${WEAVE_ADMIN_DB_PASSWORD:-$(openssl rand -hex 24)}"
export WEAVE_DB_PASSWORD="${WEAVE_DB_PASSWORD:-$(openssl rand -hex 24)}"
export WEAVE_INTERNAL_TOKEN="${WEAVE_INTERNAL_TOKEN:-$(openssl rand -hex 32)}"
export WEAVE_POSTGRES_PORT="${WEAVE_POSTGRES_PORT:-15432}"
export WEAVE_HTTP_PORT="${WEAVE_HTTP_PORT:-18000}"

cleanup() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    docker compose -p "$project" -f "$compose_file" logs --no-color --tail=200 >&2 || true
  fi
  docker compose -p "$project" -f "$compose_file" down --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf -- "$temporary"
  exit "$status"
}
trap cleanup EXIT

for command in docker curl git openssl python; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 2; }
done

if [ "${WEAVE_PARITY_SKIP_BUILD:-false}" = "true" ]; then
  docker compose -p "$project" -f "$compose_file" up -d --no-build
else
  docker compose -p "$project" -f "$compose_file" up -d --build
fi
base_url="http://127.0.0.1:${WEAVE_HTTP_PORT}"
for _ in $(seq 1 120); do
  if curl -fsS "$base_url/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "$base_url/health" >/dev/null
database_before_bytes="$(docker compose -p "$project" -f "$compose_file" exec -T postgres \
  psql -At --username=cognee --dbname=cognee_db -c "SELECT pg_database_size('cognee_db')")"

organization_a="7e1a7b9d-08c2-4f57-9884-623e01b68b01"
organization_b="7e1a7b9d-08c2-4f57-9884-623e01b68b02"
repository_a=920001
repository_b=920002

create_fixture() {
  local directory="$1"
  local symbol="$2"
  mkdir -p "$directory"
  git -C "$directory" init -q
  git -C "$directory" config user.name "Cognee Weave Parity"
  git -C "$directory" config user.email "weave-parity@internal.amberly"
  printf 'module github.com/amberlyhq/%s\n\ngo 1.24\n' "$(basename "$directory")" > "$directory/go.mod"
  printf 'package main\n\nfunc %s() string { return "%s" }\n\nfunc main() { println(%s()) }\n' "$symbol" "$symbol" "$symbol" > "$directory/main.go"
  git -C "$directory" add go.mod main.go
  GIT_AUTHOR_DATE="2026-01-01T00:00:00Z" GIT_COMMITTER_DATE="2026-01-01T00:00:00Z" \
    git -C "$directory" commit -q -m "fixture $symbol"
  git -C "$directory" archive --format=tar.gz -o "$directory.tar.gz" HEAD
  git -C "$directory" rev-parse HEAD
}

sha_a="$(create_fixture "$temporary/alpha" "AlphaCanary")"
sha_b="$(create_fixture "$temporary/beta" "BetaCanary")"

for organization in "$organization_a" "$organization_b"; do
  curl -fsS -X POST -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
    "$base_url/api/v1/weave/organizations/${organization}/provision" >/dev/null
done

index_fixture() {
  local organization="$1" repository_id="$2" name="$3" sha="$4" archive="$5"
  curl -fsS -X POST -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
    -F "archive=@${archive};type=application/gzip" \
    -F "github_repository_id=${repository_id}" \
    -F "repository_owner=amberlyhq" \
    -F "repository_name=${name}" \
    -F "default_branch=main" \
    -F "requested_sha=${sha}" \
    -F "pipeline_version=weave-code.v1" \
    -F "extraction_version=enola-0.3.13" \
    "$base_url/api/v1/weave/organizations/${organization}/repositories/index" >/dev/null
}

start_seconds="$(date +%s)"
index_fixture "$organization_a" "$repository_a" "weave-alpha" "$sha_a" "$temporary/alpha.tar.gz"
fixture_index_seconds_a="$(( $(date +%s) - start_seconds ))"
if [ "$fixture_index_seconds_a" -gt 300 ]; then
  echo "alpha fixture indexing exceeded the 5-minute gate" >&2
  exit 1
fi

start_seconds="$(date +%s)"
index_fixture "$organization_b" "$repository_b" "weave-beta" "$sha_b" "$temporary/beta.tar.gz"
fixture_index_seconds_b="$(( $(date +%s) - start_seconds ))"
if [ "$fixture_index_seconds_b" -gt 300 ]; then
  echo "beta fixture indexing exceeded the 5-minute gate" >&2
  exit 1
fi

curl -fsS -X DELETE -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}/repositories/${repository_a}" >/dev/null
removed_index_status="$(curl -sS -o "$temporary/removed-index.json" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  -F "archive=@${temporary}/alpha.tar.gz;type=application/gzip" \
  -F "github_repository_id=${repository_a}" \
  -F "repository_owner=amberlyhq" \
  -F "repository_name=weave-alpha" \
  -F "default_branch=main" \
  -F "requested_sha=${sha_a}" \
  -F "pipeline_version=weave-code.v1" \
  -F "extraction_version=enola-0.3.13" \
  "$base_url/api/v1/weave/organizations/${organization_a}/repositories/index")"
if [ "$removed_index_status" != "409" ]; then
  echo "removed repository indexing did not return 409" >&2
  exit 1
fi
curl -fsS -X POST -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}/repositories/${repository_a}/activate" >/dev/null
index_fixture "$organization_a" "$repository_a" "weave-alpha" "$sha_a" "$temporary/alpha.tar.gz"
export WEAVE_STRICT_MODE=false
export DB_PROVIDER=postgres DB_HOST=127.0.0.1 DB_PORT="$WEAVE_POSTGRES_PORT"
export DB_USERNAME=cognee_admin DB_PASSWORD="$WEAVE_ADMIN_DB_PASSWORD" DB_NAME=cognee_db
export ENABLE_BACKEND_ACCESS_CONTROL=true VECTOR_DB_PROVIDER=pgvector
export VECTOR_DATASET_DATABASE_HANDLER=pgvector_shared GRAPH_DATABASE_PROVIDER=postgres_demo
export GRAPH_DATASET_DATABASE_HANDLER=postgres_graph_shared COGNEE_SKIP_CONNECTION_TEST=true
export VECTOR_DB_HOST=127.0.0.1 VECTOR_DB_PORT="$WEAVE_POSTGRES_PORT"
export VECTOR_DB_USERNAME=cognee_admin VECTOR_DB_PASSWORD="$WEAVE_ADMIN_DB_PASSWORD" VECTOR_DB_NAME=cognee_db
export GRAPH_DATABASE_HOST=127.0.0.1 GRAPH_DATABASE_PORT="$WEAVE_POSTGRES_PORT"
export GRAPH_DATABASE_USERNAME=cognee_admin GRAPH_DATABASE_PASSWORD="$WEAVE_ADMIN_DB_PASSWORD"
export GRAPH_DATABASE_NAME=cognee_db

if [ -x "$root/.venv/bin/pytest" ]; then
  "$root/.venv/bin/pytest" \
    "$root/cognee/tests/e2e/postgres/test_shared_schema_isolation.py" \
    "$root/cognee/tests/e2e/postgres/test_shared_schema_concurrency.py" \
    "$root/cognee/tests/e2e/postgres/test_tenant_graph_retrieval.py" \
    "$root/cognee/tests/e2e/postgres/test_pgvector_hnsw_plan.py" \
    "$root/cognee/tests/e2e/postgres/test_weave_exact_sha_indexing.py" \
    "$root/cognee/tests/e2e/postgres/test_weave_hybrid_recall.py" \
    "$root/cognee/tests/e2e/postgres/test_weave_surface_isolation.py" \
    --timeout=300 --timeout-method=thread -q
else
  echo "run uv sync --locked --dev --extra postgres before the parity script" >&2
  exit 2
fi

recall_times="$temporary/recall-times.txt"
for _ in $(seq 1 5); do
  curl -fsS -o "$temporary/recall.json" -w '%{time_total}\n' \
    -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" -H "Content-Type: application/json" \
    -d "{\"mode\":\"repository_context\",\"query\":\"BetaCanary\",\"github_repository_ids\":[${repository_b}],\"seeds\":[\"${sha_b}\"],\"top_k\":10,\"depth\":1,\"deadline_ms\":5000}" \
    "$base_url/api/v1/weave/organizations/${organization_b}/recall" >> "$recall_times"
done
ORGANIZATION_ID="$organization_b" REPOSITORY_ID="$repository_b" SHA="$sha_b" \
  python - "$temporary/recall.json" <<'PYTHON'
import json, os, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["organization_id"] == os.environ["ORGANIZATION_ID"]
repository_id = int(os.environ["REPOSITORY_ID"])
candidates = value["graph_candidates"] + value["vector_candidates"]
assert candidates
assert all(item["github_repository_id"] == repository_id for item in candidates)
assert all(item["indexed_sha"] == os.environ["SHA"] for item in candidates)
PYTHON

if curl -fsS -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}/export?repository_id=${repository_b}" >/dev/null 2>&1; then
  echo "foreign repository export unexpectedly succeeded" >&2
  exit 1
fi

if docker compose -p "$project" -f "$compose_file" exec -T postgres \
  psql --set ON_ERROR_STOP=1 --username=cognee --dbname=cognee_db \
  -c "SELECT set_config('app.weave_organization_id', '${organization_a}', false); SELECT public.weave_drop_organization_dataset_schema('${organization_b}'::uuid)" \
  >/dev/null 2>&1; then
  echo "cross-organization schema deletion unexpectedly succeeded" >&2
  exit 1
fi

curl -fsS -X DELETE -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}" >/dev/null
curl -fsS -X POST -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}/provision" >/dev/null
curl -fsS -X DELETE -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_a}" >/dev/null

export_response="$(curl -fsS -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${organization_b}/export?repository_id=${repository_b}")"
printf '%s' "$export_response" > "$temporary/expected-export.json"
printf '%s' "$export_response" | ORGANIZATION_ID="$organization_b" REPOSITORY_ID="$repository_b" SHA="$sha_b" python -c '
import json, os, sys
value = json.load(sys.stdin)
assert value["organization_id"] == os.environ["ORGANIZATION_ID"]
repository_id = int(os.environ["REPOSITORY_ID"])
assert value["nodes"]
assert all(item["github_repository_id"] == repository_id for item in value["nodes"])
assert all(item["indexed_sha"] == os.environ["SHA"] for item in value["nodes"])
'

database_bytes="$(docker compose -p "$project" -f "$compose_file" exec -T postgres \
  psql -At --username=cognee --dbname=cognee_db -c "SELECT pg_database_size('cognee_db')")"
database_growth_bytes="$(( database_bytes - database_before_bytes ))"
if [ "$database_growth_bytes" -gt 262144000 ]; then
  echo "tiny fixtures exceeded the 250 MiB database-growth gate" >&2
  exit 1
fi
index_count="$(docker compose -p "$project" -f "$compose_file" exec -T postgres \
  psql -At --username=cognee --dbname=cognee_db -c "SELECT count(*) FROM pg_indexes WHERE indexdef ILIKE '%using hnsw%'")"
if [ "$index_count" -lt 1 ]; then
  echo "no HNSW index found" >&2
  exit 1
fi

backup="$temporary/weave.dump"
"$root/scripts/weave-backup.sh" "$backup" >/dev/null
RESTORE_ORGANIZATION_ID="$organization_b" RESTORE_REPOSITORY_ID="$repository_b" \
  RESTORE_EXPECTED_EXPORT="$temporary/expected-export.json" \
  RESTORE_HTTP_PORT=18001 RESTORE_POSTGRES_PORT=15433 \
  "$root/scripts/weave-restore-drill.sh" "$backup" >/dev/null

python - "$recall_times" "$report" "$fixture_index_seconds_a" "$fixture_index_seconds_b" "$database_bytes" "$database_growth_bytes" "$index_count" <<'PYTHON'
import json, math, pathlib, statistics, sys
times = sorted(float(item) for item in pathlib.Path(sys.argv[1]).read_text().splitlines())
receipt = {
    "schema_version": "cognee-weave-parity.v1",
    "recall_seconds": {
        "p50": statistics.median(times),
        "p90": times[max(0, math.ceil(len(times) * 0.9) - 1)],
        "p95": times[max(0, math.ceil(len(times) * 0.95) - 1)],
    },
    "fixture_index_seconds": {
        "alpha": int(sys.argv[3]),
        "beta": int(sys.argv[4]),
        "total": int(sys.argv[3]) + int(sys.argv[4]),
    },
    "database_bytes": int(sys.argv[5]),
    "database_growth_bytes": int(sys.argv[6]),
    "hnsw_index_count": int(sys.argv[7]),
    "query_plan": "passed by test_pgvector_hnsw_plan.py",
    "cross_organization_leaks": 0,
    "backup_restore": "passed",
    "runtime_lifecycle": "passed",
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2) + "\n")
if receipt["recall_seconds"]["p95"] >= 5:
    raise SystemExit("p95 recall exceeded 5 seconds")
PYTHON

echo "$report"
