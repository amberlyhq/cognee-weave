#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/backup.dump" >&2
  exit 2
fi

backup_path="$1"
if [[ "$backup_path" != /* ]] || [ ! -s "$backup_path" ]; then
  echo "backup path must be an absolute non-empty file" >&2
  exit 2
fi

: "${RESTORE_ORGANIZATION_ID:?RESTORE_ORGANIZATION_ID is required}"
: "${RESTORE_REPOSITORY_ID:?RESTORE_REPOSITORY_ID is required}"
: "${RESTORE_EXPECTED_EXPORT:?RESTORE_EXPECTED_EXPORT is required}"
if [[ "$RESTORE_EXPECTED_EXPORT" != /* ]] || [ ! -s "$RESTORE_EXPECTED_EXPORT" ]; then
  echo "RESTORE_EXPECTED_EXPORT must be an absolute non-empty file" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${WEAVE_COMPOSE_FILE:-$root/deployment/docker-compose.weave.yml}"
source_project="${WEAVE_COMPOSE_PROJECT:-cognee-weave-parity}"
restore_project="${RESTORE_COMPOSE_PROJECT:-cognee-weave-restore-$(date +%s)}"
if [[ ! "$restore_project" =~ ^cognee-weave-restore-[a-zA-Z0-9._-]+$ ]] || [ "$restore_project" = "$source_project" ]; then
  echo "restore project must be a distinct cognee-weave-restore-* name" >&2
  exit 2
fi

export WEAVE_ADMIN_DB_PASSWORD="${RESTORE_ADMIN_DB_PASSWORD:-$(openssl rand -hex 24)}"
export WEAVE_DB_PASSWORD="${RESTORE_DB_PASSWORD:-$(openssl rand -hex 24)}"
export WEAVE_INTERNAL_TOKEN="${RESTORE_INTERNAL_TOKEN:-$(openssl rand -hex 32)}"
export WEAVE_POSTGRES_PORT="${RESTORE_POSTGRES_PORT:-15433}"
export WEAVE_HTTP_PORT="${RESTORE_HTTP_PORT:-18001}"

for resource in container volume network; do
  if [ -n "$(docker "${resource}" ls -q --filter label=com.docker.compose.project="$restore_project")" ]; then
    echo "restore project already owns Docker resources: ${restore_project}" >&2
    exit 2
  fi
done

cleanup() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    docker compose -p "$restore_project" -f "$compose_file" logs --no-color --tail=200 >&2 || true
  fi
  if [ "${KEEP_WEAVE_RESTORE:-false}" != "true" ]; then
    docker compose -p "$restore_project" -f "$compose_file" down --volumes --remove-orphans >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

docker compose -p "$restore_project" -f "$compose_file" up -d postgres
for _ in $(seq 1 60); do
  if docker compose -p "$restore_project" -f "$compose_file" exec -T postgres \
    pg_isready --username=cognee_admin --dbname=cognee_db >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
docker compose -p "$restore_project" -f "$compose_file" exec -T postgres \
  pg_isready --username=cognee_admin --dbname=cognee_db >/dev/null
docker compose -p "$restore_project" -f "$compose_file" exec -T postgres \
  pg_restore --username=cognee_admin --dbname=cognee_db --clean --if-exists --exit-on-error < "$backup_path"
docker compose -p "$restore_project" -f "$compose_file" up -d weave

base_url="http://127.0.0.1:${WEAVE_HTTP_PORT}"
for _ in $(seq 1 90); do
  if curl -fsS "$base_url/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "$base_url/health" >/dev/null

export_response="$(curl -fsS \
  -H "Authorization: Bearer ${WEAVE_INTERNAL_TOKEN}" \
  "$base_url/api/v1/weave/organizations/${RESTORE_ORGANIZATION_ID}/export?repository_id=${RESTORE_REPOSITORY_ID}")"
printf '%s' "$export_response" | python -c '
import json, os, pathlib, sys
value = json.load(sys.stdin)
assert value["organization_id"] == os.environ["RESTORE_ORGANIZATION_ID"]
repository_id = int(os.environ["RESTORE_REPOSITORY_ID"])
repositories = [item for item in value["repositories"] if item["github_repository_id"] == repository_id]
assert repositories and repositories[0]["indexed_default_sha"]
assert value["nodes"]
assert value["edges"]
assert all(item["github_repository_id"] == repository_id for item in value["nodes"])
assert all(item["indexed_sha"] for item in value["nodes"])

expected = json.loads(pathlib.Path(os.environ["RESTORE_EXPECTED_EXPORT"]).read_text())
def normalized(item):
    if isinstance(item, dict):
        return {key: normalized(child) for key, child in item.items() if key != "age_seconds"}
    if isinstance(item, list):
        children = [normalized(child) for child in item]
        return sorted(children, key=lambda child: json.dumps(child, sort_keys=True))
    return item
if normalized(value) != normalized(expected):
    raise SystemExit("restored export does not match the backup source")
'

echo "restore drill passed for ${RESTORE_ORGANIZATION_ID}/${RESTORE_REPOSITORY_ID} in ${restore_project}"
