#!/usr/bin/env bash
set -euo pipefail

umask 077

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/new/backup.dump" >&2
  exit 2
fi

backup_path="$1"
if [[ "$backup_path" != /* ]] || [ -e "$backup_path" ]; then
  echo "backup path must be absolute and must not already exist" >&2
  exit 2
fi

backup_parent="$(dirname "$backup_path")"
if [ ! -d "$backup_parent" ]; then
  echo "backup parent directory does not exist" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${WEAVE_COMPOSE_FILE:-$root/deployment/docker-compose.weave.yml}"
project="${WEAVE_COMPOSE_PROJECT:-cognee-weave-parity}"
temporary="$(mktemp "${backup_path}.tmp.XXXXXX")"
cleanup() {
  if [ -f "$temporary" ]; then
    rm -f -- "$temporary"
  fi
}
trap cleanup EXIT

docker compose -p "$project" -f "$compose_file" exec -T postgres \
  pg_dump --username=cognee --dbname=cognee_db --format=custom --no-owner --no-acl > "$temporary"

if [ ! -s "$temporary" ]; then
  echo "backup is empty" >&2
  exit 1
fi
chmod 600 "$temporary"
mv "$temporary" "$backup_path"
trap - EXIT
echo "$backup_path"
