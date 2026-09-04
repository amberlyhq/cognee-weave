#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

patterns='(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{82,255}|sk-[A-Za-z0-9]{32,255}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)'
if git grep -I -n -E "$patterns" -- . \
  ':(exclude)cognee/tests/**' \
  ':(exclude)docs/**' \
  ':(exclude)scripts/weave-secret-scan.sh'; then
  echo "high-confidence credential pattern found in tracked runtime source" >&2
  exit 1
fi
