# Cognee Weave operations

Cognee Weave is one private Railway service plus one PostgreSQL service with
the `vector` extension. Neo4j is not deployed. Postgres stores the relational
control plane, graph rows, vectors, and one dataset-derived schema per Amberly
organization.

## Required runtime settings

```text
WEAVE_STRICT_MODE=true
ENABLE_BACKEND_ACCESS_CONTROL=true
DB_PROVIDER=postgres
VECTOR_DB_PROVIDER=pgvector
VECTOR_DATASET_DATABASE_HANDLER=pgvector_shared
GRAPH_DATABASE_PROVIDER=postgres_demo
GRAPH_DATASET_DATABASE_HANDLER=postgres_graph_shared
WEAVE_INTERNAL_TOKEN=<at least 32 random characters>
WEAVE_EMBEDDING_PROVIDER=fastembed
WEAVE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
WEAVE_EMBEDDING_DIMENSIONS=384
```

Set `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, and `DB_NAME` from the
private Railway Postgres service. Set the matching `VECTOR_DB_*` and
`GRAPH_DATABASE_*` host, port, username, password, and name values to the same
service. Strict mode rejects any mismatch. Do not expose a public Weave domain.
Amberly uses the Railway private hostname and the same `WEAVE_INTERNAL_TOKEN`.

`WEAVE_STRICT_MODE=true` makes container startup fail before migrations if any
tenant, graph, or vector provider setting drifts from this all-Postgres shape.

## Local parity stack

Run `scripts/weave-parity.sh`. It creates a fresh Compose project containing
only pgvector Postgres and the Weave API. Indexing runs synchronously in the API
process, so no separate worker is required. The script provisions two test
organizations, indexes exact fixture SHAs, runs the Postgres isolation suite,
checks every API surface, deletes one tenant, and performs a backup/restore
drill in a separate Compose project.

The script removes its disposable volumes by default. It never targets an
existing project name. Set `WEAVE_PARITY_REPORT` to preserve its JSON receipt.

## Deploy and verify

1. Build the reviewed `Dockerfile` with one private pgvector Postgres service.
2. Set the strict variables above and run migrations through the image
   entrypoint. Do not run raw Alembic against a fresh database.
3. Confirm `/health` inside the private network.
4. Run the same two-tenant operations from `scripts/weave-parity.sh` against
   disposable organization UUIDs.
5. Record the deployed commit, migration head, private hostname, health result,
   tenant smoke receipt, and backup/restore receipt.

CI proof is not deployment proof. A release is accepted only after all of
those running-service values are read back.

## Backups and recovery

- `scripts/weave-backup.sh /absolute/new/backup.dump` creates a mode-0600
  custom-format PostgreSQL backup and refuses to overwrite a file.
- `scripts/weave-restore-drill.sh /absolute/backup.dump` restores only into a
  fresh project whose name starts with `cognee-weave-restore-`. It rejects the
  source project name and runs canary export/provenance checks.
- Repository archives are temporary and deleted after indexing. Derived graph
  and vector data remains rebuildable from GitHub at the recorded SHA.
- Rotate `WEAVE_INTERNAL_TOKEN` in Weave and Amberly together. During mismatch,
  Amberly treats recall as unavailable and keeps its GitHub-first review path.

## Upstream updates

`.github/workflows/upstream-sync.yml` checks upstream weekly and opens an
exact-SHA proposal PR. It never merges. Review conflicts, run `weave-gate`, and
manually accept only after the tenant and Postgres boundaries remain green.
