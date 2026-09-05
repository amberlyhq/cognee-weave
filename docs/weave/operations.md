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
WEAVE_EMBEDDING_PROVIDER=openrouter
WEAVE_EMBEDDING_MODEL=openrouter/openai/text-embedding-3-small
WEAVE_EMBEDDING_DIMENSIONS=1536
WEAVE_EMBEDDING_ENDPOINT=https://openrouter.ai/api/v1
WEAVE_EMBEDDING_API_KEY=<OpenRouter API key>
```

The leading `openrouter/` is LiteLLM's routing prefix. The requested OpenRouter
model is `openai/text-embedding-3-small`. Compose reads `OPENROUTER_API_KEY`
and supplies it as `WEAVE_EMBEDDING_API_KEY`; direct runs accept either variable.
Repository-derived text is sent to this provider for indexing and query embeddings.
Provider contract: [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings).
Existing 384-dimensional FastEmbed datasets must be rebuilt in fresh storage before
switching traffic. Changing configuration alone does not convert existing vectors.

Use two Postgres users against the same private database:

- The one-shot migration service uses the database-owner credentials supplied
  by Railway. It runs `cognee-cli upgrade head` before the API starts.
- The Weave API uses the dedicated `cognee` runtime user created by the database
  bootstrap. It must be `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE`,
  `NOREPLICATION`, must not bypass row security, and must not own the database.

Give the owner and runtime roles different generated passwords. In the local
Compose contract these are `WEAVE_ADMIN_DB_PASSWORD` and `WEAVE_DB_PASSWORD`;
knowing the API password must never permit logging in as `cognee_admin`.

Set the API's matching `DB_*`, `VECTOR_DB_*`, and `GRAPH_DATABASE_*` username
and password values to that runtime user. All three providers use the same host,
port, and database. Strict mode rejects an owner/admin runtime credential or any
provider mismatch. Do not expose a public Weave domain. Amberly uses the Railway
private hostname and the same `WEAVE_INTERNAL_TOKEN`.

`WEAVE_STRICT_MODE=true` makes container startup fail before migrations if any
tenant, graph, or vector provider setting drifts from this all-Postgres shape.

## Local parity stack

Export `OPENROUTER_API_KEY`, then run `scripts/weave-parity.sh`. This makes paid
embedding requests with synthetic test repositories. CI instead sets
`WEAVE_MOCK_EMBEDDING=true` (Compose) and `MOCK_EMBEDDING=true` (host tests)
with a non-secret placeholder key to exercise storage
and tenant boundaries without credentials. Mock vectors are deterministic and
nonzero so pgvector cosine search is defined. Mocked CI is not provider verification;
never enable this flag for real indexing or recall.
The script creates a fresh Compose project containing
only pgvector Postgres and the Weave API. Indexing runs synchronously in the API
process, so no separate worker is required. The script provisions two test
organizations, indexes exact fixture SHAs, runs the Postgres isolation suite,
checks every API surface, deletes one tenant, and performs a backup/restore
drill in a separate Compose project.
The lifecycle checks prove that ordinary push provisioning cannot revive an
organization tombstone, stale generations are ignored, and only the verified
reactivation endpoints can restore organization or repository state.

The script removes its disposable volumes by default. It never targets an
existing project name. Set `WEAVE_PARITY_REPORT` to preserve its JSON receipt.

## Deploy and verify

### OpenAI embedding verification (2026-09-05)

The maintained fork unit gate plus embedding/tokenizer regressions passed 185
tests; the fork Ruff correctness gate passed. Local provider verification used
OpenRouter with mocking disabled and obtained nonzero 1536-dimensional vectors.
The private `amberlyhq/test-amberly-api` repository at
`172add81fe83bb035b9b6f985adda3de7b6e01a4` indexed into fresh PostgreSQL storage.
The live provider run indexed it successfully in 4.1 seconds.
Symbol, repository, and impact recall returned the indexed SHA with graph and
vector candidates and no diagnostics. All 12 vector columns were `vector(1536)`.
A second organization returned no candidates. This verifies the model change
locally; production rollout and rebuilding older 384-dimensional datasets remain
deployment steps. CI uses mock embeddings and makes no provider-quality claim.
The credential-free `scripts/weave-parity.sh` run passed all 12 PostgreSQL tests,
created 28 HNSW indexes, observed zero cross-organization leaks, and passed
runtime lifecycle and backup/restore checks. Review findings about host mock
configuration and zero-vector cosine distances were reproduced and corrected.

1. Build the reviewed `Dockerfile` with one private pgvector Postgres service.
2. Bootstrap the `cognee` runtime role with a separate password. Run migrations
   once with the database-owner credentials through `cognee-cli upgrade head`.
   Do not run raw Alembic against a fresh database.
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
  source project name, acquires an atomic per-project reservation before
  checking Compose resources, and runs canary export/provenance checks. The
  generated name includes the process identity and cryptographic randomness.
- Repository archives are temporary and deleted after indexing. Derived graph
  and vector data remains rebuildable from GitHub at the recorded SHA.
- Rotate `WEAVE_INTERNAL_TOKEN` in Weave and Amberly together. During mismatch,
  Amberly treats recall as unavailable and keeps its GitHub-first review path.

## Upstream updates

`.github/workflows/upstream-sync.yml` checks upstream weekly and opens an
exact-SHA proposal PR. It never merges. Review conflicts, run `weave-gate`, and
manually accept only after the tenant and Postgres boundaries remain green.
