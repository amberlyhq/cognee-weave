# Cognee Weave operations

Cognee Weave is one private Railway service plus one PostgreSQL service with
the `vector` extension. Neo4j is not deployed. Postgres stores the relational
control plane, graph rows, and vectors. Each organization retains its primary
control-plane binding; each repository has its own native memory dataset/schema.

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
LLM_API_KEY=<OpenRouter API key>
LLM_PROVIDER=openai
LLM_MODEL=openrouter/openai/gpt-oss-120b
LLM_ENDPOINT=https://openrouter.ai/api/v1
LLM_ARGS={"extra_body":{"provider":{"zdr":true}}}
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
Offline recall tests stub the paid Cognee public operation and check dataset/user
routing. The HTTP latency gate measures foreign-repository rejection, not native
answer generation. Native answer quality and latency need a separate paid test.
The host admin-role storage leg always uses mock embeddings, including when
the strict runtime container is making approved live provider calls.
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

### Native memory alignment (2026-09-05)

`modules/weave/native_memory.py` delegates to the pinned upstream public
`cognee.remember()` and `cognee.recall()` defaults. `remember()` retains native
self-improvement through `improve()`. Repository replacement and deletion use
`cognee.forget()`. Weave no longer supplies custom extraction tasks, prompts,
graph models, or its own recall ranking. These public operation implementations
and the upstream memory layer are unchanged by this alignment.

The boundary still supplies tenant identity, repository selection, model
credentials, deadlines, and exact-SHA receipts. A repository resolves only by
organization, owner, tenant, and the deterministic native dataset name. Recall
refuses legacy, failed, running, missing, or stale snapshots. Replacement takes
the existing operation lock, forgets the previous repository dataset, and runs
a full native rebuild. A failed replacement is unavailable, not an old index
labelled current. This is not incremental per-file indexing.

Migration `c7e9f1a3b5d8` permits native deletion of admin-owned dataset schemas
only after checking the organization scope and its bound native dataset. It
does not grant the runtime user database ownership. Apply it before running
the new service. Older structural receipts must be reindexed; their contents
are not silently treated as native memory.

Recall returns `native_memory`, the serialized native response, separately from
legacy symbol candidates. Amberly carries this as untrusted `nativeMemory`;
it is not source-verified evidence and cannot decide governance. Export and
visualization return `native_graph` with unmodified nodes/relationships and
repository/SHA envelopes, capped at 500 nodes, 1,000 edges, and 1 MB of text.
These are bounded inspection surfaces, not full database backups or a new UI.
Amberly rejects a source-verified label on native memory; if old and native
representations are mixed, its verifier keeps only the unverified native text.

Live local comparison on `test-amberly-api` at
`172add81fe83bb035b9b6f985adda3de7b6e01a4`:

| Measurement | Earlier structural pilot | Native remember pilot |
| --- | ---: | ---: |
| Index time | 4.72 s, including queue | 31.93 s, operation wall time |
| Graph nodes | 11 | 22 |
| Relationships | 17 | 28 |
| Vector rows | 11 | 22 |

The native run covered three code/project files and one README; the hidden
dependency file was skipped by upstream ingestion. New nodes included four
entities, four entity types, one summary, one document, and one chunk. It added
a `consumes` relationship to the shared contract. Two native recall queries
took 16.28 s and 16.53 s. The shared-contract answer was correct; the function
return-value question was not answered. Native code ingestion remains Enola's
structural parsing; documentation uses LLM extraction and summarization. We
did not force source code through a custom prose pipeline.

A second live run through the Weave wrapper indexed in 43.17 s with 24 nodes
and 31 edges; stochastic extraction explains count variation. Duplicate jobs
made no new provider calls. Foreign recall, foreign schema deletion, native
forget, recall-after-delete, and stale-job resurrection checks passed against
disposable organizations. The original test index was retained.

Models were OSS-120B internally and OpenAI `text-embedding-3-small` at 1,536
dimensions. The direct pilot inspected 19 outgoing provider requests and
required `provider.zdr=true` on every request. Chat uses `extra_body`; the
pinned LiteLLM embedding path needs top-level `provider` to put this on the
wire. This proves request routing, not an independent audit of provider
retention. See [OpenRouter ZDR](https://openrouter.ai/docs/guides/features/zdr).
Weave explicitly clears inherited per-stage and fallback routing settings so
they cannot override the approved internal model or endpoint.

The final local focused suite passed 187 Cognee unit tests. Four real-Postgres
boundary tests passed, and the new migration passed downgrade/re-upgrade with
function readback. Amberly passed 238 governance tests (eight database-specific
tests skipped), typecheck, and changed-file lint/format checks. Review caught
stage-model overrides, the non-strict parity embedding path, and a native-memory
verification-label edge case; regression tests reproduced them before fixes.

Verification is local source execution against the real Postgres adapters and
live provider, not a rebuilt-image, production, or Luna-agent evaluation claim.
The earlier image/parity results below predate this change. Re-run image,
migration/restore, and deployment acceptance before release. A small-repository
pilot cannot establish agent accuracy or token savings.

Upstream contracts: [remember](https://docs.cognee.ai/core-concepts/main-operations/remember),
[recall](https://docs.cognee.ai/core-concepts/main-operations/recall),
[improve](https://docs.cognee.ai/core-concepts/main-operations/improve),
[forget](https://docs.cognee.ai/core-concepts/main-operations/forget).

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
