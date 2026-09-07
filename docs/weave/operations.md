# Cognee Weave operations

Cognee Weave is one private Railway service plus one PostgreSQL service with
the `vector` extension. Neo4j is not deployed. Postgres stores the relational
control plane, graph rows, and vectors. Each organization retains its primary
binding and one native memory dataset/schema shared by its repositories.

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
self-improvement through `improve()`. Source replacement uses `cognee.update()`;
individual source deletion uses `cognee.forget()`. Weave no longer supplies custom extraction tasks, prompts,
graph models, or its own recall ranking. These public operation implementations
and the upstream memory layer are unchanged by this alignment.

The boundary still supplies tenant identity, repository selection, model
credentials and exact-SHA receipts. It adds no operation deadline around native
Cognee calls; upstream timeouts remain unchanged. All repositories resolve to
the customer's bound primary dataset. Recall refuses legacy, failed, running,
missing or stale snapshots and incomplete source receipts. Replacement takes
the existing customer operation lock and uses stable native source IDs. Cognee's
directory resolver produces one code manifest per project plus supported docs;
only changed items are updated. This is not per-symbol incremental indexing.

### Source and completed-review lifecycle (2026-09-06)

Apply migrations through `f2c4e6a8b0d1` before the service starts. Migration
`d9a1c3e5f7b0` adds forced-tenant-RLS source receipts; `e1b3d5f7a9c0` adds review
revision ordering; `f2c4e6a8b0d1` adds repository/customer cleanup-pending flags.
These are integration records, not custom graph ownership.
Reindex every customer repository to `weave-native-memory.v2`. Indexing does not
wipe the customer dataset or siblings. Legacy per-repository datasets are retained
during migration and excluded from recall; explicit repository removal also
forgets that repository's legacy dataset.

Deletion marks cleanup pending before removing native data. Failed cleanup keeps
reads and reactivation blocked until deletion is retried successfully. Customer
export checks every repository's native readiness, even with a repository filter.

`POST /api/v1/weave/organizations/{id}/reviews/remember` accepts final historical
review text, repository ID, review ID, PR head SHA, repository lifecycle generation
and artifact revision. Dataset/user identity is always resolved server-side.
New reviews use native `remember` (including its default `improve`). Corrections
use native `update`, then `improve`; retries do not duplicate completed sources.
An older artifact revision is ignored; conflicting text for the same revision
is rejected. A processing receipt survives a failure and blocks partial recall.
If initial remember fails during native improvement, its retry updates the source
and finishes improvement before completing the receipt. Native exceptions remain
unchanged; no custom pipeline-log interpretation is used.
The request has a content size boundary, but no custom operation timer.

Amberly's separate background workflow loads only completed, assessed reviews
from the database. Finding changes atomically queue a refresh through its outbox.
Memory is historical, untrusted context and never decides governance.

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

## Returning to an earlier default-branch commit

A historical successful index job is not proof that its sources are still current.
When a repository moves A → B → A, the index ledger requeues A and restores its
native sources before promoting the snapshot. Superseded historical jobs can also
be retried. Duplicate delivery of the currently indexed commit remains a no-op.

Release regression verification (2026-09-06): four state-machine cases failed before
the fix and passed afterward. The PostgreSQL archive regression failed on the old
code because B's content receipt remained, then passed with A's original source IDs
and content hashes restored. The maintained PostgreSQL suite passed 15 tests; the
focused Weave unit suite passed 199 tests, and Ruff correctness checks passed.
These local checks do not establish hosted CI or staging deployment acceptance.

The container exposes its build-platform parser through `/app/.cognee/bin/enola`;
it must not force the ARM64 filename on AMD64 builders. The CI PostgreSQL job
bootstraps its fresh database with `cognee-cli upgrade head` before storage tests,
and the Python gate also runs migration/bootstrap regressions.

The native export/restore gate checks repository identity and SHA in `repositories`,
and checks the customer dataset and graph in `native_graph`. It compares the complete
restored export after normalizing ordering and the nested graph JSON. Regression
cases reject foreign organizations/repositories, stale SHAs, missing graphs, changed
dataset IDs, and changed content. Equivalent JSON formatting is accepted.

Parity explicitly writes and searches a 1536-dimensional vector through the runtime
role in its disposable customer dataset, then checks that dataset's HNSW index.
Graph-only native fixtures are not evidence of vector storage. The parity restore
leg enables `RESTORE_EXPECT_VECTOR_CANARY=true` and reads the restored vector without
inserting or repairing it. Ordinary customer restore drills do not require this
synthetic canary. The standalone schema-local query-plan test remains required.

### Automatic staging releases

`.github/workflows/release-staging.yml` follows the Amberly monorepo release
pattern: only a successful **Cognee Weave Gate** run for a push to this
repository's `main` can release. It checks out that run's immutable full SHA;
PR gates and upstream/fork runs cannot release. Concurrent releases queue rather
than cancel one another. Disable Railway's source-triggered automatic deployments
for both Weave services so they cannot bypass this gate.

Configure these values in this repository's GitHub **staging** environment:

- Secret `RAILWAY_PROJECT_TOKEN`: a project token scoped only to staging.
- Variable `RAILWAY_ENVIRONMENT_ID`: the staging environment ID.
- Variable `RAILWAY_MIGRATE_SERVICE_ID`: the `weave-migrate` service ID.
- Variable `RAILWAY_WEAVE_SERVICE_ID`: the private `weave` API service ID.

Both services must retain a connected GitHub source for this repository. The
runner deploys the verified SHA through `serviceInstanceDeployV2`; it does not
deploy the moving branch tip. It verifies the token environment and the target's
`staging` name/project before mutations. The migrator must report `SUCCESS`, the
exact `meta.commitHash`, and `deploymentStopped=true` before API deployment starts.
The API must report `SUCCESS`, the same commit, and remain running. Its configured
Railway `/health` startup check remains mandatory. No public domain is required.

If API deployment fails, the runner attempts to roll back to the prior successful
API deployment and verifies its original commit again. It never rolls back database
migrations; migrations must remain compatible with the previous API. A failed
migration never starts an API deployment. If the existing API's latest deployment
is already unhealthy, automatic release stops for operator inspection.

Every attempt retains a content-free `release-evidence.json` artifact for 90 days,
including prior/new deployment IDs, exact commits, status, and rollback outcome.
Provider metadata plus startup health is not an authenticated connectivity test:
verify the private Weave API from Amberly's staging network after initial wiring
or credential changes. `/health` reports package version, not source commit.
This workflow has no production target and never changes GitHub App suspension.
