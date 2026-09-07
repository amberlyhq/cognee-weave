# Cognee Weave parity gate

The previous Amberly Weave implementation remains the reference baseline until
this fork proves the following replacement behaviors:

The legacy baseline (`/Users/siri/Developer/amberly-weave/docs/verified-local-baseline.md`)
recorded 282 Python tests, 64 web tests, and 259 two-project API smoke checks
with zero leaks. It did not prove organization-to-GitHub identity, exact-SHA
provenance, project deletion, HNSW plans, backups, restores, or production
operation. Cognee Weave must keep its zero-leak floor while adding those proofs.

| Behavior | Cognee Weave proof |
| --- | --- |
| Organization isolation | `test_shared_schema_isolation.py`, `test_shared_schema_concurrency.py` |
| Graph retrieval isolation | `test_tenant_graph_retrieval.py` |
| Vector ANN indexes and plan | `test_pgvector_hnsw_plan.py` |
| Exact-SHA code indexing and provenance | `test_weave_exact_sha_indexing.py` |
| Hybrid graph/vector recall | `test_weave_hybrid_recall.py` |
| Export, visualization, and deletion isolation | `test_weave_surface_isolation.py` |
| Ordered organization/repository lifecycle | `test_weave_surface_isolation.py`, `weave-parity.sh` |
| Private API authentication and bounded contracts | `test_internal_auth.py`, `test_recall_contract.py` |
| Safe archive extraction and retry state | `test_archive_validation.py`, `test_index_state_machine.py` |
| Backup and isolated restore | `weave-backup.sh`, `weave-restore-drill.sh` |

## Initial acceptance thresholds

- zero cross-organization candidates, nodes, edges, cache entries, or surviving
  schemas after tenant deletion;
- 100% of returned code candidates carry the requested repository identity and
  exact indexed SHA;
- every tenant collection has a schema-local HNSW cosine index and the bounded
  plan test reads that index;
- no duplicate extraction for the same organization, repository, SHA,
  pipeline version, and extraction version;
- p95 recall below 5 seconds for the tiny local parity fixtures;
- each tiny fixture indexes within 5 minutes and adds no more than 250 MiB of
  total PostgreSQL size;
- restored canary export returns the same repository identity, indexed SHA,
  nodes, and edges from an isolated fresh Compose project.
- older lifecycle generations cannot reverse a newer removal, and normal
  provisioning cannot recreate a deleted organization.

These are safety and operability floors, not claims that Cognee Weave is faster
or better than the previous implementation. Record p50/p90/p95 latency,
database size growth, index catalog readback, query-plan proof, zero-leak
counts, and restore results in the parity receipt before comparison.

Amberly verifies selected candidate paths through GitHub at the exact review
SHA. Weave never owns a Risk, Confidence, Attention, or Action decision.

Production parity is not claimed by local or CI checks. It requires a private
Railway deployment, exact-revision readback, migrations, health, backup and
restore proof, and a live two-tenant isolation smoke. Do not remove or cut over
from the previous Weave without separate approval.

`WEAVE_PARITY_SKIP_BUILD=true` is only for a local rerun when
`cognee-weave-parity:local` was already built from the same checkout. CI and a
first verification run must build the image.
