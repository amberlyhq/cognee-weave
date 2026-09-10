# Code graph first, review memory later

Repository imports call native `cognee.remember(content_type="code", index_vectors=False,
self_improvement=False)`. Enola builds code facts and relationships. They do not run
whole-repository text/image enrichment, embeddings, or `improve()`.

Code lives in the stable, tenant-owned `weave-memory-{organization}-{repository}-code-v1`
dataset. Native snapshot/delta handling replaces changed code and removes stale code
facts without deleting review lessons. Old repository memory datasets are excluded from
new retrieval and removed by repository/organization deletion.

Completed reviews carry the latest completed invocation for each reviewer role. Amberly
exports its structured result and saved activity, excluding failed/unfinished invocations.
Exports redact credentials, merge streamed text, and cap each session at 64,000 characters
and 200 source steps. Truncation is explicit. This is bounded activity, not an unlimited
session dump or a claim that every trace byte was retained.

Weave calls native `remember(QAEntry)` and `remember(TraceEntry)`, followed by native
session extraction and `cognee.session.distill_session`. Cognee's confidence gate,
curator, duplicate checks, writer, prompts, and model adapter remain in use. Only accepted
lessons enter permanent memory. General `improve()` is not called for this path because
it also enriches the wider graph. Final review reports/dispositions remain separate,
revisioned historical memory, with self-improvement disabled.

Each invocation has a durable receipt and a stable native session ID, scoped by the
trusted organization and repository. Retries read existing entries and Cognee extraction
watermarks. Failed extraction stops before a pending window can overflow. A small,
context-local strict error option propagates native curator/writer/publication/duplicate-check failures
instead of treating them as "nothing useful to learn"; default Cognee callers retain
best-effort behavior. A failed publication may retry its unfinished native work. No claim
of exactly-once model billing is made across a crash between publication and receipt commit.

The native Postgres cache persists sessions across service restarts. The admin migration
creates its tables; the runtime gets data access, not CREATE permission on public.
Session expiry is disabled for durable retry; repository/organization deletion also
removes the associated session cache. Model IDs and provider routing are unchanged.

Code recall uses native deterministic CODE queries. Completed historical review memory
also opts into native semantic recall. A pending review receipt does not disable code
navigation. A failed semantic call returns the code results with a diagnostic. Code facts and lessons share repository scope; native entity matching is not
proof that a lesson is linked to a particular verified source symbol or current commit.
Review provenance explicitly labels historical PR head and reviewer identity.

## Verification

- `cognee/tests/unit/modules/weave/test_review_sessions.py`
- `cognee/tests/e2e/postgres/test_weave_exact_sha_indexing.py`
- `cognee/tests/e2e/postgres/test_weave_gradual_review_memory.py`
- `cognee/tests/e2e/postgres/test_weave_native_storage.py`

The native integration tests use real Enola, Postgres, graph writes, native cache, extraction,
and distillation. Paid model responses/embeddings are fixtures; they prove wiring and retry
behavior, not live model quality, latency, or cost. CI runs them in the maintained fork gate.

Local verification on 2026-09-10: 513 Cognee unit tests passed (one optional test skipped),
10 native Postgres integration tests passed, 1,115 Amberly unit tests passed (75 database-
conditional tests skipped in that command), and the focused real Amberly database test
passed 86 assertions. Amberly root TypeScript, formatting, lint, and governance-engine
build passed. Independent review found and corrected SHA-as-symbol retrieval, quoted
credential leakage, and loss of code results on a paid-memory failure. No live provider
quality/cost or staging deployment acceptance is claimed by these local results.
