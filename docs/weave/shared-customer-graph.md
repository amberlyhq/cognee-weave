# Shared customer code and review graph

Weave uses one native Cognee dataset per organization. Code from all of that
organization's repositories and qualified review memory share that dataset.
Organizations keep separate dataset owners, Postgres schemas, and access scopes.
Repository identity is checked in addition to organization identity when linking
facts or removing repository data.

Code indexing still uses native Cognee code extraction without an LLM or vector
embedding pass. Real repository paths receive deterministic file nodes, including
configuration files that do not contain parsed symbols. Qualified review facts
retain their exact statement, certainty, historical status, reviewed commit, and
source evidence. A fact links to an exact file in its own repository and to its
source document chunk. Missing or ambiguous paths remain unlinked. Code recall
returns these annotations with their original qualifications.

## Consolidating legacy datasets

Run from the deployed Weave service with its existing database configuration:

```python
from uuid import UUID
from cognee.modules.weave.dataset_migration import migrate_customer_datasets

organization_id = UUID("<explicit organization UUID>")
await migrate_customer_datasets(organization_id, dry_run=True)
await migrate_customer_datasets(organization_id)
```

The dry run executes the migration and verification inside a transaction, then
rolls it back. Apply only after checking the dry-run result and recording a
separate customer's canary state. The migration checks ownership, schema,
provenance and supported table versions. It preserves historical graph/vector
content and source receipts. Unexpected ownership, unknown legacy datasets, or
unsupported collisions abort the transaction. Legacy schemas are retired only
after exact copy verification, in the same transaction.

After migration, reindex each repository at its exact intended commit. Pipeline
v5 deliberately requires a fresh code snapshot rather than relabeling an old
snapshot as verified. This creates canonical file anchors and reconnects
qualified facts without rebuilding all memory through the LLM. Verify receipt
hashes, historical documents, the new links, live recall, and the other customer's
unchanged canary before declaring acceptance.

## Verification

The maintained gate includes migration tests against real Postgres. Local release
checks passed: 503 focused tests (14 skipped), 36 Postgres tests, correctness lint,
and wheel/source builds. Tests cover identical paths and IDs in different
customers, repository-specific deletion, foreign ownership/provenance rejection,
rollback on copy or retirement failure, and A → B → A file/link restoration.
Deployment and customer migration require separate live acceptance evidence.

## Merge knowledge maintenance

Default-branch indexing records file hashes and marks facts affected by changed,
deleted or dependent code `needs_recheck` before refreshing the native graph.
The initial code import still makes no model calls. Pipeline v6 requires a fresh
index to establish these hashes.

Amberly then uploads the same authenticated exact-commit archive to
`POST /api/v1/weave/organizations/{organization_id}/repositories/merge-memory`.
Weave compares every file hash against the indexed source and rejects mismatches.
The current repository lifecycle and indexed commit must still match. Native
hash changes can complete a GitHub file comparison that reached its result cap.

A bounded qualification step uses merged source as authority and completed review
context only as hints. An independent audit rejects unsupported statements;
rejections become feedback for bounded correction. Accepted selections are saved
before native remember. Native remember receives only qualified statements and
source references, while exact source quotes stay in receipts and annotations.
Model, provider and embedding configuration remain unchanged.

Verified replacements become current facts, with the checked commit and exact
file links for every supporting source path. Historical originals remain stored
and reference their replacement source. Unsupported claims remain unverified.
Repeated delivery reuses the saved plan and selections. A return to an earlier
commit creates a new indexing epoch, so it cannot reuse old validity statuses.
Each request processes at most four unfinished batches; Amberly saves continuation
rounds independently from the successful code-index step.

Current semantic recall uses native CHUNKS with an allowlist of completed,
verified source tags and a second check of returned source IDs. Empty allowlists
skip semantic recall. Shared historical entity descriptions cannot enter this
lane. Code-linked historical annotations carry explicit lifecycle labels.

Source batches are bounded to 28,000 characters and 12 prior facts. Truncated or
binary source stays unverified and produces `coverage_complete=false`; completion
of processing never means every possible repository fact was proven. Amberly
limits a single continuation chain to 128 rounds and reports exhaustion explicitly.

Local verification includes native PostgreSQL ingestion and CHUNKS retrieval,
replay without model calls, A-B-A epochs, customer canaries, deletion, and injected
provider failures. Live provider quality and deployment are verified separately
in the staging E2E run.

Merge-maintenance release checks: 555 focused tests passed (14 optional skips),
38 real PostgreSQL tests passed, correctness lint passed, and wheel/source builds
passed. A crash injected after hash stamping reproduced lost changed paths with
the old algorithm; the fixed retry retained the delta and qualified the changed
source even when the external comparison was incomplete.
