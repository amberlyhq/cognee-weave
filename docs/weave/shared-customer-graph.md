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
