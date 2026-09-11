# Agent memory storage

Amberly governance engine owns qualification and maintenance agents. Weave receives their selected results; it does not inspect review sessions, run qualification prompts, cap fact counts, or infer which claims are true.

The internal bearer-authenticated API is:

- `GET /organizations/{organization_id}/repositories/{github_repository_id}/memory-notes?include_retired=true&offset=0&limit=100`
- `POST /organizations/{organization_id}/repositories/{github_repository_id}/memory-notes/apply`
- `POST /organizations/{organization_id}/repositories/{github_repository_id}/memory-notes/jobs/{job_id}/supersede` with `{ "lifecycle_generation": 1 }`

Exact JSON models: `cognee/modules/weave/agent_memory_contracts.py`. A job carries its UUID, lifecycle generation, `review` or `default_branch` source kind, source ID, exact SHA and a list of actions. Each action has stable operation/note UUIDs, expected note version, add/update/retire/unchanged action, full selected content, source paths, provenance and reason. There is no semantic clipping. The host must paginate reads.

Weave resolves the tenant/dataset internally. The caller cannot supply a dataset, owner, or native data ID. Review heads may differ from the indexed default branch; default-branch results must match the current indexed and requested SHA. A completed job returns its original receipts without reapplying even if the branch advanced. The host explicitly supersedes a job only after establishing that its review or default-branch source is obsolete. Weave restores the last committed native note, or forgets a provisional new note, before marking unfinished operations superseded. An interrupted rollback remains retryable and blocks competing writes until recovery completes. Completed jobs remain unchanged. Cancelling before first delivery writes a tombstone. Superseded jobs cannot resume when the repository returns to an earlier SHA.

409 responses include `detail.code` (`stale_source`, `version_conflict`, or `not_ready`) and `detail.message`. Governance can re-qualify after loading a newer note version. A stale source requires a new job against the current source.

The accepted job payload is immutable and saved before native memory processing. Native source completion and document/file links must finish before the note and operation become complete. Retirements run after every add/update in the job so a failed replacement does not remove the previous note. Native `remember`, `update`, and `forget` use the existing Cognee model configuration. `self_improvement` stays unchanged at false for these writes.

Each note uses one native document. Native extracted graph nodes remain connected through that document; file links use `memory_context_for` edges directly from the document, not an extra `ReviewKnowledge` graph. Native update retains the data identity. Retired content and accepted job payloads remain in scoped relational history until repository/customer deletion. Native update itself uses Cognee's deletion/re-ingestion semantics; an interrupted update is unavailable for recall until successfully resumed.

Existing legacy qualified sources are exposed as full notes when read or applied, preserving their original native data IDs and source keys. This is a metadata import, with no LLM calls or semantic rewrite. Historical qualification metadata is carried in provenance for the governance maintainer to inspect. Legacy `ReviewKnowledge` nodes are never newly generated; native update/retirement cleans the old source's annotations.

Local coverage: `test_weave_agent_memory.py` runs native code loading and memory persistence against Postgres; paid model responses and embeddings are fixtures. It covers customer isolation, source revisions, native links, retries, interrupted completion, A → B → A, and replay. This is storage integration proof, not live model-quality evidence.

Migration: `c5e7f9a1b3d6` adds notes/jobs/operations with tenant RLS; fresh-schema bootstrap also includes those tables. Apply migrations with the existing migration/admin role before starting the restricted runtime. No customer dataset rewrite or model change runs in this migration.
