# Upstream

Cognee Weave is an Apache-2.0 fork of
[`topoteretes/cognee`](https://github.com/topoteretes/cognee), maintained by
Amberly for its Postgres-only, tenant-scoped memory and code-context needs.

## Fork point

- Upstream repository: `https://github.com/topoteretes/cognee.git`
- Initial upstream branch: `main`
- Initial upstream SHA: `8b86f868fcab8d688b41f430e4cc7be1d8427e11`
- Initial upstream commit date: `2026-08-31T11:06:01Z`
- Initial Cognee version: `1.5.3`
- License: Apache License 2.0; the upstream `LICENSE` and notices are preserved.

## Remote policy

- `origin` is `amberlyhq/cognee-weave`.
- `upstream` fetches from `topoteretes/cognee` and has push disabled.
- Amberly-specific code stays in a thin, clearly named layer.
- General fixes should be prepared so they can be proposed upstream.

## Synchronization policy

Upstream is reviewed weekly. Automation may fetch an exact upstream SHA,
create a synchronization branch, and open or update a pull request. It must
never merge automatically or push directly to the protected default branch.

Every synchronization pull request must:

1. Record the exact proposed upstream SHA.
2. Preserve the tenant, provenance, migration, and Postgres storage boundaries.
3. Pass Cognee Weave's unit, Postgres isolation, migration, and security gates.
4. Receive manual review before merge.

## Accepted upstream history

| Accepted on | Upstream SHA | Notes |
| --- | --- | --- |
| 2026-08-31 | `8b86f868fcab8d688b41f430e4cc7be1d8427e11` | Initial fork point |
