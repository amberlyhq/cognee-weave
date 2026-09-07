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
The implementation lives in `.github/workflows/upstream-sync.yml`.

Every synchronization pull request must:

1. Record the exact proposed upstream SHA.
2. Preserve the tenant, provenance, migration, and Postgres storage boundaries.
3. Pass Cognee Weave's unit, Postgres isolation, migration, and security gates.
4. Receive manual review before merge.

## Accepted upstream history

| Accepted on | Upstream SHA | Notes |
| --- | --- | --- |
| 2026-08-31 | `8b86f868fcab8d688b41f430e4cc7be1d8427e11` | Initial fork point |

## Fork CI and release ownership

The active workflows are owned by Amberly:

- `weave-gate.yml`: locked Python dependencies, correctness lint, offline core/CLI/
  telemetry tests, Weave unit tests, real Postgres tenant/index/recall/deletion tests,
  migrations, package build, secret scan, Docker isolation, performance and restore checks.
- `release-staging.yml`: a successful main-push gate releases that exact commit to
  private Railway staging, migrations first. GitHub's staging environment holds the
  staging-only token. Direct Railway repository auto-deploy links stay disabled.
- `scorecard.yml`: supply-chain security analysis.
- `upstream-sync.yml`: reviewed upstream update proposals, never automatic merges.

Inherited Cognee publishing, private-runner, Linear/team automation, and unused
provider/OS/example matrices are removed. They assumed Cognee-owned DockerHub,
PyPI, cloud services and credentials. We do not publish to Cognee's registries or
claim coverage for its full provider matrix. Their source tests remain available;
the useful secret-free core, CLI and telemetry suites now block our release gate.
Historical failed/cancelled runs remain as audit history; new commits use this contract.
