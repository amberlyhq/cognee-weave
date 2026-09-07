"""Exercise the real release runner with only Railway's network boundary replaced."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[5] / "scripts/release/staging.py"
spec = importlib.util.spec_from_file_location("weave_staging_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
SHA = "a" * 40
OLD = "b" * 40


class Railway:
    def __init__(self, fail=None, wrong_sha=False):
        self.events = []
        self.fail = fail
        self.wrong_sha = wrong_sha

    def check_scope(self):
        self.events.append("scope")
        if self.fail == "scope":
            raise RuntimeError("environment mismatch")

    def current(self):
        self.events.append("current")
        return {"id": "old", "sha": OLD}

    def deploy(self, service, sha):
        self.events.append(("deploy", service, sha))
        return service

    def wait(self, deployment_id, sha, *, migration=False):
        self.events.append(("wait", deployment_id, sha))
        if self.fail == deployment_id:
            raise RuntimeError("deployment failed")
        if self.wrong_sha and deployment_id == "migrate":
            raise RuntimeError("deployed commit mismatch")
        return {"id": deployment_id, "sha": sha, "status": "SUCCESS"}

    def rollback(self, deployment_id):
        self.events.append(("rollback", deployment_id))
        return "restored"


def test_migration_finishes_before_api_is_started():
    client = Railway()
    evidence = {}
    release.release(client, SHA, evidence)
    assert client.events == [
        "scope",
        "current",
        ("deploy", "migrate", SHA),
        ("wait", "migrate", SHA),
        ("deploy", "weave", SHA),
        ("wait", "weave", SHA),
    ]
    assert evidence["status"] == "success"


@pytest.mark.parametrize("failure", ["scope", "migrate"])
def test_failure_before_api_never_deploys_api(failure):
    client = Railway(fail=failure)
    with pytest.raises(RuntimeError):
        release.release(client, SHA, {})
    assert ("deploy", "weave", SHA) not in client.events
    assert not any(isinstance(e, tuple) and e[0] == "rollback" for e in client.events)


def test_wrong_migration_commit_prevents_api():
    client = Railway(wrong_sha=True)
    with pytest.raises(RuntimeError, match="commit mismatch"):
        release.release(client, SHA, {})
    assert ("deploy", "weave", SHA) not in client.events


def test_api_failure_restores_previous_exact_commit():
    client = Railway(fail="weave")
    evidence = {}
    with pytest.raises(RuntimeError, match="deployment failed"):
        release.release(client, SHA, evidence)
    assert client.events[-2:] == [("rollback", "old"), ("wait", "restored", OLD)]
    assert evidence["rollback"]["sha"] == OLD


def test_nonimmutable_sha_rejected_before_provider_calls():
    client = Railway()
    with pytest.raises(ValueError):
        release.release(client, "main", {})
    assert client.events == []


def real_client(monkeypatch, responses):
    client = release.Railway(
        {
            "RAILWAY_PROJECT_TOKEN": "test-placeholder",
            "RAILWAY_ENVIRONMENT_ID": "staging-id",
            "RAILWAY_MIGRATE_SERVICE_ID": "migrate-id",
            "RAILWAY_WEAVE_SERVICE_ID": "api-id",
        }
    )
    calls = []

    def api(query, variables):
        calls.append((query, variables))
        return responses.pop(0)

    monkeypatch.setattr(client, "api", api)
    return client, calls


@pytest.mark.parametrize("scope", [None, {"projectId": "project", "environmentId": "prod-id"}])
def test_real_client_rejects_token_environment_mismatch_before_mutation(monkeypatch, scope):
    client, calls = real_client(monkeypatch, [{"projectToken": scope}])
    with pytest.raises(RuntimeError, match="environment mismatch"):
        client.deploy("migrate", SHA)
    assert len(calls) == 1
    assert calls[0][0].startswith("query ")


def test_configured_production_token_cannot_make_this_a_production_release(monkeypatch):
    client, calls = real_client(
        monkeypatch,
        [
            {"projectToken": {"projectId": "project", "environmentId": "staging-id"}},
            {"environment": {"projectId": "project", "name": "production"}},
        ],
    )
    with pytest.raises(RuntimeError, match="must be staging"):
        client.deploy("migrate", SHA)
    assert all(query.startswith("query ") for query, _ in calls)


def test_real_client_passes_full_sha_and_scoped_ids_to_provider(monkeypatch):
    client, calls = real_client(
        monkeypatch,
        [
            {"projectToken": {"projectId": "project", "environmentId": "staging-id"}},
            {"environment": {"projectId": "project", "name": "staging"}},
            {"serviceInstanceDeployV2": "deployment-id"},
        ],
    )
    assert client.deploy("migrate", SHA) == "deployment-id"
    assert calls[-1][1] == {
        "environmentId": "staging-id",
        "serviceId": "migrate-id",
        "commitSha": SHA,
    }


def detail(*, sha=SHA, stopped=False, status="SUCCESS"):
    return {
        "deployment": {
            "id": "deployment-id",
            "status": status,
            "meta": {"commitHash": sha},
            "deploymentStopped": stopped,
        }
    }


def test_successful_start_is_not_completed_migration(monkeypatch):
    client, calls = real_client(monkeypatch, [detail(), detail(stopped=True)])
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    receipt = client.wait("deployment-id", SHA, migration=True)
    assert receipt["deploymentStopped"] is True
    assert sleeps == [5]
    assert len(calls) == 2


def test_provider_wrong_commit_is_rejected_even_with_success(monkeypatch):
    client, _ = real_client(monkeypatch, [detail(sha=OLD)])
    with pytest.raises(RuntimeError, match="commit mismatch"):
        client.wait("deployment-id", SHA)


@pytest.mark.parametrize("status", ["FAILED", "CRASHED", "REMOVED", "SKIPPED", "CANCELED"])
def test_provider_terminal_failure_is_not_success(monkeypatch, status):
    client, _ = real_client(monkeypatch, [detail(status=status)])
    with pytest.raises(RuntimeError, match="ended in"):
        client.wait("deployment-id", SHA, migration=True)


def test_stopped_api_is_not_accepted(monkeypatch):
    client, _ = real_client(monkeypatch, [detail(stopped=True)])
    with pytest.raises(RuntimeError, match="stopped after startup"):
        client.wait("deployment-id", SHA)


def test_rollback_failure_is_reported_without_claiming_recovery():
    client = Railway(fail="weave")
    client.rollback = lambda _: "weave"
    evidence = {}
    with pytest.raises(RuntimeError, match="rollback could not be verified"):
        release.release(client, SHA, evidence)
    assert evidence["status"] == "failed"
    assert evidence["rollback"]["status"] == "failed"


@pytest.mark.parametrize("gate_sha,checkout", [(OLD, SHA), (SHA, OLD)])
def test_entrypoint_rejects_unverified_input_before_credentials(
    monkeypatch, tmp_path, gate_sha, checkout
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_SHA", SHA)
    monkeypatch.setenv("VERIFIED_WORKFLOW_SHA", gate_sha)
    monkeypatch.setattr(release.subprocess, "check_output", lambda *a, **kw: checkout)
    monkeypatch.setattr(release, "Railway", lambda _: pytest.fail("provider client must not start"))
    assert release.main() == 1
    assert '"status": "failed"' in (tmp_path / "release-evidence.json").read_text()


def test_release_workflow_only_accepts_same_repo_successful_main_push():
    import yaml

    workflow = yaml.load(
        (SCRIPT.parents[2] / ".github/workflows/release-staging.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"] == {
        "workflow_run": {"workflows": ["Cognee Weave Gate"], "types": ["completed"]}
    }
    job = workflow["jobs"]["release"]
    assert job["environment"] == "staging"
    for guard in (
        "conclusion == 'success'",
        "event == 'push'",
        "head_branch == 'main'",
        "repository.full_name == github.repository",
        "head_repository.full_name == github.repository",
    ):
        assert guard in job["if"]
    assert job["steps"][0]["with"]["ref"] == "${{ env.RELEASE_SHA }}"
    assert job["steps"][-1]["if"] == "always()"


def test_failure_evidence_keeps_attempted_deployment_id():
    client = Railway(fail="migrate")
    evidence = {}
    with pytest.raises(RuntimeError):
        release.release(client, SHA, evidence)
    assert evidence["deployments"] == [{"service": "migrate", "id": "migrate"}]


def test_api_uses_explicit_user_agent_for_railway_edge(monkeypatch):
    import io
    from urllib.error import HTTPError

    client = release.Railway(
        {
            "RAILWAY_PROJECT_TOKEN": "test-placeholder",
            "RAILWAY_ENVIRONMENT_ID": "staging-id",
            "RAILWAY_MIGRATE_SERVICE_ID": "migrate-id",
            "RAILWAY_WEAVE_SERVICE_ID": "api-id",
        }
    )

    def edge(request, **kwargs):
        # Railway's edge returns HTTP403/1010 for urllib's default user agent.
        if not request.get_header("User-agent"):
            raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
        return io.BytesIO(b'{"data":{"ok":true}}')

    monkeypatch.setattr(release, "urlopen", edge)
    assert client.api("query { ok }", {}) == {"ok": True}
