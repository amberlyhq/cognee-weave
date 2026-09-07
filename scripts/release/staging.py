#!/usr/bin/env python3
"""Release a gate-verified Git commit to private Railway staging; never production."""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ENDPOINT = "https://backboard.railway.com/graphql/v2"


def exact_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("An immutable full commit SHA is required")
    return value


def deployment_sha(meta):
    if isinstance(meta, str):
        meta = json.loads(meta)
    return exact_sha(meta.get("commitHash") if isinstance(meta, dict) else None)


def release(client, sha, evidence):
    exact_sha(sha)
    evidence.update(schemaVersion=1, environment="staging", sha=sha, deployments=[])
    client.check_scope()
    prior = client.current()
    evidence["priorDeployment"] = prior
    api_attempted = False
    try:
        for service in ("migrate", "weave"):
            api_attempted = service == "weave"
            deployment_id = client.deploy(service, sha)
            attempt = {"service": service, "id": deployment_id}
            evidence["deployments"].append(attempt)
            attempt.update(client.wait(deployment_id, sha, migration=service == "migrate"))
    except Exception:
        evidence["status"] = "failed"
        if api_attempted and prior:
            try:
                restored_id = client.rollback(prior["id"])
                evidence["rollback"] = client.wait(restored_id, prior["sha"])
            except Exception:
                evidence["rollback"] = {"status": "failed", "priorDeploymentId": prior["id"]}
                raise RuntimeError("Staging release failed and API rollback could not be verified")
        raise
    evidence["status"] = "success"
    return evidence


class Railway:
    def __init__(self, env):
        self.token = self.required(env, "RAILWAY_PROJECT_TOKEN")
        self.environment_id = self.required(env, "RAILWAY_ENVIRONMENT_ID")
        self.services = {
            "migrate": self.required(env, "RAILWAY_MIGRATE_SERVICE_ID"),
            "weave": self.required(env, "RAILWAY_WEAVE_SERVICE_ID"),
        }
        if self.services["migrate"] == self.services["weave"]:
            raise ValueError("Migration and API services must be distinct")
        self.scoped = False

    @staticmethod
    def required(env, name):
        value = env.get(name)
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def api(self, query, variables):
        request = Request(
            ENDPOINT,
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Content-Type": "application/json",
                "Project-Access-Token": self.token,
                "User-Agent": "cognee-weave-staging-release/1.0",
            },
        )
        # Never include provider response bodies or credentials in release artifacts/errors.
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except Exception:
            raise RuntimeError("Railway API request failed") from None
        if result.get("errors") or not result.get("data"):
            raise RuntimeError("Railway rejected the API request")
        return result["data"]

    def check_scope(self):
        if self.scoped:
            return
        scope = self.api("query Scope { projectToken { projectId environmentId } }", {})[
            "projectToken"
        ]
        if not scope or scope.get("environmentId") != self.environment_id:
            raise RuntimeError("Railway project token environment mismatch")
        environment = self.api(
            "query Environment($id: String!) { environment(id: $id) { name projectId } }",
            {"id": self.environment_id},
        )["environment"]
        if environment["name"] != "staging" or environment["projectId"] != scope["projectId"]:
            raise RuntimeError("Release target must be staging in the token project")
        self.scoped = True

    def detail(self, deployment_id):
        return self.api(
            "query Deployment($id: String!) { deployment(id: $id) "
            "{ id status meta deploymentStopped } }",
            {"id": deployment_id},
        )["deployment"]

    def current(self):
        self.check_scope()
        latest = self.api(
            "query Current($serviceId: String!, $environmentId: String!) { "
            "serviceInstance(serviceId: $serviceId, environmentId: $environmentId) "
            "{ latestDeployment { id status } } }",
            {"serviceId": self.services["weave"], "environmentId": self.environment_id},
        )["serviceInstance"]["latestDeployment"]
        if latest is None:
            return None
        detail = self.detail(latest["id"])
        if detail["status"] != "SUCCESS" or detail["deploymentStopped"]:
            raise RuntimeError("Current API deployment is not healthy; refusing automatic release")
        return {"id": latest["id"], "sha": deployment_sha(detail["meta"])}

    def deploy(self, service, sha):
        self.check_scope()
        result = self.api(
            "mutation Deploy($environmentId: String!, $serviceId: String!, $commitSha: String!) "
            "{ serviceInstanceDeployV2(environmentId: $environmentId, "
            "serviceId: $serviceId, commitSha: $commitSha) }",
            {
                "environmentId": self.environment_id,
                "serviceId": self.services[service],
                "commitSha": exact_sha(sha),
            },
        )["serviceInstanceDeployV2"]
        if not result:
            raise RuntimeError("Railway returned no deployment ID")
        return result

    def wait(self, deployment_id, sha, *, migration=False):
        exact_sha(sha)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            detail = self.detail(deployment_id)
            status = detail["status"]
            if status in {"FAILED", "CRASHED", "REMOVED", "SKIPPED", "CANCELED"}:
                raise RuntimeError(f"Railway deployment {deployment_id} ended in {status}")
            if status == "SUCCESS":
                if deployment_sha(detail["meta"]) != sha:
                    raise RuntimeError("Railway deployed commit mismatch")
                # A migration must have exited; SUCCESS alone can mean it is still running.
                if migration and not detail["deploymentStopped"]:
                    time.sleep(5)
                    continue
                if not migration and detail["deploymentStopped"]:
                    raise RuntimeError("API deployment stopped after startup")
                return {
                    "id": deployment_id,
                    "sha": sha,
                    "status": status,
                    "deploymentStopped": detail["deploymentStopped"],
                }
            time.sleep(5)
        raise RuntimeError(f"Railway deployment {deployment_id} timed out")

    def rollback(self, deployment_id):
        self.check_scope()
        return self.api(
            "mutation Rollback($id: String!) { deploymentRollback(id: $id) { id } }",
            {"id": deployment_id},
        )["deploymentRollback"]["id"]


def main():
    evidence = {"createdAt": datetime.now(timezone.utc).isoformat(), "status": "failed"}
    try:
        sha = exact_sha(os.environ.get("RELEASE_SHA"))
        if sha != os.environ.get("VERIFIED_WORKFLOW_SHA"):
            raise ValueError("Release SHA must match the successful gate SHA")
        checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if checkout != sha:
            raise ValueError("Checkout does not match the verified release SHA")
        release(Railway(os.environ), sha, evidence)
        return 0
    except Exception as error:
        # Exception types and this script's fixed messages are safe; HTTP bodies are not retained.
        print(str(error), file=sys.stderr)
        return 1
    finally:
        Path("release-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
