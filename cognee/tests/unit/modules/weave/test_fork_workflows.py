"""Keep fork CI self-contained and require offline coverage before release."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]


def test_fork_workflows_do_not_schedule_upstream_infrastructure():
    paths = set(
        path.name
        for path in (ROOT / ".github/workflows").iterdir()
        if path.suffix in {".yml", ".yaml"}
    )
    assert paths == {"weave-gate.yml", "release-staging.yml", "scorecard.yml", "upstream-sync.yml"}


def test_release_gate_includes_offline_core_cli_and_telemetry_tests():
    gate = yaml.safe_load((ROOT / ".github/workflows/weave-gate.yml").read_text())
    jobs = gate["jobs"]
    assert {
        "python",
        "postgres-isolation",
        "parity-and-security",
        "unit-tests",
        "cli-unit-tests",
        "telemetry-test",
    } <= jobs.keys()
    for name in ("unit-tests", "cli-unit-tests", "telemetry-test"):
        job = jobs[name]
        assert job["timeout-minutes"] <= 30
        assert job["runs-on"] == "ubuntu-22.04"
        text = yaml.safe_dump(job)
        assert "secrets." not in text
        assert "--locked" in text
        assert "--timeout=300" in text
        assert "pytest" in text
