"""Exercise the export assertions executed by the shell parity/restore gates."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[5]
ORG = "7e1a7b9d-08c2-4f57-9884-623e01b68b02"
SHA = "b" * 40


def _export():
    return {
        "organization_id": ORG,
        "repositories": [{"github_repository_id": 920002, "indexed_default_sha": SHA}],
        "native_graph": json.dumps(
            [
                {
                    "dataset_id": "04f787f8-3071-4c7e-8c51-34bb68bd8199",
                    "scope": "customer",
                    "nodes": [["node", {"name": "BetaCanary"}]],
                    "edges": [["node", "node", "references", {}]],
                },
                {
                    "dataset_id": "07432929-98df-4e54-95d1-7d8d4e1305be",
                    "scope": "customer",
                    "nodes": [],
                    "edges": [],
                },
            ]
        ),
    }


def _validate(script, value, tmp_path):
    source = (ROOT / "scripts" / script).read_text()
    snippets = re.findall(r"python -c '(.*?)\n'", source, re.S)
    validation = next(item for item in snippets if "value = json.load(sys.stdin)" in item)
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(_export()))
    return subprocess.run(
        [sys.executable, "-c", validation],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "ORGANIZATION_ID": ORG,
            "REPOSITORY_ID": "920002",
            "SHA": SHA,
            "RESTORE_ORGANIZATION_ID": ORG,
            "RESTORE_REPOSITORY_ID": "920002",
            "RESTORE_EXPECTED_EXPORT": str(expected),
        },
    )


@pytest.mark.parametrize("script", ["weave-parity.sh", "weave-restore-drill.sh"])
def test_native_customer_export_passes_the_actual_script_validation(script, tmp_path):
    result = _validate(script, _export(), tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["weave-parity.sh", "weave-restore-drill.sh"])
@pytest.mark.parametrize("mutation", ["organization", "repository", "sha", "scope", "nodes"])
def test_export_validation_still_rejects_wrong_identity_or_missing_graph(
    script, mutation, tmp_path
):
    value = _export()
    if mutation == "organization":
        value["organization_id"] = "foreign"
    elif mutation == "repository":
        value["repositories"][0]["github_repository_id"] = 920001
    elif mutation == "sha":
        value["repositories"][0]["indexed_default_sha"] = "a" * 40
    else:
        native = json.loads(value["native_graph"])
        native[0][mutation] = [] if mutation == "nodes" else "repository"
        value["native_graph"] = json.dumps(native)
    assert _validate(script, value, tmp_path).returncode != 0


def test_restore_compares_native_graph_contents_not_json_formatting(tmp_path):
    value = _export()
    value["native_graph"] = json.dumps(json.loads(value["native_graph"]), indent=2, sort_keys=True)
    result = _validate("weave-restore-drill.sh", value, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation", ["dataset", "content"])
def test_restore_rejects_changed_native_dataset_or_content(mutation, tmp_path):
    value = _export()
    native = json.loads(value["native_graph"])
    if mutation == "dataset":
        native[0]["dataset_id"] = "794a2ddd-2e80-4eeb-a7fe-f11ae0c45843"
    else:
        native[0]["nodes"][0][1]["name"] = "ForeignCanary"
    value["native_graph"] = json.dumps(native)
    assert _validate("weave-restore-drill.sh", value, tmp_path).returncode != 0
