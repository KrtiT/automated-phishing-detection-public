"""Bind prospective measurement choices to code and preserved contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from automated_phishing_detection import http_replay

ROOT = Path(__file__).resolve().parents[1]


def contract(name):
    return json.loads((ROOT / "data" / name).read_text())


@pytest.mark.parametrize(
    "name", ["http-replay-contract-v1.json", "evaluation-manifest-contract-v1.json"]
)
def test_supplements_preserve_executed_method_and_do_not_unlock_access(name):
    value = contract(name)
    assert value["protected_evaluation_ready"] is False
    assert value["status"] == "specified_synthetic_integration"
    for key, filename in (
        ("matrix_sha256", "docs/advisor-approval/2026-08-16-realignment-matrix.md"),
        ("evaluation_contract_sha256", "data/evaluation-contract-v1.json"),
    ):
        assert value[key] == hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()


def test_http_contract_binds_versions_and_client_conventions():
    value = contract("http-replay-contract-v1.json")
    runtime = value["runtime"]
    assert (
        runtime["uv_lock_sha256"]
        == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    # Inspect declared pins, without relying on which optional wheel was installed.
    dependencies = (ROOT / "pyproject.toml").read_text()
    for package, version in runtime["versions"].items():
        assert f'"{package}=={version}"' in dependencies
    assert value["client"]["total_deadline_ms"] == http_replay.DEADLINE_SECONDS * 1000
    assert value["runs"]["concurrency_order"] == list(http_replay.CONCURRENCIES)
    assert value["runs"]["primary_http"]["measured_denominator"] == 50000
    assert value["runs"]["reference_invocations"] == {
        "prevalence_basis_points": 100,
        "concurrency": 1,
        "run_index": 1,
        "denominator": 10000,
        "maximum_fraction": 0.3,
    }
    assert (
        value["singleton_amendment_sha256"]
        == hashlib.sha256(
            (ROOT / "data/singleton-inference-amendment-v1.json").read_bytes()
        ).hexdigest()
    )


def test_manifest_contract_fixes_counts_roles_and_label_free_controls():
    value = contract("evaluation-manifest-contract-v1.json")
    manifest = value["replay_manifest"]
    assert manifest["seed"] == 20260816
    assert manifest["prevalence_basis_points"] == [10, 100, 500]
    for counts in manifest["class_counts"]:
        assert counts["negative"] + counts["positive"] == 10000
        assert counts["positive"] == counts["basis_points"]
    assert set(value["external_stream_join"]["roles"]) == {
        "gold",
        "certified",
        "tranco",
        "secondary",
    }
    assert value["external_stream_join"]["roles"]["tranco"].startswith("null;")
