"""The complete candidate declaration remains non-executable and unadopted."""

import inspect
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256

import pytest
from operational_profile_fixtures import (
    COMMON,
    REQUIRED,
    WORKING,
    api,
    profile_case,
    resolve,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_schedule import planned_cells

__all__ = ["profile_case"]


def test_candidate_is_canonical_closed_and_incomplete(profile_case):
    profile = resolve(profile_case)
    value = profile.projection()
    assert profile.canonical_bytes == canonical_bytes(value)
    assert profile.profile_sha256 == sha256(profile.canonical_bytes).hexdigest()
    assert value["schema_version"] == 1
    assert value["profile_id"] == "operational-candidate-v1"
    assert value["status"] == "incomplete_closed_candidate"
    assert value["protected_evaluation_ready"] is False
    assert value["protected_evaluation_authorized"] is False
    assert profile.protected_evaluation_ready is False
    assert value["protective_deadlines_seconds"] == dict.fromkeys(
        ("startup", "shutdown", "terminate", "kill")
    )
    assert value["session_exclusivity"] == "pending_review"
    assert value["pre_prediction_policy"] == {
        "scope": "whole_study",
        "dataset_check": "both_datasets_once_before_predictions",
        "missing_promised_task": "hold_entire_study",
        "status": "selected_for_review_not_adopted",
    }


def test_profile_joins_original_schedule_execution_and_all_public_pins(profile_case):
    value = resolve(profile_case).projection()
    assert value["execution"] == {
        "revision": "a" * 40,
        "execution_contract_sha256": "b" * 64,
        "runtime_sha256": sha256(
            profile_case.binding.runtime_json.encode()
        ).hexdigest(),
        "source_spec_sha256": profile_case.hashes["data/sources.json"],
    }
    assert value["bound_file_sha256"] == profile_case.hashes
    assert value["schedule"] == {
        "cells": [asdict(cell) for cell in planned_cells()],
        "reference_ordinal": 1,
        "primary_http_ordinals": [21, 22, 23, 24, 25],
    }
    assert profile_case.checks == [profile_case.binding, profile_case.binding]


def test_fixed_commands_expose_no_execution_override(profile_case):
    assert resolve(profile_case).projection()["commands"] == {
        "service": {
            "script": "scripts/run_operational_service.py",
            "arguments": COMMON
            + [
                "--length-only",
                "--logistic-l1",
                "--transformer-bundle",
                "--gmm",
            ],
        },
        "client": {"script": "scripts/run_operational_client.py", "arguments": COMMON},
    }
    assert tuple(inspect.signature(api().resolve_operational_profile).parameters) == (
        "binding",
    )


def test_exact_success_inventory_and_manifest_conventions(profile_case):
    value = resolve(profile_case).projection()
    outputs = WORKING[1:]
    snapshot = [f"attempt/{name}" for name in WORKING]
    snapshot += ["attempt/finalize.claim", "attempt/outcome.json"]
    snapshot += [f"attempt/evidence/{name}" for name in outputs]
    snapshot += ["public-summary.json"]
    assert len(snapshot) == len(set(snapshot)) == 36
    assert value["retention"] == {
        "protocol": "operational-cell-v1",
        "working_names": WORKING,
        "private_output_names": outputs,
        "snapshot_names": snapshot,
    }
    assert value["manifest_sha256"] == {
        "http": "original_compact_sorted_utf8_json_without_newline",
        "shift": "exact_retained_test_jsonl_bytes_in_retained_order",
    }


def test_frozen_candidate_and_fresh_projections_cannot_open_gate(profile_case):
    profile = resolve(profile_case)
    original = profile.canonical_bytes
    changed = profile.projection()
    changed["protected_evaluation_ready"] = True
    changed["protective_deadlines_seconds"]["startup"] = 60
    changed["schedule"]["cells"].clear()
    assert profile.canonical_bytes == original
    assert profile.protected_evaluation_ready is False
    with pytest.raises(FrozenInstanceError):
        profile.canonical_bytes = canonical_bytes(changed)


@pytest.mark.parametrize(
    "name", (*REQUIRED, "src/automated_phishing_detection/other_bound_module.py")
)
def test_every_bound_public_digest_changes_candidate_identity(profile_case, name):
    original = resolve(profile_case)
    hashes = profile_case.hashes | {name: "e" * 64}
    changed = resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert changed.profile_sha256 != original.profile_sha256


def test_resolution_never_reads_sources_models_or_starts_processes(
    profile_case, monkeypatch
):
    import subprocess
    from pathlib import Path

    from automated_phishing_detection import bound_models, phiusiil

    def forbidden(*args, **kwargs):
        pytest.fail("metadata candidate performed protected or numerical work")

    for owner, name in (
        (Path, "read_bytes"),
        (Path, "open"),
        (subprocess, "Popen"),
        (bound_models, "load_bound_models"),
        (phiusiil, "prepare_phiusiil"),
        (phiusiil, "resolve_rows"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    profile = resolve(profile_case)
    assert profile.protected_evaluation_ready is False
