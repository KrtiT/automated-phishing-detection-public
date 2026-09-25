"""The declared complete orchestration remains metadata-only and unadopted."""

from dataclasses import replace
from hashlib import sha256

import pytest
from operational_profile_fixtures import STUDY_REQUIRED, api, profile_case, resolve

from automated_phishing_detection import _study_root_records as root_records
from automated_phishing_detection._external_completion_files import PAYLOAD_NAMES
from automated_phishing_detection._external_completion_records import _LOGICAL_NAMES
from automated_phishing_detection._internal_handoff_validation import SNAPSHOT_NAMES
from automated_phishing_detection._study_run_schema import STAGES
from automated_phishing_detection.study_preparation_retention import PREPARATION_ORDER

__all__ = ["profile_case"]


def test_closed_whole_study_fields_and_shared_command(profile_case):
    value = resolve(profile_case).projection()["whole_study"]
    from automated_phishing_detection import _study_cli_protocol as command

    assert set(value) == {
        "protocol",
        "command",
        "stage_names",
        "preparation",
        "source_pair",
        "retention",
    }
    assert value["protocol"] == "study-root-v1"
    assert value["command"] == {
        "script": command.SCRIPT,
        "arguments": list(command.ARGUMENTS),
    }
    assert command.SCRIPT == "scripts/run_study.py"
    assert len(command.ARGUMENTS) == 33
    assert value["stage_names"] == list(STAGES)
    assert len(STAGES) == len(set(STAGES)) == 14


def test_original_preparation12_excludes_separate_reservation(profile_case):
    preparation = resolve(profile_case).projection()["whole_study"]["preparation"]
    assert preparation == {
        "protocol": "study-preparation-v1",
        "reservation_name": "reservation.json",
        "payload_names": list(PREPARATION_ORDER),
    }
    assert len(PREPARATION_ORDER) == len(set(PREPARATION_ORDER)) == 12
    assert "reservation.json" not in PREPARATION_ORDER
    assert PREPARATION_ORDER[-2:] == ("feasibility.json", "preparation-complete.json")


def test_same_parent_source_order_and_unchanged_snapshot_inventories(profile_case):
    sources = resolve(profile_case).projection()["whole_study"]["source_pair"]
    assert sources == {
        "order": ["internal", "external"],
        "source_interface": "retained_study_preparation_v1",
        "internal": {
            "script": "scripts/run_prepared_internal_evaluation.py",
            "fixed_flags": ["--worker"],
            "snapshot_names": sorted(SNAPSHOT_NAMES),
        },
        "external": {
            "script": "scripts/run_prepared_external_evaluation.py",
            "fixed_flags": [],
            "private_output_names": sorted(PAYLOAD_NAMES),
            "snapshot_names": sorted(_LOGICAL_NAMES),
        },
    }
    assert len(SNAPSHOT_NAMES) == 35
    assert len(PAYLOAD_NAMES) == 36
    assert len(_LOGICAL_NAMES) == 76


@pytest.mark.parametrize("success", [False, True])
def test_root_exact10_or14_logical_names(profile_case, success):
    retention = resolve(profile_case).projection()["whole_study"]["retention"]
    assert set(retention) == {"whole_study_hold", "study_evidence_published"}
    status = "study_evidence_published" if success else "whole_study_hold"
    checkpoints = root_records.ORDER if success else root_records.HOLD_ORDER
    outputs = checkpoints + (root_records.EXTRA_NAMES if success else ())
    names = ["attempt/reservation.json"]
    names.extend(f"attempt/{name}" for name in checkpoints)
    names.extend(("attempt/finalize.claim", "attempt/outcome.json"))
    names.extend(f"attempt/evidence/{name}" for name in outputs)
    names.append("public-summary.json")
    assert retention[status] == {
        "checkpoint_names": list(checkpoints),
        "private_output_names": list(outputs),
        "snapshot_names": names,
    }
    assert len(names) == len(set(names)) == (14 if success else 10)


@pytest.mark.parametrize("name", STUDY_REQUIRED)
def test_whole_study_declarations_require_each_public_pin(profile_case, name):
    hashes = profile_case.hashes.copy()
    del hashes[name]
    changed = replace(profile_case.binding, source_hashes=tuple(sorted(hashes.items())))
    with pytest.raises(api().OperationalProfileError):
        api().resolve_operational_profile(changed)


def test_study_projection_is_fresh_and_cannot_adopt_policy(profile_case):
    profile = resolve(profile_case)
    before = profile.canonical_bytes
    changed = profile.projection()
    changed["whole_study"]["command"]["arguments"].append("--resume")
    changed["whole_study"]["preparation"]["payload_names"].clear()
    changed["whole_study"]["retention"]["whole_study_hold"]["snapshot_names"].clear()
    assert profile.canonical_bytes == before
    assert profile.profile_sha256 == sha256(before).hexdigest()
    assert profile.protected_evaluation_ready is False
    current = profile.projection()
    assert current["protected_evaluation_ready"] is False
    assert current["protected_evaluation_authorized"] is False
    assert current["status"] == "incomplete_closed_candidate"
    assert current["session_exclusivity"] == "pending_review"
    assert set(current["protective_deadlines_seconds"].values()) == {None}
    assert (
        current["pre_prediction_policy"]["status"] == "selected_for_review_not_adopted"
    )
