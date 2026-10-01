"""Rehashed invalid histories cannot obtain even a sampled order witness."""

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import refresh_accounting
from stopped_study_timeline_fixtures import (
    make_timeline,
    refresh_timeline,
    verify_timeline,
)
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("launch.json", "automatic_retry", True),
        ("launch.json", "workload_cutoff_seconds", 300),
        ("launch.json", "root_pid", 888),
        ("launch.json", "supervisor_pid", 902),
        ("launch.json", "caffeinate_pid", True),
        ("launch.json", "cwd", "/other/root"),
        ("launch.json", "arguments", ["/invented/python"]),
        ("launch.json", "profile_sha256", "0" * 64),
        ("launch.json", "envelope_sha256", "0" * 64),
        ("launch.json", "supervisor_files_sha256", {}),
        ("launch.json", "launched_at", "2026-01-01T00:00:31+00:00"),
        ("pre.json", "caffeinate_pid", 900),
        ("pre.json", "battery", "Now drawing from 'Battery Power'\n"),
        ("pre.json", "assertions", ""),
        ("post.json", "root_exit_code", 0),
        ("post.json", "root_exit_code", 130.0),
        ("post.json", "root_pid", 888),
        ("post.json", "session_violation", "observed_competing_workload"),
        ("post.json", "supervisor_error_type", "KeyboardInterrupt"),
        ("post.json", "ended_at", "2026-01-01T00:00:01+00:00"),
        ("sleep-cleanup.json", "exit_code", 0),
        ("sleep-cleanup.json", "caffeinate_pid", 901),
        ("sleep-cleanup.json", "assertions", "pid 902(caffeinate): PreventSystemSleep"),
    ],
)
def test_changed_identity_or_interruption_rejected(
    prepared, manifests, record, key, value
):
    case = make_timeline(prepared, manifests)
    case.values[record][key] = value
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("observed_at", "2026-01-01T00:00:02+00:00"),
        ("observed_at", "2026-01-01T00:00:30"),
        ("battery", "Now drawing from 'Battery Power'\n"),
        ("battery", "unknown"),
        ("power_settings", "Battery Power:\n powermode 0\n"),
        ("power_settings", "AC Power:\n powermode 1\n"),
        ("power_settings", "AC Power:\n powermode 0\n lowpowermode 1\n"),
        ("assertions", "pid 901(caffeinate): PreventSystemSleep"),
        ("processes", "invalid table"),
        ("competing_known_workload_pids", [999]),
        ("competing_known_workload_pids", [True]),
        ("interference_scope", "all interference excluded"),
    ],
)
def test_bad_clean_witness_sample_rejected(prepared, manifests, key, value):
    case = make_timeline(prepared, manifests)
    case.values["conditions.jsonl"][1][key] = value
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


@pytest.mark.parametrize(
    "name", ["pre.json", "launch.json", "post.json", "sleep-cleanup.json"]
)
def test_unknown_record_fields_rejected(prepared, manifests, name):
    case = make_timeline(prepared, manifests)
    case.values[name]["invented_extension"] = True
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_all_samples_retained_after_first_violation(prepared, manifests):
    case = make_timeline(prepared, manifests)
    tail = case.values["conditions.jsonl"][-1].copy()
    tail.update(observed_at="2026-01-01T00:00:48+00:00", battery="unknown")
    case.values["conditions.jsonl"].append(tail)
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_unpinned_changes_rejected_even_with_valid_semantics(prepared, manifests):
    case = make_timeline(prepared, manifests)
    previous = case.pins.copy()
    case.values["conditions.jsonl"][1]["thermal"] = "different retained thermal output"
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case, expected_observation_sha256=previous)


def test_next_child_only_after_violation_is_not_witness(prepared, manifests):
    case = make_timeline(prepared, manifests)
    samples = case.values["conditions.jsonl"]
    samples[3]["processes"] = samples[1]["processes"]
    for sample in samples[1:3]:
        sample["processes"] = samples[0]["processes"]
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_witness_requires_later_clean_capture_before_absence(prepared, manifests):
    case = make_timeline(prepared, manifests)
    case.values["conditions.jsonl"].pop(2)
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_earlier_violation_not_hidden_by_later_clean_witness(prepared, manifests):
    case = make_timeline(prepared, manifests)
    case.values["conditions.jsonl"][0]["competing_known_workload_pids"] = [999]
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_reused_historical_child_pid_cannot_identify_next_child(prepared, manifests):
    case = make_timeline(prepared, manifests)
    entries = case.root.accounting["authorization_ledger"]["admissions"]
    previous_pid = entries[2]["launched_pid"]
    entries[-1]["launched_pid"] = previous_pid
    refresh_accounting(case.root)
    sample = case.values["conditions.jsonl"][1]
    sample["processes"] = sample["processes"].replace("903 ", f"{previous_pid} ")
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)
