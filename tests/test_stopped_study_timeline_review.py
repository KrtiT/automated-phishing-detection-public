"""Independent temporal review using invented supervisor and root histories."""

from dataclasses import replace

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


def _change_supervisor_parent(sample, supervisor, parent):
    lines = sample["processes"].splitlines()
    sample["processes"] = (
        "\n".join(
            f"{supervisor} {parent} 0.0 0.1 python"
            if line.split()[0] == str(supervisor)
            else line
            for line in lines
        )
        + "\n"
    )


@pytest.mark.parametrize("parent_kind", ["root", "caffeinate", "next_service"])
def test_owned_process_ancestry_cannot_cycle(prepared, manifests, parent_kind):
    case = make_timeline(prepared, manifests)
    launch = case.values["launch.json"]
    next_entry = case.root.accounting["authorization_ledger"]["admissions"][-1]
    parent = (
        next_entry["launched_pid"]
        if parent_kind == "next_service"
        else launch[f"{parent_kind}_pid"]
    )
    _change_supervisor_parent(
        case.values["conditions.jsonl"][1], launch["supervisor_pid"], parent
    )
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_supervisor_pid_cannot_be_next_service_witness(prepared, manifests):
    case = make_timeline(prepared, manifests)
    launch = case.values["launch.json"]
    next_entry = case.root.accounting["authorization_ledger"]["admissions"][-1]
    previous_pid = next_entry["launched_pid"]
    next_entry["launched_pid"] = launch["supervisor_pid"]
    refresh_accounting(case.root)
    sample = case.values["conditions.jsonl"][1]
    sample["processes"] = (
        "\n".join(
            line
            for line in sample["processes"].splitlines()
            if line.split()[0] != str(previous_pid)
        )
        + "\n"
    )
    _change_supervisor_parent(sample, launch["supervisor_pid"], launch["root_pid"])
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "list"])
def test_entire_five_member_timeline_is_mandatory(prepared, manifests, mutation):
    case = make_timeline(prepared, manifests)
    if mutation == "missing":
        case.payloads = case.payloads[:-1]
    elif mutation == "extra":
        case.payloads += (("substitute.json", b"{}\n"),)
    elif mutation == "duplicate":
        case.payloads += (case.payloads[0],)
    else:
        case.payloads = list(case.payloads)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


@pytest.mark.parametrize("inventory", ["observation", "supervisor"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong", "bytes"])
def test_independent_pin_inventories_are_closed(
    prepared, manifests, inventory, mutation
):
    case = make_timeline(prepared, manifests)
    pins = dict(case.pins if inventory == "observation" else case.hashes)
    name = next(iter(pins))
    if mutation == "missing":
        pins.pop(name)
    elif mutation == "extra":
        pins["unreviewed-member"] = "0" * 64
    else:
        pins[name] = b"0" * 64 if mutation == "bytes" else "0" * 64
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case, **{f"expected_{inventory}_sha256": pins})


def test_timeline_reauthenticates_original_failed_root(prepared, manifests):
    case = make_timeline(prepared, manifests)
    case.root.snapshot = replace(case.root.snapshot, reservation_sha256="0" * 64)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


@pytest.mark.parametrize("source", ["initial", "pre", "early_condition"])
def test_all_prior_known_competition_blocks_later_witness(prepared, manifests, source):
    case = make_timeline(prepared, manifests)
    selected = {
        "initial": case.values["pre.json"]["initial"],
        "pre": case.values["pre.json"],
        "early_condition": case.values["conditions.jsonl"][0],
    }[source]
    selected["competing_known_workload_pids"] = [765432]
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_cleanup_cannot_precede_recorded_root_exit(prepared, manifests):
    case = make_timeline(prepared, manifests)
    case.values["sleep-cleanup.json"]["recorded_at"] = "2026-01-01T00:00:44+00:00"
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)


def test_maximum_sample_gap_includes_late_post_observation(prepared, manifests):
    case = make_timeline(prepared, manifests)
    case.values["post.json"].update(
        observed_at="2026-01-01T00:10:50+00:00",
        ended_at="2026-01-01T00:10:51+00:00",
    )
    case.values["sleep-cleanup.json"]["recorded_at"] = "2026-01-01T00:10:52+00:00"
    refresh_timeline(case)
    result = verify_timeline(case)
    assert result.maximum_sample_start_gap_seconds == 605
    assert result.root_exit_recorded_at == "2026-01-01T00:10:51+00:00"


def test_unadmitted_preflight_child_is_not_next_service_witness(prepared, manifests):
    case = make_timeline(prepared, manifests)
    sample = case.values["conditions.jsonl"][1]
    lines = sample["processes"].splitlines()
    lines[-1] = " ".join(lines[-1].split()[:4]) + " /usr/bin/git"
    sample["processes"] = "\n".join(lines) + "\n"
    refresh_timeline(case)
    with pytest.raises(ValueError, match="invalid_stopped_study_timeline"):
        verify_timeline(case)
