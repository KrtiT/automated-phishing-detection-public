"""Series-only admissions and invalid inputs fail before the target launch."""

import asyncio
import os
import subprocess

import pytest
from study_admission_fixtures import frame as legacy_frame
from study_series_admission_fixtures import api, frame
from study_series_process_fixtures import (
    admitted_options,
    assert_closed,
    assert_resources_closed,
    issuer,
    module,
    observe,
    progress,
    resources,
)
from test_operational_process import inputs
from test_operational_process_writer import pair_options

from automated_phishing_detection import operational_process as original
from automated_phishing_detection._study_admission_parent import ParentAdmission


def _invalid_call(reserved, options, **changes):
    values = (
        pair_options(options)
        | {
            "writer": original._record,
            "series_admissions": issuer({}),
        }
        | changes
    )
    selected = values.pop("attempt", reserved)
    return asyncio.run(module().observe_series_operational_children(selected, **values))


def _forbid_launch(monkeypatch):
    calls = []

    def forbidden(*arguments, **keywords):
        calls.append(arguments)
        raise AssertionError("invalid input reached Popen")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return calls


@pytest.mark.parametrize(
    "name,value",
    (
        ("attempt", None),
        ("attempt", {}),
        ("service_command", ()),
        ("service_command", ["python"]),
        ("service_command", ("",)),
        ("service_command", ("bad\0",)),
        ("client_command", ()),
        ("client_command", (False,)),
        ("client_command", ["python"]),
        ("client_command", ("bad\0",)),
        ("writer", None),
        ("writer", False),
        ("writer", b"writer"),
        ("series_admissions", None),
        ("series_admissions", False),
        ("series_admissions", b"issuer"),
    ),
)
def test_invalid_input_does_not_claim_or_launch(tmp_path, monkeypatch, name, value):
    attempt, options = inputs(tmp_path)
    calls = _forbid_launch(monkeypatch)
    with pytest.raises(original.OperationalProcessError):
        _invalid_call(attempt, options, **{name: value})
    assert not calls
    assert os.listdir(attempt.directory) == ["reservation.json"]


@pytest.mark.parametrize("name", ("startup", "shutdown", "terminate", "kill"))
@pytest.mark.parametrize("value", (True, 0, -1, float("inf"), float("nan"), "5"))
def test_each_invalid_deadline_fails_before_claim(tmp_path, monkeypatch, name, value):
    attempt, options = inputs(tmp_path)
    deadlines = pair_options(options)["deadlines"] | {name: value}
    calls = _forbid_launch(monkeypatch)
    with pytest.raises(original.OperationalProcessError, match="protective_deadline"):
        _invalid_call(attempt, options, deadlines=deadlines)
    assert not calls
    assert os.listdir(attempt.directory) == ["reservation.json"]


@pytest.mark.parametrize("kind", ("missing", "extra", "list", "none"))
def test_deadline_inventory_is_exact_before_claim(tmp_path, monkeypatch, kind):
    attempt, options = inputs(tmp_path)
    deadlines = pair_options(options)["deadlines"]
    variants = {
        "missing": {name: value for name, value in deadlines.items() if name != "kill"},
        "extra": deadlines | {"request": 2.0},
        "list": list(deadlines.items()),
        "none": None,
    }
    calls = _forbid_launch(monkeypatch)
    with pytest.raises(original.OperationalProcessError):
        _invalid_call(attempt, options, deadlines=variants[kind])
    assert not calls
    assert os.listdir(attempt.directory) == ["reservation.json"]


def _mismatch(role, command, kind):
    if kind == "legacy":
        return ParentAdmission(legacy_frame(command, role))
    selected_role = "client" if role == "service" else "service"
    selected = frame(
        command if kind != "command" else ("wrong",),
        selected_role if kind == "role" else role,
    )
    admission = api().SeriesParentAdmission(selected)
    if kind == "protocol":
        admission.frame = legacy_frame(command, role)
    elif kind == "parent":
        admission.frame = frame(command, role, parent_pid=os.getppid())
    return admission


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize("kind", ("legacy", "protocol", "role", "command", "parent"))
def test_mismatched_admission_rejects_before_target_popen(
    tmp_path, monkeypatch, role, kind
):
    attempt, options = inputs(tmp_path)
    admissions, launched, captured = {}, [], resources(monkeypatch)
    popen = subprocess.Popen

    def launch(command, **keywords):
        launched.append(command)
        return popen(command, **keywords)

    def issue(actual_role, command):
        admission = (
            _mismatch(actual_role, command, kind)
            if actual_role == role
            else api().SeriesParentAdmission(frame(command, actual_role))
        )
        admissions[actual_role] = admission
        return admission

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issue)
    assert launched == ([] if role == "service" else [options["service_command"]])
    observed = progress(caught.value)
    assert observed[role]["pid"] is observed[role]["exit_code"] is None
    assert admissions[role].entered is False and admissions[role].read_fd is None
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


def test_installed_intent_cannot_launch_another_series_pair(tmp_path, monkeypatch):
    attempt, options = inputs(tmp_path)
    observe(attempt, admitted_options(options), issuer({}))
    retained = {path.name: path.read_bytes() for path in attempt.directory.iterdir()}
    calls = _forbid_launch(monkeypatch)
    with pytest.raises(original.OperationalProcessError):
        observe(attempt, options, issuer({}))
    assert not calls
    assert retained == {
        path.name: path.read_bytes() for path in attempt.directory.iterdir()
    }
