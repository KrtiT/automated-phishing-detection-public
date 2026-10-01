"""Independent adversarial checks of historical authority and stopped admissions."""

import builtins
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import (
    change_frame,
    make_stopped,
    refresh_accounting,
    verify,
)
from study_run_record_fixtures import prepared

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


def _forbidden(*args, **kwargs):
    pytest.fail("historical verification cannot observe processes or access files")


def _guard_observations(monkeypatch):
    for name in (
        "_operational_process_records",
        "_external_process_context",
        "prepared_external_process",
        "study_runner",
        "operational_cell_runner",
        "internal_process_handoff",
        "external_source_process",
        "external_source_handoff",
        "stopped_study_authorization",
    ):
        importlib.import_module(f"automated_phishing_detection.{name}")
    guarded = set()
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("automated_phishing_detection."):
            continue
        for name, value in tuple(vars(module).items()):
            if not isinstance(value, type):
                continue
            if value.__name__ == "ProcessObservation" or value.__name__.startswith(
                "Observed"
            ):
                guarded.add(value.__name__)
                monkeypatch.setattr(module, name, _forbidden)
    assert len(guarded) == 9


def test_raw_history_cannot_create_observations_or_access_files(
    prepared, manifests, monkeypatch
):
    case = make_stopped(prepared, manifests, stopped_admissions=2)
    _guard_observations(monkeypatch)
    monkeypatch.setattr(builtins, "open", _forbidden)
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr(os, "open", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    assert verify(case).accepted_ordinals == (1,)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("parent", [0, -1, True, 987654])
def test_every_admission_requires_the_same_positive_parent(
    prepared, manifests, position, parent
):
    case = make_stopped(prepared, manifests, stopped_admissions=2)
    change_frame(case, position, parent_pid=parent)
    with pytest.raises(ValueError, match="^invalid_stopped_study_authorization$"):
        verify(case)


@pytest.mark.parametrize("position", [4, 5])
@pytest.mark.parametrize(
    "field",
    [
        "profile_sha256",
        "envelope_sha256",
        "root_reservation_sha256",
        "intent_sha256",
        "barrier_sha256",
        "preparation_reservation_sha256",
        "preparation_completion_sha256",
        "predecessor_sha256",
        "accepted_inputs_sha256",
        "cell_binding_sha256",
    ],
)
def test_unaccepted_tail_still_requires_original_admission_joins(
    prepared, manifests, position, field
):
    case = make_stopped(prepared, manifests, stopped_admissions=2)
    change_frame(case, position, **{field: "0" * 64})
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "states",
    [
        ((None, False, None),),
        ((12345, False, None),),
        ((12345, True, -15),),
        ((12345, True, 0),),
        ((12345, False, None), (None, False, None)),
        ((12345, False, None), (23456, False, None)),
        ((12345, True, -15), (23456, True, -9)),
        ((12345, True, 0), (23456, True, 0)),
    ],
)
def test_stopped_launch_and_exit_states_are_preserved(prepared, manifests, states):
    case = make_stopped(prepared, manifests, stopped_admissions=len(states))
    entries = case.accounting["authorization_ledger"]["admissions"][4:]
    for entry, (pid, observed, code) in zip(entries, states, strict=True):
        entry.update(launched_pid=pid, exit_observed=observed, exit_code=code)
    refresh_accounting(case)
    retained = canonical_bytes(case.accounting)
    result = verify(case)
    assert result.accounting_bytes == retained
    assert canonical_bytes(case.accounting) == retained
    assert result.accepted_ordinals == (1,) and result.stopped_ordinal == 2


@pytest.mark.parametrize(
    "state",
    [(None, True, 0), (12345, False, 0), (12345, True, None), (True, True, 0)],
)
def test_stopped_tail_cannot_invent_launch_or_exit(prepared, manifests, state):
    case = make_stopped(prepared, manifests, stopped_admissions=1)
    entry = case.accounting["authorization_ledger"]["admissions"][4]
    pid, observed, code = state
    entry.update(launched_pid=pid, exit_observed=observed, exit_code=code)
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


def test_stopped_client_requires_launched_service(prepared, manifests):
    case = make_stopped(prepared, manifests, stopped_admissions=2)
    entry = case.accounting["authorization_ledger"]["admissions"][4]
    entry.update(launched_pid=None, exit_observed=False, exit_code=None)
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)
