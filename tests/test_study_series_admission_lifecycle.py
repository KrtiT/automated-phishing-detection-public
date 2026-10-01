"""One-shot launch/cleanup and separate legacy validators remain closed."""

import os
import subprocess

import pytest
from study_admission_fixtures import frame as original_frame
from study_series_admission_fixtures import api, child_command, command, frame

from automated_phishing_detection._study_admission_parent import (
    ParentAdmission,
    validate_launch_admission,
)


def test_legacy_and_series_launch_validators_do_not_accept_each_other():
    modern = api().SeriesParentAdmission(frame())
    original = ParentAdmission(original_frame(role="service"))
    api().validate_series_launch(modern, "service", command())
    with pytest.raises(ValueError):
        validate_launch_admission(modern, ("service",), command())
    with pytest.raises(ValueError):
        api().validate_series_launch(original, "service", command())


@pytest.mark.parametrize("mismatch", ["role", "command"])
def test_actual_child_rejects_mismatched_role_or_command(mismatch):
    arguments = child_command()
    selected = frame(
        arguments if mismatch == "role" else ("wrong",),
        role="client" if mismatch == "role" else "service",
    )
    with api().SeriesParentAdmission(selected) as admission:
        process = subprocess.Popen(
            arguments,
            env=os.environ | admission.environment,
            pass_fds=(admission.read_fd,),
            stderr=subprocess.DEVNULL,
        )
        admission.launched(process.pid)
        assert process.wait(timeout=10) != 0


def test_partial_preload_closes_all_descriptors(monkeypatch):
    descriptors = []
    original = os.pipe

    def pipe():
        created = original()
        descriptors.extend(created)
        return created

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "write", lambda descriptor, content: len(content) - 1)
    with pytest.raises(ValueError):
        with api().SeriesParentAdmission(frame()):
            pytest.fail("partial frame permitted launch")
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_parent_callbacks_are_single_observation_and_launch():
    observed = []
    with api().SeriesParentAdmission(
        frame(), on_observed=lambda known, code: observed.append((known, code))
    ) as admission:
        admission.launched(os.getpid() + 100000)
        with pytest.raises(ValueError):
            admission.launched(os.getpid() + 100001)
        admission.observed(False, None)
        admission.observed(True, 0)
    assert observed == [(False, None)]
    assert admission.exit_observed is False and admission.exit_code is None
    with pytest.raises(ValueError):
        admission.__enter__()


@pytest.mark.parametrize(
    "known,code", [(0, None), (True, False), (False, 0), (True, None)]
)
def test_malformed_observation_rejected(known, code):
    with api().SeriesParentAdmission(frame()) as admission:
        admission.launched(os.getpid() + 100000)
        with pytest.raises(ValueError):
            admission.observed(known, code)


def test_prelaunch_observation_rejected():
    with api().SeriesParentAdmission(frame()) as admission:
        with pytest.raises(ValueError):
            admission.observed(True, 0)


def test_closed_parent_cannot_record_a_late_launch():
    with api().SeriesParentAdmission(frame()) as admission:
        pass
    with pytest.raises(ValueError):
        admission.launched(os.getpid() + 100000)
