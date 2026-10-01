"""Real service handles and selected file descriptors survive no interruption."""

import os
import signal
from contextlib import nullcontext
from dataclasses import replace

import pytest
from operational_input_signal_fixtures import assert_closed, watch_open
from operational_service_child_fixtures import inherited
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_operational_fixtures import hold, setup

from automated_phishing_detection import _study_series_child_transport as transport

__all__ = ["candidates", "child_case", "manifests", "series_case"]


def service_context(case):
    case.arguments.role = "service"
    case.held.admission.frame = replace(case.held.admission.frame, role="service")
    case.held.environment.update(
        {
            name: value
            for name, value in os.environ.items()
            if name
            in {"APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD", "APD_BASE_URL"}
        }
    )


@pytest.mark.parametrize("role", ("client", "service"))
def test_real_signal_during_restore_closes_selected_files_and_service_controls(
    child_case, series_case, tmp_path, monkeypatch, role
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    completed = []

    def interrupted(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        completed.append(True)

    monkeypatch.setattr(transport, "restore_series_child_inputs", interrupted)
    with inherited(monkeypatch) if role == "service" else nullcontext() as controls:
        if controls is not None:
            service_context(case)
        descriptors = watch_open(monkeypatch, "accepted-inputs.json")
        with pytest.raises(KeyboardInterrupt):
            with hold(case):
                pytest.fail("interrupted inputs yielded")
        assert_closed(descriptors + ([] if controls is None else list(controls[3])))
    assert completed == []


def test_real_service_handles_close_after_valid_computational_context(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    with inherited(monkeypatch) as controls:
        service_context(case)
        with hold(case) as runtime:
            assert runtime.role_context.base_url == controls[0]
            assert runtime.handles[0].fileno() == controls[3][0]
            assert runtime.handles[1:] == controls[3][1:]
            runtime.retain("service-role.json", b"invented service")
        assert_closed(controls[3])


def test_input_acquisition_failure_progress_survives_final_public_recheck(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    first, later = ValueError("restoration"), KeyboardInterrupt("final public check")
    first.progress = b"invented failed restoration"

    def restore(*args, **kwargs):
        raise first

    def recheck(held):
        case.events.append("check")
        if len(case.events) == 2:
            raise later

    monkeypatch.setattr(transport, "restore_series_child_inputs", restore)
    monkeypatch.setattr(case.module, "recheck_held_child", recheck)
    with pytest.raises(KeyboardInterrupt) as caught:
        with hold(case):
            pytest.fail("restoration failure yielded")
    assert caught.value is later and later.progress == first.progress
