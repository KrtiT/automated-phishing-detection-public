"""Real synthetic children establish live pipe behavior, not study authority."""

import os
import subprocess

import pytest
from study_series_admission_fixtures import api, child_command, frame


@pytest.mark.parametrize("role", ["service", "client"])
def test_actual_child_consumes_matching_live_parent_channel(role):
    command = child_command(role)
    events = []
    admission = api().SeriesParentAdmission(
        frame(command, role),
        on_launched=lambda pid: events.append(("launched", pid)),
        on_observed=lambda known, code: events.append(("observed", known, code)),
    )
    with admission:
        descriptor = admission.read_fd
        process = subprocess.Popen(
            command, env=os.environ | admission.environment, pass_fds=(descriptor,)
        )
        admission.launched(process.pid)
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert process.wait(timeout=10) == 0
        admission.observed(True, process.returncode)
    assert events == [("launched", process.pid), ("observed", True, 0)]


def test_parent_cannot_claim_its_own_pid_as_a_launched_child():
    with api().SeriesParentAdmission(frame()) as admission:
        with pytest.raises(ValueError):
            admission.launched(os.getpid())


def test_parent_must_be_actual_current_process():
    with pytest.raises(ValueError):
        with api().SeriesParentAdmission(frame(parent_pid=os.getppid())):
            pytest.fail("foreign parent acquired a channel")


def test_writer_loss_is_detected_by_actual_child():
    suffix = (
        "print('consumed', flush=True)\nsys.stdin.readline()\n"
        "try:\n    admission.check()\n"
        "except ValueError:\n    raise SystemExit(13)\n"
        "raise SystemExit(99)\n"
    )
    command = child_command(suffix=suffix)
    with api().SeriesParentAdmission(frame(command)) as admission:
        process = subprocess.Popen(
            command,
            env=os.environ | admission.environment,
            pass_fds=(admission.read_fd,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        admission.launched(process.pid)
        assert process.stdout.readline() == "consumed\n"
    process.communicate("continue\n", timeout=10)
    assert process.returncode == 13
