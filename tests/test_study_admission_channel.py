import os
import subprocess

import pytest
from study_admission_fixtures import child_command, frame, module


def parent(frame_value, **callbacks):
    api = module()
    assert hasattr(api, "ParentAdmission"), "missing parent-owned admission channel"
    return api.ParentAdmission(frame_value, **callbacks)


@pytest.mark.parametrize("role", ("internal", "external", "service", "client"))
def test_real_child_consumes_exact_role_command_and_live_parent(role):
    arguments = child_command(role)
    events = []
    admission = parent(
        frame(arguments, role),
        on_launched=lambda pid: events.append(("launched", pid)),
        on_observed=lambda known, code: events.append(("observed", known, code)),
    )
    with admission:
        descriptor = admission.read_fd
        process = subprocess.Popen(
            arguments,
            env={**os.environ, **admission.environment},
            pass_fds=(descriptor,),
        )
        admission.launched(process.pid)
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert process.wait(timeout=10) == 0
        admission.observed(True, process.returncode)
    assert events == [("launched", process.pid), ("observed", True, 0)]


def test_child_observes_writer_loss_without_inherited_writer():
    suffix = (
        "print('consumed', flush=True)\n"
        "sys.stdin.readline()\n"
        "try:\n    admission.check()\n"
        "except ValueError:\n    raise SystemExit(13)\n"
        "raise SystemExit(99)\n"
    )
    arguments = child_command(suffix=suffix)
    with parent(frame(arguments)) as admission:
        process = subprocess.Popen(
            arguments,
            env={**os.environ, **admission.environment},
            pass_fds=(admission.read_fd,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        admission.launched(process.pid)
        assert process.stdout.readline() == "consumed\n"
    process.communicate("continue\n", timeout=10)
    assert process.returncode == 13


@pytest.mark.parametrize("mismatch", ("role", "command", "parent"))
def test_real_child_rejects_wrong_role_argv_or_parent(mismatch):
    arguments = child_command()
    changes = {"parent_pid": os.getppid()} if mismatch == "parent" else {}
    bound = frame(
        arguments if mismatch != "command" else ("wrong",),
        "external" if mismatch == "role" else "internal",
        **changes,
    )
    if mismatch == "parent":
        with pytest.raises(module().StudyAdmissionError):
            with parent(bound):
                pytest.fail("wrong parent must never launch")
        return
    with parent(bound) as admission:
        process = subprocess.Popen(
            arguments,
            env={**os.environ, **admission.environment},
            pass_fds=(admission.read_fd,),
            stderr=subprocess.DEVNULL,
        )
        admission.launched(process.pid)
        assert process.wait(timeout=10) != 0


def test_short_nonblocking_preload_closes_both_descriptors(monkeypatch):
    descriptors = []
    original = os.pipe

    def pipe():
        created = original()
        descriptors.extend(created)
        return created

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "write", lambda descriptor, content: len(content) - 1)
    with pytest.raises(module().StudyAdmissionError):
        with parent(frame()):
            pytest.fail("short preload must not launch")
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
