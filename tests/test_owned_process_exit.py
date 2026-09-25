import asyncio
import importlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest


def _module() -> ModuleType:
    name = "automated_phishing_detection._owned_process_exit"
    assert importlib.util.find_spec(name) is not None, "missing owned exit probe"
    return importlib.import_module(name)


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("status,code", [(0, 0), (17 << 8, 17), (15, -15)])
def test_completed_status_sets_only_observed_returncode(
    monkeypatch: pytest.MonkeyPatch, block: bool, status: int, code: int
) -> None:
    module = _module()
    calls = []
    process = SimpleNamespace(pid=123, returncode=99)

    def waitpid(pid: int, flags: int) -> tuple[int, int]:
        calls.append((pid, flags))
        return pid, status

    monkeypatch.setattr(module.os, "waitpid", waitpid)
    observed = module.observe_owned_exit(process, block=block)
    assert observed == module.OwnedProcessExit(123, True, code)
    assert process.returncode == code
    assert calls == [(123, 0 if block else os.WNOHANG)]
    with pytest.raises(FrozenInstanceError):
        observed.exit_code = 0


def test_pending_does_not_trust_or_change_cached_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    process = SimpleNamespace(pid=123, returncode=0)
    monkeypatch.setattr(module.os, "waitpid", lambda *_: (0, 0))
    assert module.observe_owned_exit(process) is None
    assert process.returncode == 0


@pytest.mark.parametrize("cached", [None, 0, 17])
def test_external_reaping_is_terminal_unknown(
    monkeypatch: pytest.MonkeyPatch, cached: int | None
) -> None:
    module = _module()
    process = SimpleNamespace(pid=123, returncode=cached)

    def waitpid(pid: int, flags: int) -> tuple[int, int]:
        raise ChildProcessError("private process diagnostic")

    monkeypatch.setattr(module.os, "waitpid", waitpid)
    observed = module.observe_owned_exit(process)
    assert observed == module.OwnedProcessExit(123, False, None)
    assert process.returncode == cached
    assert "private" not in repr(observed)


@pytest.mark.parametrize("pid", [True, None, 0, -1, "123"])
def test_invalid_requested_pid_never_waits(
    monkeypatch: pytest.MonkeyPatch, pid: object
) -> None:
    module = _module()

    def forbidden(*args: object) -> None:
        pytest.fail("invalid request reached waitpid")

    monkeypatch.setattr(module.os, "waitpid", forbidden)
    with pytest.raises(module.OwnedProcessExitError, match="^invalid_process_pid$"):
        module.observe_owned_exit(SimpleNamespace(pid=pid, returncode=None))


@pytest.mark.parametrize("block", [0, 1, None, "private"])
def test_invalid_block_flag_never_waits(
    monkeypatch: pytest.MonkeyPatch, block: object
) -> None:
    module = _module()

    def forbidden(*args: object) -> None:
        pytest.fail("invalid request reached waitpid")

    monkeypatch.setattr(module.os, "waitpid", forbidden)
    with pytest.raises(module.OwnedProcessExitError, match="^invalid_wait_mode$"):
        module.observe_owned_exit(SimpleNamespace(pid=123), block=block)


@pytest.mark.parametrize(
    "result,block",
    [
        (None, False),
        ([123, 0], False),
        ((123,), False),
        ((True, 0), False),
        ((124, 0), False),
        ((0, 1), False),
        ((0, 0), True),
        ((123, True), False),
        ((123, -1), False),
        ((123, 10**100), False),
        ((123, 0x137F), False),
        ((123, 0xFFFF), False),
    ],
)
def test_invalid_wait_result_cannot_manufacture_completion(
    monkeypatch: pytest.MonkeyPatch, result: object, block: bool
) -> None:
    module = _module()
    process = SimpleNamespace(pid=123, returncode=None)
    monkeypatch.setattr(module.os, "waitpid", lambda *_: result)
    with pytest.raises(module.OwnedProcessExitError, match="^invalid_wait_result$"):
        module.observe_owned_exit(process, block=block)
    assert process.returncode is None


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt(), SystemExit(9), asyncio.CancelledError(), OSError()]
)
def test_wait_failures_propagate_original_without_mutation(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    module = _module()
    process = SimpleNamespace(pid=123, returncode=None)

    def waitpid(pid: int, flags: int) -> tuple[int, int]:
        raise failure

    monkeypatch.setattr(module.os, "waitpid", waitpid)
    with pytest.raises(type(failure)) as caught:
        module.observe_owned_exit(process, block=True)
    assert caught.value is failure
    assert process.returncode is None


@pytest.mark.parametrize(
    "program,code",
    [
        ("raise SystemExit(0)", 0),
        ("raise SystemExit(17)", 17),
        ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", -15),
    ],
)
def test_real_owned_exit_and_external_reaping(program: str, code: int) -> None:
    module = _module()
    with subprocess.Popen([sys.executable, "-c", program]) as process:
        observed = module.observe_owned_exit(process, block=True)
        assert observed == module.OwnedProcessExit(process.pid, True, code)
        assert process.returncode == code
    with subprocess.Popen([sys.executable, "-c", program]) as process:
        waited_pid, status = os.waitpid(process.pid, 0)
        assert waited_pid == process.pid
        assert os.waitstatus_to_exitcode(status) == code
        assert module.observe_owned_exit(process) == module.OwnedProcessExit(
            process.pid, False, None
        )
        assert process.returncode is None


def test_real_pending_child_then_owned_completion() -> None:
    module = _module()
    command = [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            assert module.observe_owned_exit(process) is None
            assert process.returncode is None
        finally:
            process.stdin.close()
        assert module.observe_owned_exit(process, block=True) == (
            module.OwnedProcessExit(process.pid, True, 0)
        )
