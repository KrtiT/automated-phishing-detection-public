import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest


def _module():
    name = "automated_phishing_detection.owned_worker"
    assert importlib.util.find_spec(name) is not None, "missing owned worker"
    return importlib.import_module(name)


def _command(program):
    return sys.executable, "-c", program


@pytest.mark.parametrize(
    "program,code",
    [
        ("raise SystemExit(0)", 0),
        ("raise SystemExit(17)", 17),
        ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", -15),
    ],
)
def test_real_worker_binds_actual_exit_and_command(program, code):
    module = _module()
    command = _command(program)
    observed = module.observe_worker(command)
    content = json.dumps(list(command), ensure_ascii=True, separators=(",", ":"))
    assert observed.command_sha256 == sha256(content.encode("ascii")).hexdigest()
    assert observed.exit.exit_observed is True
    assert observed.exit.exit_code == code
    assert observed.stdout_sha256 == sha256(b"").hexdigest()
    assert observed.stderr_sha256 == sha256(b"").hexdigest()
    with pytest.raises(ChildProcessError):
        os.waitpid(observed.exit.pid, os.WNOHANG)
    with pytest.raises(FrozenInstanceError):
        observed.command_sha256 = "replacement"


def test_large_private_output_cannot_deadlock_or_leak(capsys):
    module = _module()
    program = (
        "import sys; "
        "sys.stdout.buffer.write(b'private-output' * 100000); "
        "sys.stderr.buffer.write(b'private-error' * 100000); "
        "assert sys.stdin.buffer.read() == b''"
    )
    observed = module.observe_worker(_command(program))
    assert observed.exit.exit_code == 0
    assert observed.stdout_sha256 == sha256(b"private-output" * 100000).hexdigest()
    assert observed.stderr_sha256 == sha256(b"private-error" * 100000).hexdigest()
    assert "private" not in repr(observed)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("code", [0, 17])
def test_external_reaping_is_unknown_even_with_cached_zero(monkeypatch, code):
    module = _module()
    original = subprocess.Popen
    spawned = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        os.waitpid(process.pid, 0)
        process.returncode = 0
        return process

    def forbidden(*args, **kwargs):
        pytest.fail("lost ownership must never signal or use a Popen reaper")

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module.os, "kill", forbidden)
    for name in ("poll", "wait", "communicate", "__enter__", "__exit__"):
        monkeypatch.setattr(original, name, forbidden)
    observed = module.observe_worker(_command(f"raise SystemExit({code})"))
    assert len(spawned) == 1
    assert observed.exit.pid == spawned[0].pid
    assert observed.exit.exit_observed is False
    assert observed.exit.exit_code is None


def test_cached_zero_never_substitutes_for_live_status(monkeypatch):
    module = _module()
    original = subprocess.Popen
    launched = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        process.returncode = 0
        launched.append((args, kwargs))
        return process

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    observed = module.observe_worker(_command("raise SystemExit(17)"))
    assert observed.exit.exit_code == 17
    assert len(launched) == 1
    positional, keywords = launched[0]
    assert positional == (_command("raise SystemExit(17)"),)
    assert keywords["stdin"] == subprocess.DEVNULL
    assert keywords.get("shell", False) is False
    assert keywords["close_fds"] is True
    assert keywords["stdout"].closed and keywords["stderr"].closed


def test_terminal_status_is_not_waited_twice(monkeypatch):
    module = _module()
    original = module.observe_owned_exit
    observations = []

    def observe(process, *, block=False):
        observations.append((process.pid, block))
        return original(process, block=block)

    monkeypatch.setattr(module, "observe_owned_exit", observe)
    observed = module.observe_worker(_command("raise SystemExit(0)"))
    assert observations == [(observed.exit.pid, True)]
