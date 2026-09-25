"""The executable boundary never exposes private exceptions or false success."""

import asyncio
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_internal_evaluation.py"
    spec = importlib.util.spec_from_file_location("internal_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(cli, worker):
    arguments = ["--worker"] if worker else []
    for action in cli.parser()._actions:
        if action.required:
            arguments.extend((action.option_strings[0], "invented-unused"))
    return arguments


class PrivateFatal(BaseException):
    def __str__(self):
        raise AssertionError("private exception formatter must not run")


class PrivateExit(SystemExit):
    def __getattribute__(self, name):
        if name == "code":
            raise ValueError("private-code-property-canary")
        return super().__getattribute__(name)


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize(
    "failure,expected_code,reason",
    [
        (ValueError("private-canary"), 2, "execution_failed"),
        (asyncio.CancelledError("private-canary"), 130, "cancelled"),
        (KeyboardInterrupt("private-canary"), 130, "keyboard_interrupt"),
        (SystemExit(17), 17, "system_exit"),
        (SystemExit(None), 2, "system_exit"),
        (SystemExit(0), 2, "system_exit"),
        (SystemExit(256), 2, "system_exit"),
        (SystemExit(-256), 2, "system_exit"),
        (SystemExit(True), 2, "system_exit"),
        (SystemExit("private-canary"), 2, "system_exit"),
        (PrivateFatal("private-canary"), 2, "interrupted"),
        (PrivateExit(17), 2, "system_exit"),
    ],
)
def test_cli_failures_are_symbolic_nonzero(
    cli, monkeypatch, capsys, worker, failure, expected_code, reason
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise failure

    name = "run_internal_evaluation" if worker else "run_internal_process"
    monkeypatch.setattr(cli, name, fail)
    assert cli.main(_arguments(cli, worker)) == expected_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"Internal execution stopped: {reason}\n"
    assert calls == [True]


@pytest.mark.parametrize("worker", [False, True])
def test_cli_success_message_follows_one_return(cli, monkeypatch, capsys, worker):
    calls = []

    def succeed(*args, **kwargs):
        calls.append(True)

    name = "run_internal_evaluation" if worker else "run_internal_process"
    monkeypatch.setattr(cli, name, succeed)
    assert cli.main(_arguments(cli, worker)) == 0
    captured = capsys.readouterr()
    expected = (
        "Internal evidence published." if worker else "Internal completion verified."
    )
    assert captured.out == expected + "\n"
    assert captured.err == ""
    assert calls == [True]
