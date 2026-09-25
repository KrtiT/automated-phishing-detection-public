"""CLI failure privacy and the unchanged closed access gate."""

import asyncio
from pathlib import Path

import pytest
from test_external_source_cli import arguments
from test_external_source_cli import cli as cli

from automated_phishing_detection.execution_preflight import ExecutionBinding


class PrivateFatal(BaseException):
    def __str__(self):
        raise AssertionError("private exception formatter must not run")


class PrivateExit(SystemExit):
    def __getattribute__(self, name):
        if name == "code":
            raise ValueError("private-code-property-canary")
        return super().__getattribute__(name)


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
def test_failures_are_symbolic_and_never_successful(
    cli, monkeypatch, capsys, failure, expected_code, reason
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "run_external_evaluation", fail)
    assert cli.main(arguments()) == expected_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"External execution stopped: {reason}\n"
    assert calls == [True]


@pytest.mark.parametrize(
    "archive", ["\x00unopened", "../unopened", "~/unopened", "unopened\npath"]
)
def test_closed_binding_stops_before_any_supplied_path_access(
    cli, monkeypatch, capsys, archive
):
    from automated_phishing_detection import external_source_runner, source_runner

    binding = ExecutionBinding(Path("invented root"), "a" * 40, "b" * 64, (), "{}")
    monkeypatch.setattr(
        external_source_runner, "bind_execution", lambda *args, **kwargs: binding
    )

    def forbidden(*args, **kwargs):
        pytest.fail("closed CLI inspected a supplied path")

    values = arguments()
    values[values.index("--archive") + 1] = archive
    with monkeypatch.context() as guard:
        for name in ("read_bytes", "read_text", "open", "stat", "lstat", "resolve"):
            guard.setattr(Path, name, forbidden)
        guard.setattr(source_runner, "_read_file_once", forbidden)
        assert cli.main(values) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "External execution stopped: execution_failed\n"
