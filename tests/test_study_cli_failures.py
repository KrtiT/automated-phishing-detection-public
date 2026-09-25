import asyncio
from types import SimpleNamespace

import pytest
from study_cli_fixtures import arguments, cli, result

from automated_phishing_detection import study_runner

__all__ = ["cli"]


@pytest.mark.parametrize(
    "error,kind,code",
    [
        (RuntimeError("private secret"), "execution_failed", 2),
        (KeyboardInterrupt("private secret"), "keyboard_interrupt", 130),
        (asyncio.CancelledError("private secret"), "cancelled", 130),
        (SystemExit(0), "system_exit", 2),
        (SystemExit(17), "system_exit", 17),
        (SystemExit(True), "system_exit", 2),
        (SystemExit(256), "system_exit", 2),
        (SystemExit("private secret"), "system_exit", 2),
        (BaseException("private secret"), "interrupted", 2),
    ],
)
def test_symbolic_failure_and_nonzero_normalization(
    cli, monkeypatch, capsys, error, kind, code
):
    async def fail(*arguments, **keywords):
        raise error

    monkeypatch.setattr(study_runner, "run_study", fail)
    assert cli.main(arguments()) == code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"Study execution stopped: {kind}\n"
    assert "private secret" not in captured.err


@pytest.mark.parametrize("status", ["accepted", "completed", None, True, []])
def test_unknown_return_status_cannot_print_success(cli, monkeypatch, capsys, status):
    async def run(*arguments, **keywords):
        return result(status)

    monkeypatch.setattr(study_runner, "run_study", run)
    assert cli.main(arguments()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Study execution stopped: execution_failed\n"


def test_closed_binding_blocks_hostile_paths_before_private_context(
    cli, monkeypatch, capsys
):
    calls = []

    def bind(root, **keywords):
        calls.append((root, keywords))
        return SimpleNamespace(protected_evaluation_ready=False)

    def forbidden(*arguments, **keywords):
        pytest.fail("closed CLI reached private context")

    monkeypatch.setattr(study_runner, "bind_execution", bind)
    for name in (
        "_run_bound_study",
        "resolve_external_source_profile",
        "resolve_operational_profile",
        "validate_context",
    ):
        monkeypatch.setattr(study_runner, name, forbidden)
    values = arguments()
    values[values.index("--archive") + 1] = "../opaque\0archive"
    assert cli.main(values) == 2
    assert len(calls) == 1
    assert capsys.readouterr().err == "Study execution stopped: execution_failed\n"
