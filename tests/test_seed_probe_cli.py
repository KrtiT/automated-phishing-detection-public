import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli():
    path = ROOT / "scripts/run_secondary_seed_probes.py"
    assert path.exists(), "missing seed/probe entry point"
    spec = importlib.util.spec_from_file_location("seed_probe_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    return [
        "--repo-root",
        str(ROOT),
        "--expected-revision",
        "a" * 40,
        "--expected-profile-sha256",
        "b" * 64,
    ]


def test_check_accepts_no_data_paths(cli, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_execution",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert cli.main(arguments() + ["--check"]) == 0
    assert len(calls) == 1
    assert "Research inputs were not read" in capsys.readouterr().out


def test_check_rejects_supplied_input_before_binding(cli, monkeypatch):
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_execution",
        lambda *args, **kwargs: pytest.fail("bound despite invalid check arguments"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(arguments() + ["--check", "--validation", "/must-not-open"])
    assert error.value.code == 2


def test_execution_requires_every_named_path(cli):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments())
    assert error.value.code == 2


def test_metadata_failure_is_sanitized(cli, monkeypatch, capsys):
    def stop(*args, **kwargs):
        raise ValueError("private-url-or-path")

    monkeypatch.setattr(cli, "bind_seed_probe_execution", stop)
    assert cli.main(arguments() + ["--check"]) == 2
    assert "private-url-or-path" not in capsys.readouterr().err


def test_saved_output_verification_requires_actual_exit_argument(cli):
    paths = [
        item
        for name in cli.PATH_ARGUMENTS
        for item in ("--" + name.replace("_", "-"), "/unused")
    ]
    with pytest.raises(SystemExit) as error:
        cli.main(arguments() + ["--verify"] + paths)
    assert error.value.code == 2


def test_verification_dispatches_without_running_workers(cli, monkeypatch):
    calls = []
    paths = [
        item
        for name in cli.PATH_ARGUMENTS
        for item in ("--" + name.replace("_", "-"), "/unused")
    ]
    monkeypatch.setattr(
        cli, "verify_seed_probe_run", lambda *args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        cli, "run_seed_probes", lambda *args, **kwargs: pytest.fail("fit from verifier")
    )
    assert (
        cli.main(arguments() + ["--verify", "--producer-exit-code", "0"] + paths) == 0
    )
    assert calls[0]["producer_exit_code"] == 0
