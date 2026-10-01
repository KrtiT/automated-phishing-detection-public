"""The physical supervisor has only closed identity options and no bypass."""

import importlib.util
import sys
from pathlib import Path

import pytest
from study_series_session_fixtures import api
from test_study_series_runner_cli import arguments


def cli():
    path = Path(__file__).parents[1] / "scripts/run_study_series_session.py"
    assert path.is_file(), "missing fixed series physical-session CLI"
    spec = importlib.util.spec_from_file_location("series_session_test_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "extra",
    [
        ["--cell", "73"],
        ["--timeout", "900"],
        ["--command", "arbitrary"],
        ["--run"],
        ["--expected-profile-sha", "b" * 64],
    ],
)
def test_no_workload_or_interference_overrides(extra):
    with pytest.raises(SystemExit):
        cli()._parser().parse_args(arguments() + extra)


def test_fixed_root_command_preserves_interpreter_spelling(monkeypatch):
    module = api("study_series_session")
    selected = cli()._parser().parse_args(arguments())
    interpreter = "/private/invented/../environment/bin/python"
    monkeypatch.setattr(sys, "executable", interpreter)
    assert module.root_command(selected) == [
        interpreter,
        "scripts/run_study_series.py",
        *arguments(),
    ]


def test_candidate_policy_rejects_before_physical_access(monkeypatch):
    module = api("study_series_session")
    monkeypatch.setattr(
        module, "bind_series_public_execution", lambda *args, **kwargs: object()
    )

    def reject(public):
        raise ValueError("candidate_closed")

    def forbidden(*args, **kwargs):
        pytest.fail("physical IO or process launch preceded final-policy check")

    monkeypatch.setattr(module, "require_final_policy", reject)
    monkeypatch.setattr(module.host, "capture", forbidden)
    monkeypatch.setattr(module, "hold_records", forbidden)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="candidate_closed"):
        module.run_series_session(cli()._parser().parse_args(arguments()))
