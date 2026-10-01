"""The series executable accepts only its fixed independently pinned command."""

import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).parents[1] / "scripts/run_study_series_child.py"
    assert path.is_file(), "missing fixed series child executable"
    spec = importlib.util.spec_from_file_location("series_child_test_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    return [
        "--role",
        "client",
        "--repo-root",
        "/invented/repo",
        "--expected-revision",
        "a" * 40,
        "--expected-profile-sha256",
        "b" * 64,
        "--envelope",
        "/invented/envelope.json",
        "--expected-envelope-sha256",
        "c" * 64,
        "--cell-ordinal",
        "73",
        "--expected-binding-sha256",
        "d" * 64,
    ]


def test_exact_fixed_flags_parse_without_io():
    result = api()._parser().parse_args(arguments())
    assert vars(result) == dict(
        role="client",
        repo_root=Path("/invented/repo"),
        expected_revision="a" * 40,
        expected_profile_sha256="b" * 64,
        envelope=Path("/invented/envelope.json"),
        expected_envelope_sha256="c" * 64,
        cell_ordinal=73,
        expected_binding_sha256="d" * 64,
    )


@pytest.mark.parametrize(
    "extra",
    (
        ["--workload", "shift_period"],
        ["--timeout", "999"],
        ["--gmm", "/invented/model"],
        ["--expected-profile-sha", "b" * 64],
    ),
)
def test_no_workload_model_deadline_or_abbreviation_override(extra):
    with pytest.raises(SystemExit) as caught:
        api()._parser().parse_args(arguments() + extra)
    assert caught.value.code == 2


@pytest.mark.parametrize("role", ("internal", "external", "parent"))
def test_no_additional_roles(role):
    command = arguments()
    command[1] = role
    with pytest.raises(SystemExit):
        api()._parser().parse_args(command)


def test_main_runs_exact_dispatch_and_uses_symbolic_failure(monkeypatch, capsys):
    from automated_phishing_detection import _study_series_child_runtime as runtime

    async def rejected(arguments):
        assert arguments.cell_ordinal == 73
        raise ValueError("private path and content must stay hidden")

    monkeypatch.setattr(runtime, "run_child", rejected)
    assert api().main(arguments()) == 2
    captured = capsys.readouterr()
    assert "private path" not in captured.err
    assert "Study series child stopped" in captured.err


def test_success_requires_parent_verification_message(monkeypatch, capsys):
    from automated_phishing_detection import _study_series_child_runtime as runtime

    async def complete(arguments):
        return None

    monkeypatch.setattr(runtime, "run_child", complete)
    assert api().main(arguments()) == 0
    assert "parent verification required" in capsys.readouterr().out
