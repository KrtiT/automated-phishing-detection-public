"""Worker-only CLI shape on invented paths without protected input access."""

import importlib.util
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from automated_phishing_detection._external_source_records import ExternalRunPaths
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths

OPTION_NAMES = (
    "repo-root",
    "expected-revision",
    "expected-contract-sha256",
    "archive",
    "suffix-rules",
    "length-only",
    "logistic-l1",
    "transformer-bundle",
    "gmm",
    "formatting",
    "permutation-42",
    "permutation-43",
    "permutation-44",
    "permutation-45",
    "permutation-46",
    "random-forest",
    "seed-43-weights",
    "seed-44-weights",
    "seed-45-weights",
    "seed-46-weights",
    "training-reference",
    "validation-audit",
    "attempt",
    "public-summary",
    "internal-transport",
    "expected-handoff-sha256",
)


def script_path():
    path = Path(__file__).resolve().parents[1] / "scripts/run_external_evaluation.py"
    assert path.is_file(), "missing external worker-only CLI"
    return path


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("external_cli_fixture", script_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    return [
        argument
        for name in OPTION_NAMES
        for argument in (f"--{name}", f"invented private/{name}")
    ]


def test_parser_has_only_exact_required_worker_options(cli):
    actions = cli.parser()._actions
    assert [action.option_strings for action in actions] == [
        ["-h", "--help"],
        *[[f"--{name}"] for name in OPTION_NAMES],
    ]
    assert all(action.required for action in actions[1:])


def _expected_paths():
    groups = [
        group(
            *[
                Path(f"invented private/{field.name.replace('_', '-')}")
                for field in fields(group)
            ]
        )
        for group in (ArtifactPaths, SecondaryArtifactPaths, DriftArtifactPaths)
    ]
    return ExternalRunPaths(
        Path("invented private/archive"),
        Path("invented private/suffix-rules"),
        *groups,
        Path("invented private/attempt"),
        Path("invented private/public-summary"),
    )


def test_success_dispatches_exact_typed_paths_once(cli, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "run_external_evaluation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert cli.main(arguments()) == 0
    assert calls == [
        (
            (Path("invented private/repo-root"),),
            {
                "expected_revision": "invented private/expected-revision",
                "expected_contract_sha256": "invented private/expected-contract-sha256",
                "paths": _expected_paths(),
                "internal_transport": Path("invented private/internal-transport"),
                "expected_handoff_sha256": "invented private/expected-handoff-sha256",
            },
        )
    ]
    captured = capsys.readouterr()
    assert captured.out == "External evidence published.\n"
    assert captured.err == ""


@pytest.mark.parametrize("name", OPTION_NAMES)
def test_every_declared_option_is_required(cli, name):
    values = arguments()
    position = values.index(f"--{name}")
    del values[position : position + 2]
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(values)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "option",
    [
        "--worker",
        "--archive-sha256",
        "--archive-size-bytes",
        "--profile",
        "--profile-sha256",
        "--ready",
        "--allow-protected",
        "--producer-exit-code",
        "--overlap",
        "--internal-handoff",
        "--expected-handoff",
    ],
)
def test_no_override_or_abbreviated_option_is_accepted(cli, option):
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(arguments() + [option, "invented"])
    assert caught.value.code == 2


def test_actual_script_help_is_worker_only_and_does_not_preflight():
    path = script_path()
    environment = dict(os.environ, PYTHONPATH=str(path.parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "--internal-transport" in result.stdout
    assert "--expected-handoff-sha256" in result.stdout
    assert "--worker" not in result.stdout
    assert "verified" not in result.stdout.lower()
