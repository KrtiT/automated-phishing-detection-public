import asyncio
from dataclasses import fields
from pathlib import Path

import pytest
from study_cli_fixtures import NAMES, SCRIPT, arguments, cli, result

from automated_phishing_detection import study_runner
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths

__all__ = ["cli"]


def test_whole_study_entry_exists():
    assert SCRIPT.is_file(), "missing whole-study CLI"


def test_exact_required_option_order_and_shared_protocol(cli):
    from automated_phishing_detection import _study_cli_protocol as protocol

    actions = cli.parser()._actions[1:]
    assert (
        tuple(action.option_strings[0] for action in actions)
        == tuple(f"--{name}" for name in NAMES)
        == protocol.ARGUMENTS
    )
    assert all(action.required for action in actions)
    assert protocol.SCRIPT == "scripts/run_study.py"


@pytest.mark.parametrize("name", NAMES)
def test_every_option_is_required(cli, name):
    values = arguments()
    position = values.index(f"--{name}")
    del values[position : position + 2]
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(values)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "name",
    [
        "worker",
        "resume",
        "override",
        "deadline",
        "startup",
        "ready",
        "allow-access",
        "command",
        "subset",
        "run-index",
        "expected-preparation-completion-sha256",
        "expected-operational-profile",
        "expected-archive-sha256",
    ],
)
def test_no_execution_override_or_abbreviated_option(cli, name):
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(arguments() + [f"--{name}", "invented"])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "status,message",
    [
        ("whole_study_hold", "Study held: insufficient population capacity."),
        ("study_evidence_published", "Study evidence published."),
    ],
)
def test_once_public_async_dispatch_and_exact_typed_paths(
    cli, monkeypatch, capsys, status, message
):
    calls = []

    async def run(root, **keywords):
        assert asyncio.get_running_loop().is_running()
        calls.append((root, keywords))
        return result(status)

    monkeypatch.setattr(study_runner, "run_study", run)
    assert cli.main(arguments()) == 0
    assert len(calls) == 1
    root, keywords = calls[0]
    assert root == Path("invented/repo-root")
    assert set(keywords) == {
        "expected_revision",
        "expected_contract_sha256",
        "expected_operational_profile_sha256",
        "paths",
    }
    for name in tuple(keywords)[:3]:
        assert keywords[name] == f"invented/{name.replace('_', '-')}"
    check_paths(keywords["paths"])
    captured = capsys.readouterr()
    assert captured.out == message + "\n" and captured.err == ""


def check_paths(paths):
    assert type(paths) is study_runner.StudyRunPaths
    assert (
        paths.internal.preparation
        == paths.external.preparation
        == paths.preparation.attempt
    )
    assert paths.preparation.source_csv == Path("invented/source-csv")
    assert paths.preparation.suffix_rules == Path("invented/suffix-rules")
    assert paths.preparation.archive == Path("invented/archive")
    assert paths.preparation.attempt == Path("invented/preparation-attempt")
    assert paths.internal.artifacts is paths.external.artifacts
    assert paths.internal.secondary_artifacts is paths.external.secondary_artifacts
    for name in (
        "attempt",
        "public_summary",
        "accepted_inputs_directory",
        "cells_directory",
    ):
        option = name.replace("_directory", "_dir").replace("_", "-")
        assert getattr(paths, name) == Path(f"invented/{option}")
    for role in ("internal", "external"):
        for name in ("attempt", "public_summary"):
            assert getattr(getattr(paths, role), name) == Path(
                f"invented/{role}-{name.replace('_', '-')}"
            )
    check_artifacts(paths)


def check_artifacts(paths):
    for group, expected in (
        (paths.internal.artifacts, ArtifactPaths),
        (paths.internal.secondary_artifacts, SecondaryArtifactPaths),
        (paths.external.drift_artifacts, DriftArtifactPaths),
    ):
        assert type(group) is expected
        for member in fields(expected):
            assert getattr(group, member.name) == Path(
                f"invented/{member.name.replace('_', '-')}"
            )
