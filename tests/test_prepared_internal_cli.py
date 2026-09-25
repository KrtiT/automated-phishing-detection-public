"""The fixed retained worker command cannot carry original input paths."""

import importlib.util
from dataclasses import fields
from pathlib import Path

import pytest
from test_source_runner import inputs, runner

from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths

__all__ = ["inputs", "runner"]
ROOT = Path(__file__).resolve().parents[1]


def module():
    name = "automated_phishing_detection._prepared_internal_process"
    assert importlib.util.find_spec(name), "missing prepared internal process"
    return importlib.import_module(name)


def command(inputs):
    binding, original, unused_session, unused_events = inputs
    paths = PreparedInternalRunPaths(
        original.attempt.parent / "preparation",
        original.artifacts,
        original.secondary_artifacts,
        original.attempt,
        original.public_summary,
    )
    return module()._worker_command(
        binding, paths, reservation_sha256="a" * 64, completion_sha256="b" * 64
    )


def script():
    filename = ROOT / "scripts/run_prepared_internal_evaluation.py"
    assert filename.is_file(), "missing prepared-only worker script"
    spec = importlib.util.spec_from_file_location("prepared_internal_cli", filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_worker_command_has_closed_preparation_and_complete_model_arguments(inputs):
    invocation = command(inputs)
    assert Path(invocation[1]).name == "run_prepared_internal_evaluation.py"
    assert "--source-csv" not in invocation and "--suffix-rules" not in invocation
    expected = {
        "--worker",
        "--repo-root",
        "--expected-revision",
        "--expected-contract-sha256",
        "--preparation",
        "--expected-preparation-reservation-sha256",
        "--expected-preparation-completion-sha256",
        "--attempt",
        "--public-summary",
        *(
            "--" + member.name.replace("_", "-")
            for group in (ArtifactPaths, SecondaryArtifactPaths)
            for member in fields(group)
        ),
    }
    assert {value for value in invocation if value.startswith("--")} == expected
    args = script().parser().parse_args(invocation[2:])
    assert args.expected_preparation_reservation_sha256 == "a" * 64
    assert args.expected_preparation_completion_sha256 == "b" * 64


@pytest.mark.parametrize("flag", ["--source-csv", "--suffix-rules", "--archive"])
def test_prepared_script_rejects_original_source_arguments(inputs, flag):
    with pytest.raises(SystemExit) as caught:
        script().parser().parse_args([*command(inputs)[2:], flag, "forbidden"])
    assert caught.value.code == 2


@pytest.mark.parametrize("digest", [None, True, "A" * 64, "x" * 64])
def test_worker_command_rejects_noncanonical_expected_digest(inputs, digest):
    binding, original, unused_session, unused_events = inputs
    paths = PreparedInternalRunPaths(
        original.attempt.parent,
        original.artifacts,
        original.secondary_artifacts,
        original.attempt,
        original.public_summary,
    )
    with pytest.raises(ValueError):
        module()._worker_command(
            binding, paths, reservation_sha256=digest, completion_sha256="b" * 64
        )
