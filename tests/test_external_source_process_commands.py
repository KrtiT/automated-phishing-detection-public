"""One deterministic argument tuple binds worker observation and verification."""

import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_external_source_process import module

from automated_phishing_detection._external_source_records import ExternalRunPaths
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths


def command_case():
    groups = tuple(
        kind(*(Path("/invented") / member.name for member in fields(kind)))
        for kind in (ArtifactPaths, SecondaryArtifactPaths, DriftArtifactPaths)
    )
    paths = ExternalRunPaths(
        Path("/invented/archive"),
        Path("/invented/suffix"),
        *groups,
        Path("/invented/attempt"),
        Path("/invented/public"),
    )
    binding = SimpleNamespace(
        root=Path("/invented/checkout"), revision="a" * 40, contract_sha256="b" * 64
    )
    transport = SimpleNamespace(
        directory=Path("/invented/transport"), expected_handoff_sha256="c" * 64
    )
    return binding, paths, transport


def test_exact_worker_command_is_immutable_deterministic_and_has_no_override():
    api = module()
    binding, paths, transport = command_case()
    command = api._worker_command(binding, paths, transport)
    assert type(command) is tuple
    assert command == api._worker_command(binding, paths, transport)
    assert command[:2] == (
        sys.executable,
        "/invented/checkout/scripts/run_external_evaluation.py",
    )
    options = dict(zip(command[2::2], command[3::2], strict=True))
    _assert_options(options, transport)


def _assert_options(options, transport):
    assert len(options) == 26
    assert tuple(options)[:5] == (
        "--repo-root",
        "--expected-revision",
        "--expected-contract-sha256",
        "--archive",
        "--suffix-rules",
    )
    assert tuple(options)[-4:] == (
        "--attempt",
        "--public-summary",
        "--internal-transport",
        "--expected-handoff-sha256",
    )
    assert options["--internal-transport"] == str(transport.directory)
    assert options["--expected-handoff-sha256"] == transport.expected_handoff_sha256
    assert not {
        "--worker",
        "--profile",
        "--archive-sha256",
        "--producer-exit-code",
        "--override",
    } & set(options)


@pytest.mark.parametrize(
    "field", ["artifacts", "secondary_artifacts", "drift_artifacts"]
)
def test_command_rejects_untyped_artifact_groups(field):
    api = module()
    binding, paths, transport = command_case()
    with pytest.raises(api.ExternalSourceExecutionError):
        api._worker_command(binding, replace(paths, **{field: None}), transport)
