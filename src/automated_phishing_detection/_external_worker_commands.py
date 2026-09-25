"""Exact original and retained-source worker commands, without access authority."""

import sys
from dataclasses import fields

from ._external_source_records import ExternalRunPaths, ExternalSourceExecutionError
from ._prepared_external_records import PreparedExternalRunPaths
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths


def _artifact_options(paths):
    groups = (
        (paths.artifacts, ArtifactPaths),
        (paths.secondary_artifacts, SecondaryArtifactPaths),
        (paths.drift_artifacts, DriftArtifactPaths),
    )
    if any(type(group) is not kind for group, kind in groups):
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    return tuple(
        (member.name.replace("_", "-"), getattr(group, member.name))
        for group, unused in groups
        for member in fields(group)
    )


def _worker_options(paths):
    if type(paths) is not ExternalRunPaths:
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    return (
        ("archive", paths.archive),
        ("suffix-rules", paths.suffix_rules),
        *_artifact_options(paths),
        ("attempt", paths.attempt),
        ("public-summary", paths.public_summary),
    )


def _command(binding, script, options, transport):
    options = (
        ("repo-root", binding.root),
        ("expected-revision", binding.revision),
        ("expected-contract-sha256", binding.contract_sha256),
        *options,
        ("internal-transport", transport.directory),
        ("expected-handoff-sha256", transport.expected_handoff_sha256),
    )
    return (
        sys.executable,
        str(binding.root / "scripts" / script),
        *(
            argument
            for name, value in options
            for argument in (f"--{name}", str(value))
        ),
    )


def _worker_command(binding, paths, transport):
    return _command(
        binding, "run_external_evaluation.py", _worker_options(paths), transport
    )


def _prepared_worker_command(binding, paths, transport, preparation):
    if type(paths) is not PreparedExternalRunPaths:
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    options = (
        ("preparation", paths.preparation),
        *_artifact_options(paths),
        ("attempt", paths.attempt),
        ("public-summary", paths.public_summary),
        ("expected-preparation-reservation-sha256", preparation.reservation_sha256),
        ("expected-preparation-completion-sha256", preparation.completion_sha256),
    )
    return _command(binding, "run_prepared_external_evaluation.py", options, transport)
