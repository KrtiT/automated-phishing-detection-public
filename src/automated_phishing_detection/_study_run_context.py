"""Pure whole-study context; matching paths or profiles grant no data access."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

from . import _operational_profile as profiles
from ._checkpoint_codec import canonical_bytes
from ._prepared_external_records import PreparedExternalRunPaths
from ._prepared_internal_records import PreparedInternalRunPaths
from ._study_preparation_records import StudyPreparationPaths
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import ExecutionBinding


class StudyRunError(ValueError):
    """Symbolic study rejection without private paths or parser diagnostics."""


@dataclass(frozen=True)
class StudyRunPaths:
    preparation: StudyPreparationPaths
    internal: PreparedInternalRunPaths
    external: PreparedExternalRunPaths
    attempt: Path
    public_summary: Path
    accepted_inputs_directory: Path
    cells_directory: Path


def require(condition):
    if not condition:
        raise StudyRunError("invalid_study_context")


def _path(value):
    require(isinstance(value, Path) and value.is_absolute())
    require(".." not in value.parts and "\0" not in str(value))
    return value


def _artifact_paths(group, expected):
    require(type(group) is expected)
    return tuple(_path(getattr(group, member.name)) for member in fields(expected))


def _sources(paths):
    require(type(paths.preparation) is StudyPreparationPaths)
    require(type(paths.internal) is PreparedInternalRunPaths)
    require(type(paths.external) is PreparedExternalRunPaths)
    _path(paths.internal.preparation)
    _path(paths.external.preparation)
    require(
        paths.internal.preparation
        == paths.external.preparation
        == paths.preparation.attempt
    )
    require(paths.internal.artifacts == paths.external.artifacts)
    require(paths.internal.secondary_artifacts == paths.external.secondary_artifacts)
    raw = tuple(
        _path(getattr(paths.preparation, name))
        for name in ("source_csv", "suffix_rules", "archive")
    )
    require(len(set(raw)) == 3)
    models = (
        *_artifact_paths(paths.internal.artifacts, ArtifactPaths),
        *_artifact_paths(paths.external.artifacts, ArtifactPaths),
        *_artifact_paths(paths.internal.secondary_artifacts, SecondaryArtifactPaths),
        *_artifact_paths(paths.external.secondary_artifacts, SecondaryArtifactPaths),
        *_artifact_paths(paths.external.drift_artifacts, DriftArtifactPaths),
    )
    return raw, models


def output_paths(paths):
    return tuple(
        _path(value)
        for value in (
            paths.preparation.attempt,
            paths.internal.attempt,
            paths.internal.public_summary,
            paths.external.attempt,
            paths.external.public_summary,
            paths.attempt,
            paths.public_summary,
            paths.accepted_inputs_directory,
            paths.cells_directory,
        )
    )


def _disjoint(first, second):
    return not (first.is_relative_to(second) or second.is_relative_to(first))


def _paths(binding, paths):
    require(type(paths) is StudyRunPaths)
    sources, models = _sources(paths)
    outputs = output_paths(paths)
    checkout = _path(binding.root)
    require(all(_disjoint(model, checkout) for model in models))
    for position, output in enumerate(outputs):
        require(_disjoint(output, checkout))
        require(all(_disjoint(output, source) for source in (*sources, *models)))
        require(all(_disjoint(output, prior) for prior in outputs[:position]))


def _deadlines(deadlines):
    require(type(deadlines) is dict)
    require(set(deadlines) == {"startup", "shutdown", "terminate", "kill"})
    for value in deadlines.values():
        require(type(value) in (int, float) and math.isfinite(value) and value > 0)


def validate_context(binding, profile, paths, deadlines):
    """Reject pure inconsistencies before reserving or inspecting any private path."""
    try:
        require(type(binding) is ExecutionBinding)
        require(type(profile) is profiles.CandidateOperationalProfile)
        require(type(profile.canonical_bytes) is bytes)
        require(
            profile.canonical_bytes == canonical_bytes(profiles._projection(binding))
        )
        _deadlines(deadlines)
        _paths(binding, paths)
    except Exception:
        raise StudyRunError("invalid_study_context") from None
