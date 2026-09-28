"""Project the original thirty reviewed path arguments without filesystem access."""

from dataclasses import fields

from ._prepared_external_records import PreparedExternalRunPaths
from ._prepared_internal_records import PreparedInternalRunPaths
from ._study_execution_schema import lexical_path
from ._study_preparation_records import StudyPreparationPaths
from ._study_run_context import StudyRunPaths, _paths
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths


def _group(mapping, kind):
    return kind(*(mapping[member.name.replace("_", "-")] for member in fields(kind)))


def _source_paths(mapping):
    primary = _group(mapping, ArtifactPaths)
    secondary = _group(mapping, SecondaryArtifactPaths)
    drift = _group(mapping, DriftArtifactPaths)
    preparation = StudyPreparationPaths(
        *(
            mapping[name]
            for name in ("source-csv", "suffix-rules", "archive", "preparation-attempt")
        )
    )
    internal = PreparedInternalRunPaths(
        preparation.attempt,
        primary,
        secondary,
        mapping["internal-attempt"],
        mapping["internal-public-summary"],
    )
    external = PreparedExternalRunPaths(
        preparation.attempt,
        primary,
        secondary,
        drift,
        mapping["external-attempt"],
        mapping["external-public-summary"],
    )
    return preparation, internal, external


def project_paths(base, value):
    mapping = {name: lexical_path(raw) for name, raw in value.items()}
    result = StudyRunPaths(
        *_source_paths(mapping),
        *(
            mapping[name]
            for name in (
                "attempt",
                "public-summary",
                "accepted-inputs-dir",
                "cells-dir",
            )
        ),
    )
    _paths(base, result)
    return result
