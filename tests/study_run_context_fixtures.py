"""Pure invented study paths, never private source or model acquisition."""

from dataclasses import fields
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from automated_phishing_detection._prepared_external_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)
from automated_phishing_detection._study_preparation_records import (
    StudyPreparationPaths,
)
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths


def api():
    name = "automated_phishing_detection._study_run_context"
    assert find_spec(name), "missing same-parent study context"
    return import_module(name)


def model_paths(private):
    artifacts = ArtifactPaths(*(private / item.name for item in fields(ArtifactPaths)))
    secondary = SecondaryArtifactPaths(
        *(private / item.name for item in fields(SecondaryArtifactPaths))
    )
    drift = DriftArtifactPaths(
        *(private / item.name for item in fields(DriftArtifactPaths))
    )
    return artifacts, secondary, drift


def internal_paths(preparation, artifacts, secondary, outputs):
    return PreparedInternalRunPaths(
        preparation,
        artifacts,
        secondary,
        outputs / "internal",
        outputs / "internal.json",
    )


def paths(module):
    private = Path("/invented/private")
    artifacts, secondary, drift = model_paths(private)
    outputs = Path("/invented/output")
    preparation = outputs / "preparation"
    return module.StudyRunPaths(
        StudyPreparationPaths(
            private / "original.csv",
            private / "suffix.dat",
            private / "archive.zip",
            preparation,
        ),
        internal_paths(preparation, artifacts, secondary, outputs),
        PreparedExternalRunPaths(
            preparation,
            artifacts,
            secondary,
            drift,
            outputs / "external",
            outputs / "external.json",
        ),
        outputs / "root",
        outputs / "root.json",
        outputs / "accepted",
        outputs / "cells",
    )


def deadlines():
    return dict(startup=10.0, shutdown=10.0, terminate=2.0, kill=2.0)
