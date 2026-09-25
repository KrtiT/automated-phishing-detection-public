"""Compose external comparators before numerical ownership, without access authority."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .bound_drift import BoundDrift, DriftArtifactPaths, load_bound_drift
from .bound_models import ArtifactPaths, load_bound_models
from .bound_runtime import BoundEvaluationSession, BoundSession
from .bound_secondary import SecondaryArtifactPaths, load_bound_secondary
from .execution_preflight import ExecutionBinding, recheck_binding
from .selective_inference import SelectiveCascade


@dataclass(frozen=True)
class BoundExternalSession:
    evaluation: BoundEvaluationSession
    drift: BoundDrift


@contextmanager
def open_bound_external_session(
    binding: ExecutionBinding,
    primary_paths: ArtifactPaths,
    secondary_paths: SecondaryArtifactPaths,
    drift_paths: DriftArtifactPaths,
) -> Iterator[BoundExternalSession]:
    """Bind all external comparators before taking numerical ownership."""
    recheck_binding(binding)
    models = load_bound_models(binding.root, primary_paths)
    secondary = load_bound_secondary(binding.root, secondary_paths, models.cascade)
    drift = load_bound_drift(binding, drift_paths, models)
    recheck_binding(binding)
    try:
        with SelectiveCascade(models.cascade) as scorer:
            evaluation = BoundEvaluationSession(BoundSession(models, scorer), secondary)
            yield BoundExternalSession(evaluation, drift)
    finally:
        recheck_binding(binding)
