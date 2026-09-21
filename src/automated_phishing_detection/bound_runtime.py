"""Compose reviewed bindings and accepted artifacts on one inference owner.

These entry points do not authorize protected-data access or publish measurements.
The experiment producer must still enforce its frozen input and attempt rules.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from automated_phishing_detection.bound_models import (
    ArtifactPaths,
    BoundModels,
    load_bound_models,
)
from automated_phishing_detection.execution_preflight import (
    ExecutionBinding,
    recheck_binding,
)
from automated_phishing_detection.live_monitor import LiveMonitor
from automated_phishing_detection.selective_inference import SelectiveCascade
from automated_phishing_detection.selective_service import create_app
from automated_phishing_detection.shift_schema import ShiftPlan
from automated_phishing_detection.shift_service import create_shift_app


@dataclass(frozen=True)
class BoundSession:
    models: BoundModels
    scorer: SelectiveCascade


@contextmanager
def open_bound_session(
    binding: ExecutionBinding, paths: ArtifactPaths
) -> Iterator[BoundSession]:
    """Check identity before loading, before inference, and after restoration."""
    recheck_binding(binding)
    models = load_bound_models(binding.root, paths)
    recheck_binding(binding)
    try:
        with SelectiveCascade(models.cascade) as scorer:
            yield BoundSession(models, scorer)
    finally:
        recheck_binding(binding)


def make_bound_cascade_factory(binding: ExecutionBinding, paths: ArtifactPaths):
    """Defer all artifact loading and numerical ownership until factory entry."""

    @contextmanager
    def factory():
        with open_bound_session(binding, paths) as session:
            yield session.scorer

    return factory


def create_bound_app(
    binding: ExecutionBinding,
    paths: ArtifactPaths,
    *,
    workload: str = "fixed_cascade",
):
    return create_app(make_bound_cascade_factory(binding, paths), workload=workload)


def create_bound_shift_app(
    binding: ExecutionBinding, paths: ArtifactPaths, plan: ShiftPlan
):
    @contextmanager
    def factory():
        with open_bound_session(binding, paths) as session:
            yield LiveMonitor(
                session.scorer,
                stage1_model=session.models.cascade.stage1_model,
                gmm=session.models.gmm,
                boundary=session.models.monitor_boundary,
            )

    return create_shift_app(factory, plan)
