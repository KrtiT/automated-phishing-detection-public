"""Invented owner-thread sessions exercise the real bound-runtime lifecycle."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

from operational_runtime_fixtures import binding, inputs, primary
from test_selective_service import SyntheticScorer

from automated_phishing_detection import bound_runtime
from automated_phishing_detection.bound_models import ArtifactPaths


def models():
    expected = primary()
    thresholds = expected["thresholds"]
    return SimpleNamespace(
        artifact_hashes=tuple(sorted(expected["artifact_hashes"].items())),
        length_only=SimpleNamespace(
            validation_threshold_record={
                "status": "selected",
                "threshold": thresholds["length_only"],
            }
        ),
        cascade=SimpleNamespace(
            stage1_threshold=thresholds["logistic_l1"],
            transformer_threshold=thresholds["transformer"],
            half_width=thresholds["half_width"],
            stage1_model=object(),
        ),
        gmm={},
        monitor_boundary=thresholds["monitor_boundary"],
    )


def owner_case(monkeypatch, ordinal=1):
    case = SimpleNamespace(
        binding=binding(),
        inputs=inputs(ordinal),
        models=models(),
        events=[],
        scorer=None,
        retained=[],
        active=False,
    )
    root = case.binding.root
    case.paths = ArtifactPaths(
        *(root / name for name in ("length", "logistic", "transformer", "gmm"))
    )

    def checked(value):
        assert value is case.binding and not case.active
        case.events.append(("check", threading.get_ident()))

    def loaded(root, paths):
        assert root == case.binding.root and paths is case.paths
        case.events.append(("load", threading.get_ident()))
        return case.models

    monkeypatch.setattr(bound_runtime, "recheck_binding", checked)
    monkeypatch.setattr(bound_runtime, "load_bound_models", loaded)
    monkeypatch.setattr(
        bound_runtime, "SelectiveCascade", lambda model: scorer_owner(case, model)
    )
    return case


@contextmanager
def scorer_owner(case, loaded):
    assert loaded is case.models.cascade
    case.scorer = SyntheticScorer()
    case.active = True
    with case.scorer:
        try:
            yield case.scorer
        finally:
            case.active = False


def retain(case, name, content):
    assert case.active and case.scorer.urls == []
    case.events.append(("role", threading.get_ident()))
    case.retained.append((name, content))
