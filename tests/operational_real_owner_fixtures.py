"""Genuine invented artifacts load once inside the real bound owner lifecycle."""

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from operational_runtime_fixtures import binding, inputs

from automated_phishing_detection import bound_models, bound_runtime
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.selective_inference import SelectiveCascade


def expected_primary(paths, summaries):
    baseline = summaries["baseline"]["models"]
    hashes = summaries["transformer"]["artifact_hashes"] | {
        "length-only.json": baseline["length-only"]["artifact_sha256"],
        "logistic-l1.json": baseline["Logistic-L1"]["artifact_sha256"],
        "gmm.json": summaries["gmm"]["artifact_hashes"]["gmm.json"],
    }
    metadata = json.loads((paths.transformer_bundle / "transformer.json").read_bytes())
    cascade = json.loads((paths.transformer_bundle / "cascade.json").read_bytes())
    return {
        "artifact_hashes": hashes,
        "thresholds": {
            "length_only": baseline["length-only"]["validation_threshold"]["threshold"],
            "logistic_l1": baseline["Logistic-L1"]["validation_threshold"]["threshold"],
            "transformer": metadata["validation_threshold"]["threshold"],
            "half_width": cascade["calibration"]["half_width"],
            "monitor_boundary": summaries["gmm"]["threshold"],
        },
    }


def real_case(fixture, monkeypatch, ordinal):
    root, paths, summaries, loads = fixture
    cell = inputs(ordinal)
    accepted = json.loads(cell.accepted_bytes)
    accepted["primary"] = expected_primary(paths, summaries)
    case = SimpleNamespace(
        binding=replace(binding(), root=root),
        paths=paths,
        inputs=replace(cell, accepted_bytes=canonical_bytes(accepted)),
        loads=loads,
        events=[],
        retained=[],
        session=None,
        forwards=[],
    )
    monkeypatch.setattr(bound_runtime, "recheck_binding", lambda unused: None)
    monkeypatch.setattr(
        bound_runtime, "load_bound_models", lambda root, paths: load(case, root, paths)
    )
    monkeypatch.setattr(
        bound_runtime, "SelectiveCascade", lambda model: owner(case, model)
    )
    return case


def load(case, root, paths):
    case.events.append(("load", threading.get_ident()))
    models = bound_models.load_bound_models(root, paths)
    original = models.cascade._model.forward

    def forward(*args, **kwargs):
        assert len(case.retained) == 1
        case.forwards.append(threading.get_ident())
        return original(*args, **kwargs)

    models.cascade._model.forward = forward
    return models


@contextmanager
def owner(case, model):
    with SelectiveCascade(model, _fixture_cpu=True) as scorer:
        case.events.append(("enter", threading.get_ident()))
        case.session = scorer
        try:
            yield scorer
        finally:
            case.events.append(("exit", threading.get_ident()))


def retain(case, name, content):
    assert case.session.counts.completed_requests == 0
    assert case.session.counts.transformer_forward_attempts == 0
    case.events.append(("role", threading.get_ident()))
    case.retained.append((name, content))
