import copy
import json
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from automated_phishing_detection.transport_neutral_features import (
    extract_transport_neutral_features,
)


def module():
    name = "automated_phishing_detection.transport_neutral_model"
    assert find_spec(name) is not None, "follow-up model implementation is missing"
    return import_module(name)


def rows(count):
    urls = [
        f"https://safe{index}.example/"
        if index % 2 == 0
        else f"http://verify-account-{index}.example/login?password=11111111111111"
        for index in range(count)
    ]
    return urls, np.asarray([index % 2 for index in range(count)], dtype=np.int8)


@pytest.fixture(scope="module")
def fitted():
    return module().fit_model(*rows(200), *rows(1000))


def test_training_only_scaling_and_separate_artifact_identity(fitted):
    expected = np.asarray(
        [extract_transport_neutral_features(url) for url in rows(200)[0]]
    )
    assert np.allclose(fitted["scaler"]["mean"], expected.mean(axis=0), atol=1e-12)
    assert fitted["scaler"]["n_samples_seen"] == 200
    assert fitted["artifact_type"] == "transport-neutral-followup-model-v1"
    assert fitted["representation_id"] == "transport-neutral-structural-v1"
    assert len(fitted["features"]) == 24
    assert fitted["validation_threshold"]["status"] == "selected"
    assert fitted["validation_threshold"]["fpr_upper_95"] <= 0.01


def test_serialized_model_has_exact_scheme_invariance_and_repeatability(fitted):
    payload = json.loads(json.dumps(fitted, allow_nan=False))
    urls = [
        "http://new.example/login?password=1",
        "https://new.example/login?password=1",
    ]
    scores, audit = module().score_urls(payload, urls)
    repeated, _ = module().score_urls(payload, urls)
    assert scores[0] == scores[1]
    assert scores == repeated
    assert audit["max_absolute_probability_difference"] <= 1e-12


@pytest.mark.parametrize(
    "field", ["features", "representation_id", "software_versions"]
)
def test_incompatible_model_cannot_be_scored(fitted, field):
    payload = copy.deepcopy(fitted)
    payload[field] = "incorrect"
    with pytest.raises(ValueError):
        module().score_urls(payload, ["https://safe.example"])


@pytest.mark.parametrize("replacement", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scaler_cannot_silently_emit_scores(fitted, replacement):
    payload = copy.deepcopy(fitted)
    payload["scaler"]["scale"][0] = replacement
    with pytest.raises(ValueError):
        module().score_urls(payload, ["https://safe.example"])


def test_convergence_warning_is_fatal_and_no_retry_occurs(monkeypatch):
    extension = module()
    calls = []

    def fail_fit(*args, **kwargs):
        calls.append(1)
        raise ConvergenceWarning("synthetic convergence failure")

    monkeypatch.setattr(extension.LogisticRegression, "fit", fail_fit)
    with pytest.raises(ConvergenceWarning):
        extension.fit_model(*rows(20), *rows(1000))
    assert calls == [1]


@pytest.mark.parametrize("labels", [[0, 0], [0, 2], [False, True], [0.0, 1.0]])
def test_invalid_or_single_class_training_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        module().fit_model(rows(2)[0], labels, *rows(1000))
