"""Fit-boundary regression checks use invented URLs, never study records."""

import hashlib
import json

import numpy as np
import pytest

from automated_phishing_detection import secondary_tabular as tabular


def inputs():
    labels = tuple(index % 2 for index in range(96))
    urls = tuple(
        f"https://invented-{index}.example/" + "a" * (3 + 30 * label)
        for index, label in enumerate(labels)
    )
    return urls, labels, urls, labels


def options(receipts, seed=42):
    return {
        "seed": seed,
        "training_record_ids": tuple(f"training-{index}" for index in range(96)),
        "fit_input_receipt": receipts.append,
    }


@pytest.mark.parametrize("seed", range(42, 47))
def test_receipt_matches_actual_fit_arrays_and_preserves_original_model(
    monkeypatch, seed
):
    original_fit = tabular.LogisticRegression.fit
    receipts, observed = [], []

    def capture(estimator, features, labels, *args, **kwargs):
        observed.append((features.copy(), labels.copy(), estimator.get_params()))
        assert len(receipts) == 1
        assert not labels.flags.writeable
        assert not features.flags.writeable
        return original_fit(estimator, features, labels, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tabular.LogisticRegression, "fit", capture)
        result = tabular.fit_label_permutation(*inputs(), **options(receipts, seed))
    legacy = tabular.fit_label_permutation(*inputs(), seed=seed)
    assert result.artifact_bytes == legacy.artifact_bytes
    assert result.validation_scores == legacy.validation_scores
    prepared, returned = map(json.loads, receipts)
    features, labels, parameters = observed[0]
    expected = np.random.Generator(np.random.PCG64(seed)).permutation(inputs()[1])
    np.testing.assert_array_equal(labels, expected)
    assert prepared["event"] == "fit_call_prepared"
    assert prepared["ordered_record_ids"] == list(
        options(receipts)["training_record_ids"]
    )
    assert prepared["consumed_labels"] == labels.tolist()
    assert prepared["label_class_counts"] == {"0": 48, "1": 48}
    assert prepared["permutation_seed"] == seed
    assert prepared["parameters"] == parameters
    assert (
        prepared["consumed_labels_sha256"]
        == hashlib.sha256(labels.astype("<i8").tobytes(order="C")).hexdigest()
    )
    assert (
        prepared["scaled_features"]["sha256"]
        == hashlib.sha256(features.astype("<f8").tobytes(order="C")).hexdigest()
    )
    assert prepared["scaled_features"]["shape"] == [96, 25]
    assert returned["event"] == "fit_call_returned"
    assert (
        returned["prepared_receipt_sha256"] == hashlib.sha256(receipts[0]).hexdigest()
    )
    assert len(observed) == 1
    assert "https://" not in receipts[0].decode()


def test_original_labels_substituted_at_boundary_stop_before_fit(monkeypatch):
    original = tabular._fit_logistic
    receipts, calls = [], []

    def corrupt(train, labels, validation, **kwargs):
        return original(train, np.asarray(inputs()[1]), validation, **kwargs)

    monkeypatch.setattr(tabular, "_fit_logistic", corrupt)
    monkeypatch.setattr(
        tabular.LogisticRegression, "fit", lambda *args: calls.append(args)
    )
    with pytest.raises(tabular.SecondaryTabularError, match="permuted labels differ"):
        tabular.fit_label_permutation(*inputs(), **options(receipts))
    assert receipts == calls == []


def test_receipt_write_failure_stops_before_estimator_call(monkeypatch):
    calls = []

    def refuse(content):
        raise OSError("invented private path")

    arguments = options([])
    arguments["fit_input_receipt"] = refuse
    monkeypatch.setattr(
        tabular.LogisticRegression, "fit", lambda *args: calls.append(args)
    )
    with pytest.raises(tabular.SecondaryTabularError) as caught:
        tabular.fit_label_permutation(*inputs(), **arguments)
    assert caught.value.check_id == "fit_input_receipt"
    assert "private" not in str(caught.value)
    assert not calls


def test_failed_fit_preserves_prepared_receipt_without_success_claim(monkeypatch):
    receipts = []

    def refuse(*args, **kwargs):
        raise ValueError("invented estimator failure")

    monkeypatch.setattr(tabular.LogisticRegression, "fit", refuse)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.fit_label_permutation(*inputs(), **options(receipts))
    assert [json.loads(value)["event"] for value in receipts] == ["fit_call_prepared"]


@pytest.mark.parametrize("invalid", [[], ["same"] * 96, [""] * 96, "ids", [1] * 96])
def test_invalid_ids_stop_before_fit(monkeypatch, invalid):
    receipts, calls = [], []
    arguments = options(receipts)
    arguments["training_record_ids"] = invalid
    monkeypatch.setattr(
        tabular.LogisticRegression, "fit", lambda *args: calls.append(args)
    )
    with pytest.raises(tabular.SecondaryTabularError) as caught:
        tabular.fit_label_permutation(*inputs(), **arguments)
    assert caught.value.check_id == "input_validation"
    assert receipts == calls == []


@pytest.mark.parametrize("missing", ["training_record_ids", "fit_input_receipt"])
def test_audit_requires_both_ids_and_receipt_writer(missing):
    arguments = options([])
    del arguments[missing]
    with pytest.raises(tabular.SecondaryTabularError) as caught:
        tabular.fit_label_permutation(*inputs(), **arguments)
    assert caught.value.check_id == "input_validation"


def test_completion_receipt_failure_does_not_return_model():
    receipts = []

    def fail_second(content):
        if receipts:
            raise OSError("write failed")
        receipts.append(content)

    arguments = options(receipts)
    arguments["fit_input_receipt"] = fail_second
    with pytest.raises(tabular.SecondaryTabularError) as caught:
        tabular.fit_label_permutation(*inputs(), **arguments)
    assert caught.value.check_id == "fit_input_receipt"
    assert len(receipts) == 1
