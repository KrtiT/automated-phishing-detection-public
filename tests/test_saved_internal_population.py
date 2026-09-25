"""Expose the validated internal population from invented saved evidence bytes."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evaluation_producer import parse, synthetic_session
from test_saved_evidence import expect_synthetic_binding

from automated_phishing_detection import evaluation_producer, saved_evidence
from automated_phishing_detection.hypothesis_evaluation import SavedPopulation
from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
)


@pytest.fixture
def saved(monkeypatch):
    prepared = parse(evaluation_producer)
    session, *replay_calls = synthetic_session(evaluation_producer, monkeypatch)
    produced = evaluation_producer.produce_internal_evidence(prepared, session)
    expect_synthetic_binding(
        saved_evidence, monkeypatch, produced.private_outputs["bindings.json"]
    )
    return SimpleNamespace(
        produced=produced,
        outputs=produced.private_outputs,
        replay_calls=replay_calls,
        transformer_calls=session.primary.scorer.urls,
    )


def _arguments(outputs):
    return tuple(
        outputs[name]
        for name in (
            "predictions.jsonl",
            "manifests.json",
            "bindings.json",
            "routing.json",
        )
    )


def _reconstruct(outputs):
    assert hasattr(saved_evidence, "reconstruct_internal_evidence_and_population"), (
        "missing validated saved-population companion"
    )
    return saved_evidence.reconstruct_internal_evidence_and_population(
        *_arguments(outputs)
    )


def _observe(monkeypatch, name, calls):
    original = getattr(saved_evidence, name)

    def observed(*args, **kwargs):
        calls.append((name, args))
        return original(*args, **kwargs)

    monkeypatch.setattr(saved_evidence, name, observed)


def test_companion_returns_the_exact_validated_population_and_legacy_summary(
    saved, monkeypatch
):
    legacy = saved_evidence.reconstruct_internal_evidence(*_arguments(saved.outputs))
    observed = []
    _observe(monkeypatch, "_secondary", observed)
    result, population = _reconstruct(saved.outputs)
    assert (
        result == legacy
        and type(result) is saved_evidence.ReconstructedInternalEvidence
    )
    assert type(population) is SavedPopulation
    assert population is observed[0][1][1]
    assert population.records == tuple(
        EvaluationRecord(
            row.record.record_id, row.record.registrable_domain, row.record.is_phishing
        )
        for row in saved.produced.rows
    )
    for model, attribute in (
        ("length_only", "length_decision"),
        ("logistic_l1", "stage1_decision"),
        ("transformer", "transformer_decision"),
        ("cascade", "cascade_decision"),
    ):
        assert population.predictions[model] == tuple(
            BinaryPrediction(row.record.record_id, getattr(row, attribute))
            for row in saved.produced.rows
        )


def test_companion_keeps_single_verification_and_performs_no_extra_inference(
    saved, monkeypatch
):
    saved_evidence.reconstruct_internal_evidence(*_arguments(saved.outputs))
    for calls in saved.replay_calls:
        calls.clear()
    stages = (
        "_bindings",
        "_parse_rows",
        "_verify_monitor_path",
        "_routing",
        "_manifests",
        "_secondary",
    )
    observed = []
    for name in stages:
        _observe(monkeypatch, name, observed)

    def forbidden(*args, **kwargs):
        pytest.fail("saved population access attempted a file read or model inference")

    for name in ("open", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(evaluation_producer, "score_bound_secondary", forbidden)
    result, population = _reconstruct(saved.outputs)
    assert [name for name, args in observed] == list(stages)
    assert all(len(calls) == result.row_count for calls in saved.replay_calls)
    assert len(saved.transformer_calls) == result.row_count == len(population.records)


@pytest.mark.parametrize(
    "name", ["predictions.jsonl", "manifests.json", "bindings.json", "routing.json"]
)
def test_companion_preserves_every_saved_byte_validation(saved, name):
    with pytest.raises(saved_evidence.SavedEvidenceError):
        _reconstruct(saved.outputs | {name: b"{}\n"})


def test_companion_does_not_return_population_when_final_summary_fails(
    saved, monkeypatch
):
    def failed(*args, **kwargs):
        raise ValueError("invented late secondary reduction failure")

    monkeypatch.setattr(saved_evidence, "_secondary", failed)
    with pytest.raises(
        saved_evidence.SavedEvidenceError, match="saved evidence reconstruction failed"
    ):
        _reconstruct(saved.outputs)


def test_legacy_signature_and_result_schema_are_preserved():
    assert list(
        inspect.signature(saved_evidence.reconstruct_internal_evidence).parameters
    ) == ["predictions", "manifests", "bindings", "routing"]
    assert list(saved_evidence.ReconstructedInternalEvidence.__dataclass_fields__) == [
        "row_count",
        "domain_count",
        "class_counts",
        "inference_counts",
        "secondary_inference_counts",
        "manifests",
        "primary",
        "secondary",
    ]
