"""Primary receipt and secondary shape validation precede member inference."""

import importlib
import importlib.util
from dataclasses import replace
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
from bound_secondary_column_fixtures import scoring as scoring
from external_secondary_fixtures import module as module
from external_secondary_fixtures import phase as phase

from automated_phishing_detection.selective_inference import InferenceCounts


@pytest.mark.parametrize(
    "field,value",
    [
        ("scores", ()),
        ("records", []),
        ("checkpoint_bytes", b"changed"),
        ("receipt_bytes", b"{}\n"),
        ("thresholds", ()),
        ("inference_counts", InferenceCounts(True, 2, 2, 0)),
        ("inference_counts", InferenceCounts(3, 2, 2, 0)),
    ],
)
def test_invalid_primary_never_starts_secondary_inference(
    module: ModuleType,
    phase: SimpleNamespace,
    field: str,
    value: object,
) -> None:
    primary = replace(phase.primary, **{field: value})
    with pytest.raises(module.ExternalSecondaryError) as caught:
        module.score_external_secondary(primary, phase.bound)
    assert phase.events == phase.singletons == []
    assert str(caught.value) == "external_secondary_input_validation_failed"


@pytest.mark.parametrize("fault", ["stage1", "tabular", "seed", "hash", "callback"])
def test_invalid_bound_or_callback_prevents_every_member_call(
    module: ModuleType,
    phase: SimpleNamespace,
    fault: str,
) -> None:
    bound, retain = phase.bound, None
    if fault == "stage1":
        bound = replace(bound, stage1_threshold=0.6)
    elif fault == "tabular":
        bound = replace(bound, tabular=bound.tabular[::-1])
    elif fault == "seed":
        bound = replace(bound, seeds=bound.seeds[:-1])
    elif fault == "hash":
        first = replace(bound.tabular[0], artifact_sha256="not-a-digest")
        bound = replace(bound, tabular=(first, *bound.tabular[1:]))
    else:
        retain = "not-callable"
    with pytest.raises(module.ExternalSecondaryError):
        module.score_external_secondary(phase.primary, bound, retain=retain)
    assert phase.events == phase.singletons == []


def test_reusable_validation_helpers_accept_exact_phase_results(
    module: ModuleType,
    phase: SimpleNamespace,
) -> None:
    name = "automated_phishing_detection._external_secondary_inputs"
    assert importlib.util.find_spec(name), "missing shared secondary validators"
    validators = importlib.import_module(name)
    result = module.score_external_secondary(phase.primary, phase.bound)
    assert all(
        score.inference_counts == InferenceCounts(1, 1, 1, 0)
        for score in phase.primary.scores
    )
    assert phase.primary.inference_counts == InferenceCounts(2, 2, 2, 0)
    assert validators.validate_primary_phase(phase.primary) is None
    assert validators.validate_secondary_scoring(result.scoring, 2) is None


@pytest.mark.parametrize(
    "fault", ["rows", "count", "family", "probability", "decision", "seed"]
)
def test_reusable_secondary_validation_rejects_inconsistent_scoring(
    module: ModuleType,
    phase: SimpleNamespace,
    fault: str,
) -> None:
    validators = importlib.import_module(
        "automated_phishing_detection._external_secondary_inputs"
    )
    result = module.score_external_secondary(phase.primary, phase.bound).scoring
    if fault == "rows":
        result = replace(result, rows=result.rows[:-1])
    elif fault == "count":
        result = replace(
            result,
            counts=replace(result.counts, reused_primary_transformer_scores=True),
        )
    else:
        row = result.rows[0]
        if fault == "family":
            row = replace(row, tabular=row.tabular[::-1])
        elif fault == "seed":
            row = replace(row, seeds=(replace(row.seeds[0], seed=42.0), *row.seeds[1:]))
        else:
            value = {"probability": float("nan"), "decision": True}[fault]
            first = replace(row.tabular[0], **{fault: value})
            row = replace(row, tabular=(first, *row.tabular[1:]))
        result = replace(result, rows=(row, *result.rows[1:]))
    with pytest.raises(ValueError):
        validators.validate_secondary_scoring(result, 2)


@pytest.mark.parametrize("target", ["primary", "secondary"])
def test_shared_validators_symbolically_reject_unrepresentable_numbers(
    module: ModuleType,
    phase: SimpleNamespace,
    target: str,
) -> None:
    validators = importlib.import_module(
        "automated_phishing_detection._external_secondary_inputs"
    )
    if target == "primary":
        score = replace(phase.primary.scores[0], features=(10**1000,) * 25)
        value = replace(phase.primary, scores=(score, *phase.primary.scores[1:]))
        validate = validators.validate_primary_phase
    else:
        value = module.score_external_secondary(phase.primary, phase.bound).scoring
        row = value.rows[0]
        score = replace(row.tabular[0], probability=10**1000)
        row = replace(row, tabular=(score, *row.tabular[1:]))
        value = replace(value, rows=(row, *value.rows[1:]))
        validate = partial(validators.validate_secondary_scoring, row_count=2)
    with pytest.raises(ValueError, match="invalid_external_secondary_evidence"):
        validate(value)
