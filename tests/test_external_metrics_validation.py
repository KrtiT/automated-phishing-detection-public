"""Reject malformed external evidence before any metric reduction."""

import importlib
from dataclasses import replace
from types import ModuleType

import pytest
from external_metrics_fixture import external_row


def reducer() -> ModuleType:
    return importlib.import_module("automated_phishing_detection.external_metrics")


@pytest.mark.parametrize("bad", [[], {}, None, "private-source", (object(),)])
def test_requires_materialized_typed_rows(bad: object) -> None:
    with pytest.raises(reducer().ExternalMetricsError):
        reducer().summarize_external(bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "unknown"},
        {"role": "secondary"},
        {"is_phishing": None},
        {"is_phishing": 0},
        {"is_phishing": True},
        {"is_phishing": 1.0},
        {"source_group": "private-unknown-source"},
        {"confidence_tier": "silver"},
        {"source_class": "legitimate"},
        {"record_id": ""},
        {"record_id": "bad id"},
        {"registrable_domain": "INVALID.test"},
        {"registrable_domain": "127.0.0.1"},
        {"source_split": "other_split"},
        {"file_position": 0},
    ],
)
def test_rejects_invalid_metadata_and_policy(changes: dict) -> None:
    row = external_row("gold", 1)
    row = replace(row, record=replace(row.record, **changes))
    with pytest.raises(reducer().ExternalMetricsError) as caught:
        reducer().summarize_external((row,))
    assert "private-unknown-source" not in str(caught.value)


@pytest.mark.parametrize("label", [0, 1, False, 0.0])
def test_control_label_must_remain_null(label: object) -> None:
    row = external_row("tranco", 1)
    row = replace(row, record=replace(row.record, is_phishing=label))
    with pytest.raises(reducer().ExternalMetricsError):
        reducer().summarize_external((row,))


def test_rejects_duplicate_ids_across_different_populations() -> None:
    gold, tranco = external_row("gold", 1), external_row("tranco", 2)
    tranco = replace(
        tranco, record=replace(tranco.record, record_id=gold.record.record_id)
    )
    with pytest.raises(reducer().ExternalMetricsError, match="unique"):
        reducer().summarize_external((gold, tranco))


@pytest.mark.parametrize("family", ["secondary_tabular", "secondary_seeds"])
@pytest.mark.parametrize("change", ["missing", "extra", "reversed", "untyped", "list"])
def test_requires_fixed_typed_family_order(family: str, change: str) -> None:
    row = external_row("gold", 1)
    original = getattr(row, family)
    changed = {
        "missing": original[:-1],
        "extra": (*original, original[0]),
        "reversed": original[::-1],
        "untyped": ({}, *original[1:]),
        "list": list(original),
    }[change]
    with pytest.raises(reducer().ExternalMetricsError):
        reducer().summarize_external((replace(row, **{family: changed}),))


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
        True,
        "0.5",
        None,
        pytest.param(10**1000, id="unrepresentable_integer"),
    ],
)
@pytest.mark.parametrize("target", ["primary", "tabular", "seed", "policy"])
def test_all_score_families_require_finite_probabilities(
    bad: object, target: str
) -> None:
    row = external_row("tranco", 1)
    if target == "primary":
        row = replace(row, primary=replace(row.primary, cascade_probability=bad))
    elif target == "tabular":
        scores = (
            replace(row.secondary_tabular[0], probability=bad),
            *row.secondary_tabular[1:],
        )
        row = replace(row, secondary_tabular=scores)
    elif target == "seed":
        scores = (
            replace(row.secondary_seeds[0], cascade_probability=bad),
            *row.secondary_seeds[1:],
        )
        row = replace(row, secondary_seeds=scores)
    else:
        row = replace(row, policy_probability=bad)
    with pytest.raises(reducer().ExternalMetricsError, match="probability"):
        reducer().summarize_external((row,))


@pytest.mark.parametrize("bad", [True, 1.0, 2, None])
@pytest.mark.parametrize("target", ["primary", "tabular", "seed", "policy"])
def test_all_score_families_require_exact_binary_decisions(
    bad: object, target: str
) -> None:
    row = external_row("tranco", 1)
    if target == "primary":
        row = replace(row, primary=replace(row.primary, transformer_decision=bad))
    elif target == "tabular":
        scores = (
            replace(row.secondary_tabular[0], decision=bad),
            *row.secondary_tabular[1:],
        )
        row = replace(row, secondary_tabular=scores)
    elif target == "seed":
        scores = (
            replace(row.secondary_seeds[0], transformer_decision=bad),
            *row.secondary_seeds[1:],
        )
        row = replace(row, secondary_seeds=scores)
    else:
        row = replace(row, policy_decision=bad)
    with pytest.raises(reducer().ExternalMetricsError, match="decision"):
        reducer().summarize_external((row,))


@pytest.mark.parametrize(
    "change",
    [
        {"record": {}},
        {"primary": {}},
        {"standardized_monitor_features": ()},
        {"standardized_monitor_features": (float("nan"),) * 26},
        {"standardized_monitor_features": (10**1000,) * 26},
        {"logical_stage2_mask": 1},
        {"drift_override": 0},
    ],
)
def test_rejects_malformed_typed_row_members(change: dict) -> None:
    with pytest.raises(reducer().ExternalMetricsError):
        reducer().summarize_external((replace(external_row("gold", 1), **change),))
