"""Population reductions use complete invented evidence without source access."""

import importlib
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import replace
from types import ModuleType

import pytest
from external_metrics_fixture import EXPECTED_COLUMNS, METADATA, external_row

from automated_phishing_detection import secondary_metrics
from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
)


def reducer() -> ModuleType:
    name = "automated_phishing_detection.external_metrics"
    assert importlib.util.find_spec(name), "external population reducer is missing"
    return importlib.import_module(name)


def test_complete_columns_and_separate_population_support() -> None:
    rows = tuple(external_row(name, index + 1) for index, name in enumerate(METADATA))
    summary = reducer().summarize_external(rows)
    assert tuple(summary["detector_columns"]) == EXPECTED_COLUMNS
    assert summary["scope"] == "retained_publisher_test_stream"
    assert summary["row_count"] == summary["domain_count"] == 5
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == summary["unlabeled_count"] == 1
    assert set(summary["populations"]) == {*METADATA, "gold_plus_certified"}
    for population, result in summary["populations"].items():
        assert tuple(result["detectors"]) == EXPECTED_COLUMNS
        assert result["row_count"] == (2 if population == "gold_plus_certified" else 1)
    for result in summary["populations"]["gold"]["detectors"].values():
        assert result["true_positives"] == result["recall"]["denominator"] == 1
    for result in summary["populations"]["certified"]["detectors"].values():
        assert result["false_positives"] == result["fpr"]["denominator"] == 1


def test_combined_metrics_preserve_routed_policy_decisions() -> None:
    positive = external_row("gold", 1, 1)
    negative = replace(external_row("certified", 2, 0), policy_probability=0.9)
    result = reducer().summarize_external((positive, negative))["populations"]
    policy = result["gold_plus_certified"]["detectors"]["policy"]
    assert policy["counts"]["true_positives"] == 1
    assert policy["counts"]["true_negatives"] == 1
    assert policy["roc_auc"]["value"] == 0.0
    assert policy["brier"]["value"] == pytest.approx(0.81)
    assert policy["precision"]["value"] == policy["f2"]["value"] == 1.0
    assert len(policy["calibration_bins"]) == 10
    assert len(policy["prevalence_projections"]) == 3


def test_combined_kernel_observes_original_relative_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(
        external_row(name, position)
        for position, name in enumerate(
            ("certified", "ncsc_silver", "gold", "tranco", "certified"), 1
        )
    )
    observed = []
    original = secondary_metrics.secondary_metrics

    def capture(
        records: Sequence[EvaluationRecord],
        scores: Sequence[secondary_metrics.ScorePrediction],
        decisions: Sequence[BinaryPrediction],
    ) -> secondary_metrics.SecondaryMetrics:
        observed.append(tuple(record.record_id for record in records))
        return original(records, scores, decisions)

    monkeypatch.setattr(secondary_metrics, "secondary_metrics", capture)
    reducer().summarize_external(rows)
    assert observed == [tuple(rows[index].record.record_id for index in (0, 2, 4))] * 22


def test_silver_bronze_recall_and_label_free_tranco_stay_separate() -> None:
    rows = (
        external_row("gold", 1, 0),
        external_row("ncsc_silver", 2, 1),
        external_row("chongluadao_openphish_bronze", 3, 0),
        external_row("tranco", 4, 1),
    )
    populations = reducer().summarize_external(rows)["populations"]
    for column in EXPECTED_COLUMNS:
        assert populations["gold"]["detectors"][column]["recall"]["estimate"] == 0.0
        silver = populations["ncsc_silver"]["detectors"][column]
        bronze = populations["chongluadao_openphish_bronze"]["detectors"][column]
        control = populations["tranco"]["detectors"][column]
        assert set(silver) == set(bronze) == {"recall"}
        assert silver["recall"]["estimate"] == 1.0
        assert bronze["recall"]["estimate"] == 0.0
        assert set(control) == {"control_alert_rate"}
        assert control["control_alert_rate"]["estimate"] == 1.0


def test_contingency_counts_repeated_domains() -> None:
    original = external_row("gold", 1)
    repeated = external_row("gold", 2, 0)
    repeated = replace(
        repeated,
        record=replace(
            repeated.record, registrable_domain=original.record.registrable_domain
        ),
    )
    rows = (original, repeated, external_row("certified", 3), external_row("tranco", 4))
    summary = reducer().summarize_external(rows)
    gold = next(
        cell for cell in summary["source_contingency"] if cell["role"] == "gold"
    )
    assert gold == {
        "source_group": "ncsc",
        "confidence_tier": "gold",
        "source_class": "phishing",
        "role": "gold",
        "is_phishing": 1,
        "row_count": 2,
        "domain_count": 1,
        "positive_count": 2,
        "negative_count": 0,
        "unlabeled_count": 0,
    }


def test_public_summary_contains_no_private_record_values() -> None:
    rows = tuple(external_row(name, index + 1) for index, name in enumerate(METADATA))
    summary = reducer().summarize_external(rows)
    public = json.dumps(summary, allow_nan=False)
    for private in (
        "private-record",
        "private-domain",
        "private-path",
        "private-length",
        "private-stage1",
        "private-publisher-split",
        "a" * 64,
    ):
        assert private not in public


@pytest.mark.parametrize("population", [None, "gold", "certified", "tranco"])
def test_empty_and_single_class_populations_retain_null_reasons(
    population: str | None,
) -> None:
    rows = () if population is None else (external_row(population, 1, 0),)
    summary = reducer().summarize_external(rows)
    for result in summary["populations"].values():
        assert tuple(result["detectors"]) == EXPECTED_COLUMNS
    combined = summary["populations"]["gold_plus_certified"]["detectors"]
    for metrics in combined.values():
        assert metrics["average_precision"] == {
            "value": None,
            "reason": "both_classes_required",
        }
        assert metrics["roc_auc"]["value"] is None
    if population is None:
        assert summary["source_contingency"] == []
        assert combined["policy"]["brier"]["reason"] == "empty_population"
        control = summary["populations"]["tranco"]["detectors"]["policy"]
        assert control["control_alert_rate"]["status"] == "not_estimable"


def test_exact_gold_mcnemar_keeps_two_unadjusted_external_cells() -> None:
    rows = tuple(external_row("gold", index) for index in range(1, 4))
    rows = tuple(
        replace(row, primary=replace(row.primary, length_decision=0)) for row in rows
    )
    result = reducer().summarize_external(rows)["mcnemar"]
    assert tuple(result) == (
        "external_gold_logistic_minus_length",
        "external_gold_cascade_minus_logistic",
    )
    assert result["external_gold_logistic_minus_length"]["candidate_only_correct"] == 3
    assert result["external_gold_logistic_minus_length"]["pvalue"]["value"] == 0.25
    assert result["external_gold_cascade_minus_logistic"]["pvalue"]["value"] == 1.0
    assert "holm" not in json.dumps(result).lower()
    empty = reducer().summarize_external(())["mcnemar"]
    assert all(
        cell["pvalue"]["reason"] == "empty_population" for cell in empty.values()
    )
