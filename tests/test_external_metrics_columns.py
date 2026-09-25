"""Distinct saved scores and decisions remain attached to all frozen columns."""

from dataclasses import replace

import pytest
from external_metrics_fixture import EXPECTED_COLUMNS, external_row

from automated_phishing_detection.external_evidence_types import ScoredExternalRow
from automated_phishing_detection.external_metrics import summarize_external


def _primary_and_policy(row: ScoredExternalRow) -> ScoredExternalRow:
    changes = {}
    for index, field in enumerate(("length", "stage1", "transformer", "cascade")):
        changes[f"{field}_probability"] = (index + 1) / 100
        changes[f"{field}_decision"] = index % 2
    return replace(
        row,
        primary=replace(row.primary, **changes),
        policy_probability=0.05,
        policy_decision=0,
    )


def _distinct_columns(row: ScoredExternalRow) -> ScoredExternalRow:
    tabular = tuple(
        replace(score, probability=(index + 6) / 100, decision=(index + 5) % 2)
        for index, score in enumerate(row.secondary_tabular)
    )
    seeds = tuple(
        replace(
            score,
            transformer_probability=(13 + 2 * index) / 100,
            cascade_probability=(14 + 2 * index) / 100,
            transformer_decision=0,
            cascade_decision=1,
        )
        for index, score in enumerate(row.secondary_seeds)
    )
    return replace(
        _primary_and_policy(row), secondary_tabular=tabular, secondary_seeds=seeds
    )


def test_each_column_uses_its_own_saved_scores_and_binary_decisions() -> None:
    rows = tuple(
        _distinct_columns(external_row(name, position))
        for position, name in (
            (1, "gold"),
            (2, "certified"),
            (3, "tranco"),
        )
    )
    populations = summarize_external(rows)["populations"]
    for index, name in enumerate(EXPECTED_COLUMNS):
        probability, decision = (index + 1) / 100, index % 2
        combined = populations["gold_plus_certified"]["detectors"][name]
        assert combined["brier"]["value"] == pytest.approx(
            ((1 - probability) ** 2 + probability**2) / 2
        )
        assert combined["counts"]["true_positives"] == decision
        assert combined["counts"]["true_negatives"] == 1 - decision
        assert combined["average_precision"]["value"] == 0.5
        assert (
            populations["tranco"]["detectors"][name]["control_alert_rate"]["numerator"]
            == decision
        )
