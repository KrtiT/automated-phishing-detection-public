"""Aggregate externally scored rows after routing the full retained test stream.

Validation establishes caller consistency, not provenance or source completeness.
No source files, models, fitted thresholds or primary hypothesis gates are used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from . import saved_metrics, secondary_metrics
from ._external_metrics_validation import (
    EXTERNAL_COLUMNS,
    ExternalMetricsError,
    score_pairs,
    validate_external,
)
from .external_evidence_types import ScoredExternalRow
from .paired_evaluation import BinaryPrediction, EvaluationRecord

__all__ = ["EXTERNAL_COLUMNS", "ExternalMetricsError", "summarize_external"]


def _support(rows: tuple[ScoredExternalRow, ...]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "domain_count": len({row.record.registrable_domain for row in rows}),
        "positive_count": sum(row.record.is_phishing == 1 for row in rows),
        "negative_count": sum(row.record.is_phishing == 0 for row in rows),
        "unlabeled_count": sum(row.record.is_phishing is None for row in rows),
    }


def _records(rows: tuple[ScoredExternalRow, ...]) -> tuple[EvaluationRecord, ...]:
    return tuple(
        EvaluationRecord(
            row.record.record_id, row.record.registrable_domain, row.record.is_phishing
        )
        for row in rows
    )


def _columns(rows: tuple[ScoredExternalRow, ...]) -> dict[str, tuple]:
    values = tuple(score_pairs(row) for row in rows)
    return {
        name: (
            tuple(
                secondary_metrics.ScorePrediction(
                    row.record.record_id, scores[index][0]
                )
                for row, scores in zip(rows, values, strict=True)
            ),
            tuple(
                BinaryPrediction(row.record.record_id, scores[index][1])
                for row, scores in zip(rows, values, strict=True)
            ),
        )
        for index, name in enumerate(EXTERNAL_COLUMNS)
    }


def _population(rows: tuple[ScoredExternalRow, ...], mode: str) -> dict:
    records = () if mode == "control" else _records(rows)
    detectors = {}
    for name, (scores, decisions) in _columns(rows).items():
        if mode == "control":
            identities = tuple(row.record.record_id for row in rows)
            metric = {
                "control_alert_rate": asdict(
                    saved_metrics.control_alert_rate(identities, decisions)
                )
            }
        elif mode == "combined":
            metric = asdict(
                secondary_metrics.secondary_metrics(records, scores, decisions)
            )
        else:
            counts = saved_metrics.detection_metrics(records, decisions)
            metric = (
                {"recall": asdict(counts.recall)}
                if mode == "recall"
                else asdict(counts)
            )
        detectors[name] = metric
    return {**_support(rows), "detectors": detectors}


def _secondary_rows(
    rows: tuple[ScoredExternalRow, ...], source: str, tier: str
) -> tuple[ScoredExternalRow, ...]:
    return tuple(
        row
        for row in rows
        if (
            row.record.source_group,
            row.record.confidence_tier,
            row.record.source_class,
            row.record.role,
        )
        == (source, tier, "phishing", "secondary")
    )


def _populations(rows: tuple[ScoredExternalRow, ...]) -> dict:
    selected = {
        "gold": (tuple(row for row in rows if row.record.role == "gold"), "counts"),
        "certified": (
            tuple(row for row in rows if row.record.role == "certified"),
            "counts",
        ),
        "gold_plus_certified": (
            tuple(row for row in rows if row.record.role in ("gold", "certified")),
            "combined",
        ),
        "ncsc_silver": (
            _secondary_rows(rows, "ncsc", "silver"),
            "recall",
        ),
        "chongluadao_openphish_bronze": (
            _secondary_rows(rows, "chongluadao_openphish", "bronze"),
            "recall",
        ),
        "tranco": (
            tuple(row for row in rows if row.record.role == "tranco"),
            "control",
        ),
    }
    return {name: _population(*population) for name, population in selected.items()}


def _contingency(rows: tuple[ScoredExternalRow, ...]) -> list[dict]:
    grouped = defaultdict(list)
    names = ("source_group", "confidence_tier", "source_class", "role", "is_phishing")
    for row in rows:
        grouped[tuple(getattr(row.record, name) for name in names)].append(row)
    return [
        {**dict(zip(names, key, strict=True)), **_support(tuple(grouped[key]))}
        for key in sorted(grouped)
    ]


def _mcnemar(rows: tuple[ScoredExternalRow, ...]) -> dict:
    gold = tuple(row for row in rows if row.record.role == "gold")
    records, columns = _records(gold), _columns(gold)
    comparisons = (
        ("external_gold_logistic_minus_length", "logistic_l1", "length_only"),
        ("external_gold_cascade_minus_logistic", "cascade", "logistic_l1"),
    )
    return {
        name: asdict(
            secondary_metrics.exact_mcnemar(
                records, columns[candidate][1], columns[reference][1]
            )
        )
        for name, candidate, reference in comparisons
    }


def summarize_external(rows: tuple[ScoredExternalRow, ...]) -> dict:
    """Reduce complete, caller-retained test rows without altering frozen decisions."""
    validate_external(rows)
    return {
        "schema_version": 1,
        "scope": "retained_publisher_test_stream",
        "analysis_role": "descriptive_secondary_not_primary",
        **_support(rows),
        "detector_columns": list(EXTERNAL_COLUMNS),
        "populations": _populations(rows),
        "source_contingency": _contingency(rows),
        "mcnemar": _mcnemar(rows),
    }
