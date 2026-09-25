"""Validate caller consistency for external reductions, without source claims."""

from __future__ import annotations

import math
from numbers import Real

from . import proposed_label_contract, protocol_preflight
from .bound_secondary import SecondarySeedScore, SecondaryTabularScore
from .external_evidence_types import ScoredExternalRow
from .phishvn import PreparedExternalRow
from .primary_scores import PrimaryURLScores

TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
PRIMARY_FIELDS = ("length", "stage1", "transformer", "cascade")
EXTERNAL_COLUMNS = (
    "length_only",
    "logistic_l1",
    "transformer",
    "cascade",
    "policy",
    *(f"tabular.{name}" for name in TABULAR_NAMES),
    *(
        f"seed_{seed}.{model}"
        for seed in range(42, 47)
        for model in ("transformer", "cascade")
    ),
)


class ExternalMetricsError(ValueError):
    """External records violate the declared population or score contract."""


def score_pairs(row: ScoredExternalRow) -> tuple[tuple[float, int], ...]:
    primary = tuple(
        (
            getattr(row.primary, f"{name}_probability"),
            getattr(row.primary, f"{name}_decision"),
        )
        for name in PRIMARY_FIELDS
    )
    tabular = tuple(
        (score.probability, score.decision) for score in row.secondary_tabular
    )
    seeds = tuple(
        (getattr(score, f"{name}_probability"), getattr(score, f"{name}_decision"))
        for score in row.secondary_seeds
        for name in ("transformer", "cascade")
    )
    return (*primary, (row.policy_probability, row.policy_decision), *tabular, *seeds)


def _stable_string(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and all(
            character.isprintable() and not character.isspace() for character in value
        )
    )


def _metadata(record: PreparedExternalRow) -> None:
    if not isinstance(record, PreparedExternalRow):
        raise ExternalMetricsError("external record must be typed")
    if not _stable_string(record.record_id):
        raise ExternalMetricsError("record ID must be a nonempty stable string")
    if (
        not _stable_string(record.source_split)
        or record.published_split != record.source_split
        or type(record.file_position) is not int
        or record.file_position < 1
    ):
        raise ExternalMetricsError("source split and position must be consistent")
    try:
        domain = protocol_preflight._ascii_domain(record.registrable_domain)
        protocol_preflight._reject_ip_literal(domain)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ExternalMetricsError("invalid registrable domain") from exc
    if domain != record.registrable_domain:
        raise ExternalMetricsError("registrable domain must already be canonical")
    _source_policy(record)


def _source_policy(record: PreparedExternalRow) -> None:
    policy = proposed_label_contract.evaluate_phishvn_policy(
        record.source_group, record.confidence_tier, record.source_class
    )
    expected_role = {
        "primary_external": "gold" if policy.is_phishing == 1 else "certified",
        "secondary_or_sensitivity": "secondary",
        "reference_negative_control": "tranco",
    }.get(policy.evidence_role)
    if expected_role is None or record.role != expected_role:
        raise ExternalMetricsError("source policy and role must match")
    label = record.is_phishing
    if policy.is_phishing is None:
        if label is not None:
            raise ExternalMetricsError("control label must remain null")
    elif type(label) is not int or label != policy.is_phishing:
        raise ExternalMetricsError("source policy and binary integer label must match")


def _families(row: ScoredExternalRow) -> None:
    tabular, seeds = row.secondary_tabular, row.secondary_seeds
    if (
        type(tabular) is not tuple
        or len(tabular) != 7
        or any(not isinstance(score, SecondaryTabularScore) for score in tabular)
    ):
        raise ExternalMetricsError("tabular family must have seven typed scores")
    if tuple(score.name for score in tabular) != TABULAR_NAMES:
        raise ExternalMetricsError("tabular family names and order must match")
    if (
        type(seeds) is not tuple
        or len(seeds) != 5
        or any(not isinstance(score, SecondarySeedScore) for score in seeds)
    ):
        raise ExternalMetricsError("seed family must have five typed scores")
    if any(type(score.seed) is not int for score in seeds) or tuple(
        score.seed for score in seeds
    ) != tuple(range(42, 47)):
        raise ExternalMetricsError("seed family values and order must match")
    if any(type(score.band_selected) is not bool for score in seeds):
        raise ExternalMetricsError("seed band selections must be booleans")


def _finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _scores(row: ScoredExternalRow) -> None:
    if not isinstance(row.primary, PrimaryURLScores):
        raise ExternalMetricsError("primary scores must be typed")
    _families(row)
    for probability, decision in score_pairs(row):
        if not _finite_real(probability) or not 0 <= probability <= 1:
            raise ExternalMetricsError("probability must be finite and within [0, 1]")
        if type(decision) is not int or decision not in (0, 1):
            raise ExternalMetricsError("decision must be a binary integer")
    features = row.standardized_monitor_features
    if (
        type(features) is not tuple
        or len(features) != 26
        or any(not _finite_real(value) for value in features)
    ):
        raise ExternalMetricsError(
            "monitor representation must contain 26 finite values"
        )
    if any(
        type(value) is not bool
        for value in (
            row.drift_override,
            row.logical_stage2_mask,
            row.primary.band_selected,
        )
    ):
        raise ExternalMetricsError("routing flags must be booleans")


def validate_external(rows: tuple[ScoredExternalRow, ...]) -> None:
    if type(rows) is not tuple:
        raise ExternalMetricsError("external evidence must be a materialized tuple")
    identities = set()
    for row in rows:
        if not isinstance(row, ScoredExternalRow):
            raise ExternalMetricsError("external evidence must use typed rows")
        _metadata(row.record)
        if row.record.record_id in identities:
            raise ExternalMetricsError("record IDs must be unique")
        identities.add(row.record.record_id)
        _scores(row)
