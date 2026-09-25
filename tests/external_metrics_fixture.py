"""Invented, complete external score records for population reduction tests."""

from automated_phishing_detection.bound_secondary import (
    SecondarySeedScore,
    SecondaryTabularScore,
)
from automated_phishing_detection.external_evidence_types import ScoredExternalRow
from automated_phishing_detection.phishvn import PreparedExternalRow
from automated_phishing_detection.primary_scores import PrimaryURLScores
from automated_phishing_detection.selective_inference import InferenceCounts

TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
EXPECTED_COLUMNS = (
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
METADATA = {
    "gold": ("ncsc", "gold", "phishing", "gold", 1),
    "certified": ("trusted_registry", "certified", "legitimate", "certified", 0),
    "ncsc_silver": ("ncsc", "silver", "phishing", "secondary", 1),
    "chongluadao_openphish_bronze": (
        "chongluadao_openphish",
        "bronze",
        "phishing",
        "secondary",
        1,
    ),
    "tranco": ("tranco", "control", "reference_negative", "tranco", None),
}


def _record(population: str, position: int) -> PreparedExternalRow:
    source, tier, designation, role, label = METADATA[population]
    return PreparedExternalRow(
        f"private-record-{position}",
        "private-publisher-split",
        position,
        f"https://private-domain-{position}.test/private-path",
        "a" * 64,
        f"private-domain-{position}.test",
        label,
        role,
        source,
        designation,
        tier,
        "private-publisher-split",
    )


def _primary(alert: int) -> PrimaryURLScores:
    return PrimaryURLScores(
        (0.0,) * 25,
        0.1,
        0.2,
        0.3,
        0.4,
        alert,
        alert,
        alert,
        alert,
        False,
        0.2,
        -1.0,
        "private-length-audit",
        "private-stage1-audit",
        InferenceCounts(1, 1, 1, 0),
    )


def external_row(population: str, position: int, alert: int = 1) -> ScoredExternalRow:
    tabular = tuple(SecondaryTabularScore(name, 0.1, alert) for name in TABULAR_NAMES)
    seeds = tuple(
        SecondarySeedScore(seed, 0.2, alert, 0.3, alert, False)
        for seed in range(42, 47)
    )
    return ScoredExternalRow(
        _record(population, position),
        _primary(alert),
        tabular,
        seeds,
        (0.0,) * 26,
        0.1,
        alert,
        False,
        False,
    )
