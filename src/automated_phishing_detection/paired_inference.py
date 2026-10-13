"""Post-review intervals for independent matched binary outcomes.

The frozen v1 bootstrap remains in paired_evaluation for historical reproduction.
Newcombe (1998), Statistics in Medicine 17:2635-2650, method 10 is an approximate
score interval. The second method projects simultaneous exact binomial bounds
for the discordant probabilities; its coverage is conservative, not shortest.
Neither method treats multiple URLs within one domain as independent trials.
Distinct domains alone do not establish absence of cross-domain dependence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from scipy.stats import beta

from .paired_evaluation import BinaryPrediction, EvaluationRecord, _validate_pairs


@dataclass(frozen=True)
class PairedTable:
    both_positive: int
    candidate_only: int
    reference_only: int
    both_negative: int

    def __post_init__(self) -> None:
        values = (
            self.both_positive,
            self.candidate_only,
            self.reference_only,
            self.both_negative,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("paired counts must be nonnegative integers")
        if not self.total:
            raise ValueError("paired inference requires at least one pair")

    @property
    def total(self) -> int:
        return (
            self.both_positive
            + self.candidate_only
            + self.reference_only
            + self.both_negative
        )


@dataclass(frozen=True)
class PairedInterval:
    method: str
    confidence: float
    estimate: float
    lower: float
    upper: float


def count_independent_pairs(
    records: Sequence[EvaluationRecord],
    candidate: Sequence[BinaryPrediction],
    reference: Sequence[BinaryPrediction],
) -> PairedTable:
    """Validate alignment and uniqueness; domain independence remains assumed."""
    _validate_pairs(records, candidate, reference)
    if len({record.registrable_domain for record in records}) != len(records):
        raise ValueError(
            "independent-pair inference requires one positive record per domain"
        )
    cells = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
    for left, right in zip(candidate, reference, strict=True):
        cells[(left.decision, right.decision)] += 1
    return PairedTable(*(cells[key] for key in ((1, 1), (1, 0), (0, 1), (0, 0))))


def _validate(table: PairedTable, confidence: float) -> float:
    if not isinstance(table, PairedTable):
        raise ValueError("use a validated PairedTable")
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and strictly between zero and one")
    return 1 - confidence


def _wilson(successes: int, total: int, quantile: float) -> tuple[float, float]:
    rate = successes / total
    denominator = 1 + quantile**2 / total
    center = (rate + quantile**2 / (2 * total)) / denominator
    half_width = (
        quantile
        / denominator
        * math.sqrt(rate * (1 - rate) / total + quantile**2 / (4 * total**2))
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def _corrected_correlation(table: PairedTable) -> float:
    denominator_squared = (
        (table.both_positive + table.candidate_only)
        * (table.reference_only + table.both_negative)
        * (table.both_positive + table.reference_only)
        * (table.candidate_only + table.both_negative)
    )
    if not denominator_squared:
        return 0.0
    numerator = (
        table.both_positive * table.both_negative
        - table.candidate_only * table.reference_only
    )
    if numerator > 0:
        numerator = max(numerator - table.total / 2, 0.0)
    return numerator / math.sqrt(denominator_squared)


def _correlated_gap(first: float, second: float, correlation: float) -> float:
    return math.sqrt(first**2 + second**2 - 2 * correlation * first * second)


def paired_score_interval(
    table: PairedTable, *, confidence: float = 0.95
) -> PairedInterval:
    """Newcombe method 10; nominal two-sided coverage, not an exact guarantee."""
    alpha = _validate(table, confidence)
    quantile = NormalDist().inv_cdf(1 - alpha / 2)
    candidate_rate = (table.both_positive + table.candidate_only) / table.total
    reference_rate = (table.both_positive + table.reference_only) / table.total
    candidate_lower, candidate_upper = _wilson(
        table.both_positive + table.candidate_only, table.total, quantile
    )
    reference_lower, reference_upper = _wilson(
        table.both_positive + table.reference_only, table.total, quantile
    )
    correlation = _corrected_correlation(table)
    lower_gap = _correlated_gap(
        candidate_rate - candidate_lower, reference_upper - reference_rate, correlation
    )
    upper_gap = _correlated_gap(
        candidate_upper - candidate_rate, reference_rate - reference_lower, correlation
    )
    estimate = (table.candidate_only - table.reference_only) / table.total
    return PairedInterval(
        "newcombe_1998_method_10",
        confidence,
        estimate,
        max(-1.0, estimate - lower_gap),
        min(1.0, estimate + upper_gap),
    )


def _exact_binomial_bounds(
    successes: int, total: int, tail_alpha: float
) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail_alpha, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.isf(tail_alpha, successes + 1, total - successes))
    )
    return lower, upper


def paired_exact_envelope(
    table: PairedTable, *, confidence: float = 0.95
) -> PairedInterval:
    """Conservative unconditional coverage via four alpha/4 marginal tails.

    Each discordant count has a binomial marginal over independent paired trials.
    The union bound, which does not require independence of those two counts, gives
    simultaneous coverage at least confidence. Projecting that rectangle onto the
    difference preserves this guarantee. This is not a nuisance-optimized interval.
    """
    alpha = _validate(table, confidence)
    candidate_lower, candidate_upper = _exact_binomial_bounds(
        table.candidate_only, table.total, alpha / 4
    )
    reference_lower, reference_upper = _exact_binomial_bounds(
        table.reference_only, table.total, alpha / 4
    )
    estimate = (table.candidate_only - table.reference_only) / table.total
    return PairedInterval(
        "bonferroni_clopper_pearson_discordance",
        confidence,
        estimate,
        candidate_lower - reference_upper,
        candidate_upper - reference_lower,
    )
