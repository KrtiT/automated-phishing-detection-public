"""Structural secondary validation without fitting, inference or threshold changes."""

from dataclasses import asdict

from . import bound_secondary as secondary
from . import fixed_cascade

TABULAR_NAMES = secondary._TABULAR_NAMES
SEEDS = secondary._SEEDS


class SecondaryInputError(ValueError):
    """Supplied external phase evidence is internally inconsistent."""


def require(condition: bool) -> None:
    if not condition:
        raise SecondaryInputError("invalid_external_secondary_evidence")


def probability(value: object) -> None:
    require(type(value) in (int, float))
    require(0 <= fixed_cascade._finite_number(value, "probability") <= 1)


def binary(value: object) -> None:
    require(type(value) is int and value in (0, 1))


def validate_tabular_score(score: secondary.SecondaryTabularScore, name: str) -> None:
    require(type(score) is secondary.SecondaryTabularScore)
    require(type(score.name) is str and score.name == name)
    probability(score.probability)
    binary(score.decision)


def validate_seed_score(score: secondary.SecondarySeedScore, seed: int) -> None:
    require(type(score) is secondary.SecondarySeedScore)
    require(type(score.seed) is int and score.seed == seed)
    for prefix in ("transformer", "cascade"):
        probability(getattr(score, f"{prefix}_probability"))
        binary(getattr(score, f"{prefix}_decision"))
    require(type(score.band_selected) is bool)


def _validate_row(row: secondary.SecondaryScoredRow) -> None:
    require(type(row) is secondary.SecondaryScoredRow)
    require(type(row.tabular) is tuple and len(row.tabular) == len(TABULAR_NAMES))
    require(type(row.seeds) is tuple and len(row.seeds) == len(SEEDS))
    for score, name in zip(row.tabular, TABULAR_NAMES, strict=True):
        validate_tabular_score(score, name)
    for score, seed in zip(row.seeds, SEEDS, strict=True):
        validate_seed_score(score, seed)


def _validate_secondary_scoring(
    scoring: secondary.SecondaryScoring, row_count: int
) -> None:
    """Require every frozen family and exact completed physical counts."""
    require(type(row_count) is int and row_count >= 0)
    require(type(scoring) is secondary.SecondaryScoring)
    require(type(scoring.rows) is tuple and len(scoring.rows) == row_count)
    for row in scoring.rows:
        _validate_row(row)
    require(type(scoring.counts) is secondary.SecondaryInferenceCounts)
    expected = secondary.SecondaryInferenceCounts(
        tuple((name, row_count) for name in TABULAR_NAMES),
        tuple((seed, 0 if seed == 42 else row_count) for seed in SEEDS),
        row_count,
    )
    require(fixed_cascade._matches_exactly(asdict(scoring.counts), asdict(expected)))


def validate_secondary_scoring(
    scoring: secondary.SecondaryScoring, row_count: int
) -> None:
    """Require exact output shape and counters, exposing only a symbolic error."""
    try:
        _validate_secondary_scoring(scoring, row_count)
    except Exception:
        raise SecondaryInputError("invalid_external_secondary_evidence") from None


def validate_column(column: object, index: int, row_count: int) -> None:
    require(0 <= index < len(TABULAR_NAMES) + len(SEEDS))
    if index < len(TABULAR_NAMES):
        require(type(column) is secondary.CompletedTabularColumn)
        require(column.name == TABULAR_NAMES[index])
        require(
            type(column.singleton_calls) is int and column.singleton_calls == row_count
        )
        validate = validate_tabular_score
        identity = TABULAR_NAMES[index]
    else:
        require(type(column) is secondary.CompletedSeedColumn)
        identity = SEEDS[index - len(TABULAR_NAMES)]
        require(type(column.seed) is int and column.seed == identity)
        expected = (0, row_count) if identity == 42 else (row_count, 0)
        observed = (
            column.transformer_singleton_calls,
            column.reused_primary_transformer_scores,
        )
        require(fixed_cascade._matches_exactly(observed, expected))
        validate = validate_seed_score
    require(type(column.scores) is tuple and len(column.scores) == row_count)
    for score in column.scores:
        validate(score, identity)
