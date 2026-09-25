"""Pure saved score algebra, independent of internal or external record policy."""

from . import fixed_cascade
from .bound_secondary import (
    _SEEDS,
    _TABULAR_NAMES,
    SecondarySeedScore,
    SecondaryTabularScore,
)
from .url_features import extract_url_features

_TABULAR_FIELDS = frozenset(SecondaryTabularScore.__dataclass_fields__)
_SEED_FIELDS = frozenset(SecondarySeedScore.__dataclass_fields__)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid_saved_score_values")


def _probability(value: object) -> float:
    result = fixed_cascade._finite_number(value, "saved probability")
    _require(0 <= result <= 1)
    return result


def _decision(value: object) -> int:
    _require(type(value) is int and value in (0, 1))
    return value


def _features(row: dict) -> None:
    record, features = row["record"], row["features"]
    _require(type(record) is dict and type(record["raw_url"]) is str)
    _require(type(features) is list and len(features) == 25)
    values = tuple(
        fixed_cascade._finite_number(value, "saved feature") for value in features
    )
    _require(values == extract_url_features(record["raw_url"]))


def _component(row: dict, prefix: str, threshold: float) -> float:
    probability = _probability(row[f"{prefix}_probability"])
    _require(_decision(row[f"{prefix}_decision"]) == int(probability >= threshold))
    return probability


def _cascade(
    row: dict,
    stage1: float,
    transformer: float,
    stage1_threshold: float,
    transformer_threshold: float,
    half_width: float,
) -> None:
    expected = fixed_cascade.score_fixed_cascade(
        (stage1,),
        (transformer,),
        stage1_threshold=stage1_threshold,
        transformer_threshold=transformer_threshold,
        half_width=half_width,
    )
    invoked = expected.transformer_invoked[0]
    _require(
        _decision(row["cascade_decision"]) == expected.decisions[0]
        and type(row["band_selected"]) is bool
        and row["band_selected"] is invoked
        and _probability(row["cascade_probability"])
        == (transformer if invoked else stage1)
    )


def _primary(row: dict, thresholds: dict) -> tuple[float, float]:
    _component(row, "length", thresholds["length_only"])
    stage1 = _component(row, "stage1", thresholds["logistic_l1"])
    transformer = _component(row, "transformer", thresholds["transformer"])
    _cascade(
        row,
        stage1,
        transformer,
        thresholds["logistic_l1"],
        thresholds["transformer"],
        thresholds["half_width"],
    )
    _probability(row["monitor_probability"])
    fixed_cascade._finite_number(row["negative_log_likelihood"], "saved likelihood")
    return stage1, transformer


def _tabular(scores: list, points: list) -> None:
    _require(type(scores) is list and len(scores) == len(_TABULAR_NAMES))
    for point, score in zip(points, scores, strict=True):
        _require(
            type(score) is dict
            and set(score) == _TABULAR_FIELDS
            and score["name"] == point["name"]
        )
        probability = _probability(score["probability"])
        _require(_decision(score["decision"]) == int(probability >= point["threshold"]))


def _seed(
    score: dict, point: dict, stage1: float, primary: float, stage1_threshold: float
) -> None:
    _require(
        type(score) is dict
        and set(score) == _SEED_FIELDS
        and type(score["seed"]) is int
        and score["seed"] == point["seed"]
    )
    transformer = _component(score, "transformer", point["transformer_threshold"])
    if point["seed"] == 42:
        _require(transformer == primary)
    _cascade(
        score,
        stage1,
        transformer,
        stage1_threshold,
        point["transformer_threshold"],
        point["half_width"],
    )


def validate_score_values(row: dict, bindings: dict) -> None:
    """Check features and every frozen operating-point relation without inference."""
    _require(type(row) is dict and type(bindings) is dict)
    _features(row)
    stage1, transformer = _primary(row, bindings["thresholds"])
    secondary = bindings["secondary"]
    _tabular(row["secondary_tabular"], secondary["tabular"])
    seeds = row["secondary_seeds"]
    _require(type(seeds) is list and len(seeds) == len(_SEEDS))
    for point, score in zip(secondary["seeds"], seeds, strict=True):
        _seed(score, point, stage1, transformer, secondary["stage1_threshold"])
