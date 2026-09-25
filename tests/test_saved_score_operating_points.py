"""Primary and every secondary family retain distinct saved operating points."""

from saved_score_fixtures import saved_scores as saved_scores
from saved_score_fixtures import validate


def _primary(row, bindings):
    bindings["thresholds"].update(
        length_only=0.8, logistic_l1=0.4, transformer=0.7, half_width=0.01
    )
    row.update(
        length_probability=0.3,
        length_decision=0,
        stage1_probability=0.45,
        stage1_decision=1,
        transformer_probability=0.6,
        transformer_decision=0,
        cascade_probability=0.45,
        cascade_decision=1,
        band_selected=False,
    )


def _secondary(row, bindings):
    secondary = bindings["secondary"]
    secondary["stage1_threshold"] = 0.4
    for index, (point, score) in enumerate(
        zip(secondary["tabular"], row["secondary_tabular"], strict=True)
    ):
        point["threshold"] = (index + 1) / 10
        score.update(probability=0.4, decision=int(index < 4))
    for index, (point, score) in enumerate(
        zip(secondary["seeds"], row["secondary_seeds"], strict=True)
    ):
        point["transformer_threshold"] = (index + 1) / 5
        point["half_width"] = (0.0, 0.02, 0.05, 0.1, 0.2)[index]
        invoked = index >= 2
        score.update(
            transformer_probability=0.6,
            transformer_decision=int(index < 3),
            cascade_probability=0.6 if invoked else 0.45,
            cascade_decision=int(index < 3) if invoked else 1,
            band_selected=invoked,
        )


def test_primary_tabular_and_seed_cutoffs_are_not_interchangeable(saved_scores):
    rows, bindings = saved_scores
    row = rows[0]
    _primary(row, bindings)
    _secondary(row, bindings)
    assert validate(row, bindings) is None
