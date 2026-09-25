"""Empty external tuples retain the complete inventory without inference."""

from dataclasses import replace

import pytest
from test_bound_secondary import SEEDS, TABULAR_NAMES, _scoring_bound

from automated_phishing_detection import bound_secondary as secondary


@pytest.fixture
def forbidden_scorers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("empty tuple reached a scorer or loader")

    monkeypatch.setattr(
        secondary.SecondaryModel, "score_urls_singleton_ordered", forbidden
    )
    monkeypatch.setattr(
        secondary.secondary_transformer, "load_secondary_transformer_bytes", forbidden
    )
    monkeypatch.setattr(
        secondary.secondary_transformer, "score_secondary_transformer_urls", forbidden
    )
    monkeypatch.setattr(secondary.fixed_cascade, "score_fixed_cascade", forbidden)


@pytest.mark.usefixtures("forbidden_scorers")
def test_empty_tuple_emits_all_empty_columns_with_zero_inventory():
    bound = _scoring_bound(secondary)
    retained = []
    result = secondary.score_bound_secondary(
        bound, (), (), (), on_completed_column=retained.append
    )
    assert result == secondary.SecondaryScoring(
        (),
        secondary.SecondaryInferenceCounts(
            tuple((name, 0) for name in TABULAR_NAMES),
            tuple((seed, 0) for seed in SEEDS),
            0,
        ),
    )
    assert retained == [
        *(secondary.CompletedTabularColumn(name, (), 0) for name in TABULAR_NAMES),
        *(secondary.CompletedSeedColumn(seed, (), 0, 0) for seed in SEEDS),
    ]
    assert secondary.score_bound_secondary(bound, (), (), ()) == result


@pytest.mark.usefixtures("forbidden_scorers")
@pytest.mark.parametrize(
    "stage1,seed42", [((0.5,), ()), ((), (0.4,)), ([], ()), ((), [])]
)
def test_empty_tuple_still_rejects_misaligned_or_non_tuple_vectors(stage1, seed42):
    retained = []
    with pytest.raises(secondary.BoundSecondaryError):
        secondary.score_bound_secondary(
            _scoring_bound(secondary),
            (),
            stage1,
            seed42,
            on_completed_column=retained.append,
        )
    assert retained == []


@pytest.mark.usefixtures("forbidden_scorers")
@pytest.mark.parametrize(
    "field,value",
    [
        ("tabular", ()),
        ("seeds", ()),
        ("stage1_threshold", float("nan")),
        ("vocabulary_bytes", b""),
    ],
)
def test_empty_tuple_still_rejects_invalid_binding(field, value):
    bound = replace(_scoring_bound(secondary), **{field: value})
    retained = []
    with pytest.raises(secondary.BoundSecondaryError):
        secondary.score_bound_secondary(
            bound, (), (), (), on_completed_column=retained.append
        )
    assert retained == []


@pytest.mark.usefixtures("forbidden_scorers")
def test_empty_tuple_still_rejects_a_non_callable_callback():
    with pytest.raises(secondary.BoundSecondaryError, match="callable"):
        secondary.score_bound_secondary(
            _scoring_bound(secondary), (), (), (), on_completed_column=False
        )


@pytest.mark.usefixtures("forbidden_scorers")
@pytest.mark.parametrize("stop_after", [1, 8, 12])
def test_empty_column_callback_failure_stops_at_the_retained_prefix(stop_after):
    retained = []
    failure = RuntimeError("invented empty column consumer failure")

    def callback(column):
        retained.append(column)
        if len(retained) == stop_after:
            raise failure

    with pytest.raises(RuntimeError) as raised:
        secondary.score_bound_secondary(
            _scoring_bound(secondary), (), (), (), on_completed_column=callback
        )
    assert raised.value is failure
    assert len(retained) == stop_after
    assert all(column.scores == () for column in retained)
