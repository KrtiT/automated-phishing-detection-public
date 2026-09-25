"""A late member or consumer failure leaves only completed columns retained."""

from asyncio import CancelledError
from functools import partial

import pytest
from bound_secondary_column_fixtures import (
    expected_events,
    identity,
    retain,
    score,
    secondary,
)
from bound_secondary_column_fixtures import scoring as scoring
from test_bound_secondary import SEEDS, TABULAR_NAMES


@pytest.mark.parametrize("failure", [("tabular", "random_forest"), ("score", 45)])
@pytest.mark.parametrize("invalid_output", [False, True])
def test_late_member_failure_retains_only_complete_prefix(
    scoring, failure, invalid_output
):
    scoring.fail_at = failure
    scoring.invalid_output = invalid_output
    with pytest.raises(secondary.BoundSecondaryError):
        score(scoring, partial(retain, scoring))
    events = expected_events()
    assert scoring.events == events[: events.index(failure) + 1]
    expected_prefix = (*TABULAR_NAMES, *SEEDS)
    completed = expected_prefix[: expected_prefix.index(failure[1])]
    assert tuple(identity(column) for column in scoring.retained) == completed
    assert all(len(column.scores) == 2 for column in scoring.retained)
    assert scoring.singletons[-2:] == [(*failure, 0), (*failure, 1)]
    for column in scoring.retained:
        if hasattr(column, "singleton_calls"):
            assert column.singleton_calls == 2
        else:
            assert column.transformer_singleton_calls == (0 if column.seed == 42 else 2)
            assert column.reused_primary_transformer_scores == (
                2 if column.seed == 42 else 0
            )


@pytest.mark.parametrize(
    "exception_type", [RuntimeError, ValueError, KeyboardInterrupt, CancelledError]
)
@pytest.mark.parametrize("stop_at", ["random_forest", 42, 45])
def test_callback_failure_preserves_retained_prefix_and_stops_immediately(
    scoring, exception_type, stop_at
):
    failure = exception_type("invented consumer failure")

    def callback(column):
        retain(scoring, column)
        if identity(column) == stop_at:
            raise failure

    with pytest.raises(exception_type) as raised:
        score(scoring, callback)
    assert raised.value is failure
    events = expected_events()
    assert scoring.events == events[: events.index(("completed", stop_at)) + 1]
    expected_prefix = (*TABULAR_NAMES, *SEEDS)
    assert (
        tuple(identity(column) for column in scoring.retained)
        == expected_prefix[: expected_prefix.index(stop_at) + 1]
    )
    assert all(len(column.scores) == 2 for column in scoring.retained)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, CancelledError])
def test_model_interruption_preserves_previous_columns(
    scoring, monkeypatch, exception_type
):
    interruption = exception_type("invented scorer interruption")

    def interrupted(loaded, urls):
        scoring.events.append(("score", loaded.seed))
        raise interruption

    monkeypatch.setattr(
        secondary.secondary_transformer, "score_secondary_transformer_urls", interrupted
    )
    with pytest.raises(exception_type) as raised:
        score(scoring, partial(retain, scoring))
    assert raised.value is interruption
    events = expected_events()
    assert scoring.events == events[: events.index(("score", 43)) + 1]
    assert tuple(identity(column) for column in scoring.retained) == (
        *TABULAR_NAMES,
        42,
    )


def test_cascade_failure_does_not_publish_the_seed_column(scoring, monkeypatch):
    def rejected(*args, **kwargs):
        scoring.events.append(("cascade", 42))
        raise secondary.fixed_cascade.FixedCascadeError("invented cascade failure")

    monkeypatch.setattr(secondary.fixed_cascade, "score_fixed_cascade", rejected)
    with pytest.raises(secondary.BoundSecondaryError):
        score(scoring, partial(retain, scoring))
    events = expected_events()
    assert scoring.events == events[: events.index(("cascade", 42)) + 1]
    assert tuple(identity(column) for column in scoring.retained) == TABULAR_NAMES
