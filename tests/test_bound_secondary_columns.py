"""Completed score columns are immutable and precede the next member."""

from dataclasses import FrozenInstanceError, fields
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


def test_callbacks_preserve_legacy_results_and_complete_in_model_order(scoring):
    legacy = secondary.score_bound_secondary(
        scoring.bound, scoring.urls, scoring.stage1, scoring.seed42
    )
    legacy_singletons = tuple(scoring.singletons)
    scoring.events.clear()
    scoring.singletons.clear()
    result = score(scoring, partial(retain, scoring))
    assert result == legacy
    assert tuple(scoring.singletons) == legacy_singletons
    assert scoring.events == expected_events()
    assert [identity(column) for column in scoring.retained] == [
        *TABULAR_NAMES,
        *SEEDS,
    ]
    for index, column in enumerate(scoring.retained[:7]):
        assert column == secondary.CompletedTabularColumn(
            TABULAR_NAMES[index], tuple(row.tabular[index] for row in result.rows), 2
        )
    for index, column in enumerate(scoring.retained[7:]):
        assert column == secondary.CompletedSeedColumn(
            SEEDS[index],
            tuple(row.seeds[index] for row in result.rows),
            0 if index == 0 else 2,
            2 if index == 0 else 0,
        )


@pytest.mark.parametrize("column_index", [0, 7, 11])
def test_completed_payloads_are_typed_frozen_and_detached(scoring, column_index):
    result = score(scoring, partial(retain, scoring))
    column = scoring.retained[column_index]
    expected_type = (
        secondary.CompletedTabularColumn
        if column_index < 7
        else secondary.CompletedSeedColumn
    )
    score_type = (
        secondary.SecondaryTabularScore
        if column_index < 7
        else secondary.SecondarySeedScore
    )
    assert type(column) is expected_type and type(column.scores) is tuple
    assert all(type(value) is score_type for value in column.scores)
    with pytest.raises(FrozenInstanceError):
        column.scores = ()
    with pytest.raises(FrozenInstanceError):
        column.scores[0].decision = 0
    with pytest.raises(TypeError):
        column.scores[0] = column.scores[1]
    scoring.urls = ()
    del result
    assert len(column.scores) == 2
    assert not {"urls", "raw_urls", "labels", "row_ids", "source_ids"}.intersection(
        field.name for field in fields(column)
    )


def test_completed_payload_fields_expose_only_column_scores_and_work(scoring):
    score(scoring, partial(retain, scoring))
    assert [field.name for field in fields(scoring.retained[0])] == [
        "name",
        "scores",
        "singleton_calls",
    ]
    assert [field.name for field in fields(scoring.retained[7])] == [
        "seed",
        "scores",
        "transformer_singleton_calls",
        "reused_primary_transformer_scores",
    ]


@pytest.mark.parametrize("callback", [False, 0, "callback", [], object()])
def test_malformed_callbacks_fail_before_every_inference(scoring, callback):
    with pytest.raises(secondary.BoundSecondaryError, match="callable"):
        score(scoring, callback)
    assert scoring.events == []
    assert scoring.singletons == []


def test_explicit_none_preserves_legacy_results(scoring):
    result = score(scoring, None)
    assert result == secondary.score_bound_secondary(
        scoring.bound, scoring.urls, scoring.stage1, scoring.seed42
    )


def test_callback_is_keyword_only(scoring):
    with pytest.raises(TypeError):
        secondary.score_bound_secondary(
            scoring.bound,
            scoring.urls,
            scoring.stage1,
            scoring.seed42,
            lambda unused_column: None,
        )
    assert scoring.events == []
