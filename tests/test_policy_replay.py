"""Synthetic cached predictions exercise routing, not model performance."""

import inspect
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from automated_phishing_detection import gmm_monitor
from automated_phishing_detection.policy_replay import (
    MonitorScore,
    PairedProbabilities,
    PolicyReplayError,
    replay_policy,
)


def _inputs(count, *, nll=2.0):
    predictions = tuple(
        PairedProbabilities(f"synthetic-{index:04d}", 0.125, 0.75)
        for index in range(count)
    )
    monitor = tuple(MonitorScore(row.record_id, nll) for row in predictions)
    return predictions, monitor


def _replay(predictions, monitor, **configuration):
    return replay_policy(
        predictions,
        monitor,
        **{
            "stage1_threshold": 0.5,
            "transformer_threshold": 0.75,
            "half_width": 0.125,
            "monitor_boundary": 1.0,
            **configuration,
        },
    )


@pytest.mark.parametrize("count", [0, 1, 255])
def test_short_stream_has_no_complete_windows_or_detection_estimate(count):
    predictions, monitor = _inputs(count)
    result = _replay(predictions, monitor)

    assert len(result.rows) == count
    assert result.windows == ()
    assert result.window_alert_fraction is None
    assert not any(row.drift_override for row in result.rows)
    assert not any(row.logical_stage2_mask for row in result.rows)


@pytest.mark.parametrize("count", [256, 257, 512])
def test_first_alert_routes_only_subsequent_requests(count):
    result = _replay(*_inputs(count))

    assert result.windows[0].end_position == 256
    assert result.windows[0].alert is True
    assert not any(row.drift_override for row in result.rows[:256])
    assert all(row.drift_override for row in result.rows[256:])
    assert all(row.fixed_decision == 0 for row in result.rows)
    assert all(row.policy_decision == 0 for row in result.rows[:256])
    assert all(row.policy_decision == 1 for row in result.rows[256:])


def test_window_scores_are_exact_existing_complete_window_means():
    predictions, monitor = _inputs(601)
    monitor = tuple(
        replace(row, negative_log_likelihood=float(index % 19) / 7.0)
        for index, row in enumerate(monitor)
    )
    result = _replay(predictions, monitor)
    ends, scores = gmm_monitor.window_scores(
        [row.negative_log_likelihood for row in monitor]
    )

    assert tuple(window.end_position for window in result.windows) == ends
    assert tuple(window.score for window in result.windows) == scores
    assert tuple(window.start_position for window in result.windows) == tuple(
        end - 255 for end in ends
    )
    assert ends == (256, 320, 384, 448, 512, 576)


def test_score_equal_to_monitor_boundary_does_not_alert():
    result = _replay(*_inputs(512, nll=1.0))

    assert all(window.score == 1.0 for window in result.windows)
    assert not any(window.alert for window in result.windows)
    assert result.window_alert_fraction == 0.0
    assert not any(row.drift_override for row in result.rows)


def test_one_alert_expires_after_exactly_256_future_requests():
    predictions, monitor = _inputs(600)
    monitor = tuple(
        replace(row, negative_log_likelihood=2.0 if index < 64 else 0.0)
        for index, row in enumerate(monitor)
    )
    result = _replay(predictions, monitor, monitor_boundary=0.25)

    assert [window.end_position for window in result.windows if window.alert] == [256]
    assert [
        index + 1 for index, row in enumerate(result.rows) if row.drift_override
    ] == (list(range(257, 513)))
    assert result.rows[512].policy_decision == 0
    assert result.window_alert_fraction == 1 / 6


def test_overlapping_overrides_are_unioned_with_fixed_band_once():
    predictions, monitor = _inputs(600)
    predictions = tuple(
        replace(row, stage1_probability=0.5) if index % 2 == 0 else row
        for index, row in enumerate(predictions)
    )
    result = _replay(predictions, monitor)

    assert len(result.rows) == 600
    assert sum(row.drift_override for row in result.rows) == 344
    assert sum(row.logical_stage2_mask for row in result.rows) == 472
    assert all(
        row.logical_stage2_mask == (row.logical_band or row.drift_override)
        for row in result.rows
    )
    assert result.window_alert_fraction == 1.0


def test_incomplete_terminal_window_is_not_scored_and_last_override_is_truncated():
    result = _replay(*_inputs(400))

    assert [window.end_position for window in result.windows] == [256, 320, 384]
    assert len(result.rows) == 400
    assert result.rows[-1].drift_override is True


def test_fixed_band_is_inclusive_and_both_decisions_include_threshold_equality():
    predictions = (
        PairedProbabilities("lower", 0.375, 0.75),
        PairedProbabilities("upper", 0.625, np.nextafter(0.75, 0.0).item()),
        PairedProbabilities("outside-low", 0.25, 0.75),
        PairedProbabilities("outside-high", 0.75, 0.0),
    )
    monitor = tuple(MonitorScore(row.record_id, -3.0) for row in predictions)
    result = _replay(predictions, monitor)

    assert [row.logical_band for row in result.rows] == [True, True, False, False]
    assert [row.fixed_decision for row in result.rows] == [1, 0, 0, 1]
    assert [row.policy_decision for row in result.rows] == [1, 0, 0, 1]


def test_zero_width_includes_stage_one_threshold_equality():
    predictions = (PairedProbabilities("at-threshold", 0.5, 0.125),)
    monitor = (MonitorScore("at-threshold", 0.0),)
    row = _replay(predictions, monitor, half_width=0.0).rows[0]

    assert row.logical_band is True
    assert row.fixed_decision == 0


def test_trace_preserves_exact_order_and_omits_raw_predictions():
    predictions, monitor = _inputs(257)
    result = _replay(predictions, monitor)

    assert tuple(row.record_id for row in result.rows) == tuple(
        row.record_id for row in predictions
    )
    assert set(asdict(result.rows[0])) == {
        "record_id",
        "fixed_decision",
        "policy_decision",
        "logical_band",
        "drift_override",
        "logical_stage2_mask",
    }


def test_interface_is_label_and_source_blind():
    assert set(inspect.signature(replay_policy).parameters) == {
        "probabilities",
        "monitor_scores",
        "stage1_threshold",
        "transformer_threshold",
        "half_width",
        "monitor_boundary",
    }
    assert {field.name for field in fields(PairedProbabilities)} == {
        "record_id",
        "stage1_probability",
        "transformer_probability",
    }
    assert {field.name for field in fields(MonitorScore)} == {
        "record_id",
        "negative_log_likelihood",
    }


@pytest.mark.parametrize("change", ["missing", "extra", "reordered", "different"])
def test_rejects_monitor_alignment_instead_of_repairing_it(change):
    predictions, monitor = _inputs(3)
    if change == "missing":
        monitor = monitor[:-1]
    elif change == "extra":
        monitor += (MonitorScore("extra", 0.0),)
    elif change == "reordered":
        monitor = monitor[::-1]
    else:
        monitor = (replace(monitor[0], record_id="different"), *monitor[1:])
    with pytest.raises(PolicyReplayError, match="counts|IDs or order"):
        _replay(predictions, monitor)


@pytest.mark.parametrize("identity", ["", "bad id", "bad\n", None, 123])
def test_rejects_malformed_record_ids(identity):
    with pytest.raises(PolicyReplayError, match="record ID"):
        _replay(
            (PairedProbabilities(identity, 0.25, 0.75),),
            (MonitorScore(identity, 0.0),),
        )


def test_rejects_matching_duplicate_ids():
    predictions, monitor = _inputs(1)
    with pytest.raises(PolicyReplayError, match="unique"):
        _replay(predictions * 2, monitor * 2)


@pytest.mark.parametrize("bad", [None, "records", {}, iter(())])
@pytest.mark.parametrize("field", ["probabilities", "monitor_scores"])
def test_requires_materialized_ordered_sequences(field, bad):
    values = dict(zip(("probabilities", "monitor_scores"), _inputs(0)))
    values[field] = bad
    with pytest.raises(PolicyReplayError, match="ordered sequences"):
        replay_policy(
            **values,
            stage1_threshold=0.5,
            transformer_threshold=0.75,
            half_width=0.125,
            monitor_boundary=1.0,
        )


@pytest.mark.parametrize("field", ["predictions", "monitor"])
def test_rejects_untyped_record_mappings(field):
    predictions, monitor = _inputs(1)
    if field == "predictions":
        predictions = (asdict(predictions[0]),)
    else:
        monitor = (asdict(monitor[0]),)
    with pytest.raises(PolicyReplayError, match="typed score records"):
        _replay(predictions, monitor)


@pytest.mark.parametrize("field", ["stage1_probability", "transformer_probability"])
@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_rejects_invalid_probabilities(field, bad):
    predictions, monitor = _inputs(1)
    predictions = (replace(predictions[0], **{field: bad}),)
    with pytest.raises(PolicyReplayError, match=field):
        _replay(predictions, monitor)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.0", None])
def test_rejects_invalid_monitor_scores(bad):
    predictions, monitor = _inputs(1)
    monitor = (replace(monitor[0], negative_log_likelihood=bad),)
    with pytest.raises(PolicyReplayError, match="negative_log_likelihood"):
        _replay(predictions, monitor)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize(
    "field,bad",
    [
        ("stage1_threshold", -0.1),
        ("stage1_threshold", 1.1),
        ("transformer_threshold", float("nan")),
        ("transformer_threshold", True),
        ("half_width", -0.1),
        ("half_width", float("inf")),
        ("monitor_boundary", float("nan")),
        ("monitor_boundary", True),
    ],
)
def test_rejects_invalid_configuration_even_for_empty_input(count, field, bad):
    with pytest.raises(PolicyReplayError, match=field):
        _replay(*_inputs(count), **{field: bad})


def test_validates_terminal_rows_before_computing_any_window(monkeypatch):
    predictions, monitor = _inputs(513)
    monitor = (
        *monitor[:-1],
        replace(monitor[-1], negative_log_likelihood=float("nan")),
    )

    def forbidden_window_call(*args, **kwargs):
        pytest.fail("window computation preceded complete input validation")

    monkeypatch.setattr(gmm_monitor, "window_scores", forbidden_window_call)
    with pytest.raises(PolicyReplayError, match="negative_log_likelihood"):
        _replay(predictions, monitor)


def test_window_overflow_is_rejected_without_partial_output():
    with pytest.raises(PolicyReplayError, match="window"):
        _replay(*_inputs(256, nll=float(np.finfo(np.float64).max)))
