from dataclasses import replace

import numpy as np
import pytest
from external_replay_fixtures import inputs, relink_primary, replay_module

from automated_phishing_detection import external_monitors


@pytest.mark.parametrize(
    "count,ends", [(0, ()), (255, ()), (256, (256,)), (319, (256,)), (320, (256, 320))]
)
def test_replay_keeps_full_stream_window_positions(monkeypatch, count, ends):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, count)
    result = module.replay_external_scores(primary, secondary, reference)
    assert len(result.rows) == count
    assert tuple(row.record for row in result.rows) == primary.records
    assert tuple(row.primary for row in result.rows) == primary.scores
    assert tuple(monitor.name for monitor in result.monitors) == ("gmm", "mmd", "psi")
    for monitor in result.monitors:
        assert tuple(window.end_position for window in monitor.windows) == ends
        assert tuple(window.start_position for window in monitor.windows) == tuple(
            end - 255 for end in ends
        )
    assert result.evidence.external_windows.complete_windows == len(ends)
    assert result.evidence.external_windows.alert_windows == len(ends)
    if not ends:
        assert result.monitors[0].reason == "no_complete_256_row_window"


def test_secondary_and_control_prefix_routes_later_gold_before_filtering(monkeypatch):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch)
    result = module.replay_external_scores(primary, secondary, reference)
    assert {row.record.role for row in result.rows[:256]} == {"secondary", "tranco"}
    assert all(not row.drift_override for row in result.rows[:256])
    gold = result.rows[256]
    assert gold.record.role == "gold"
    assert gold.primary.cascade_decision == 0
    assert gold.policy_decision == 1
    assert gold.policy_probability == gold.primary.transformer_probability
    assert gold.drift_override and gold.logical_stage2_mask
    assert result.evidence.populations["gold"].predictions["policy"][0].decision == 1
    assert set(result.evidence.controls.predictions) == {"cascade", "transformer"}
    assert len(result.evidence.controls.record_ids) == 128
    assert all(
        row.record.is_phishing is None
        for row in result.rows
        if row.record.role == "tranco"
    )


def test_replay_joins_all_members_and_exact_retained_monitor_values(monkeypatch):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch)
    result = module.replay_external_scores(primary, secondary, reference)
    expected = external_monitors.replay_external_monitors(primary.scores, reference)
    assert result.monitors[1:] == expected.monitors
    for row, members, standardized in zip(
        result.rows, secondary.rows, expected.standardized_features
    ):
        assert row.secondary_tabular == members.tabular
        assert row.secondary_seeds == members.seeds
        assert row.standardized_monitor_features == standardized
        expected_vector = (
            np.asarray((*row.primary.features, row.primary.monitor_probability))
            - reference.scaler_mean
        ) / reference.scaler_scale
        assert np.array_equal(expected_vector, standardized)


def test_replay_never_rethresholds_heterogeneous_policy_probability(monkeypatch):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, 1)
    score = replace(
        primary.scores[0],
        stage1_probability=0.6,
        stage1_decision=0,
        cascade_probability=0.6,
        cascade_decision=0,
        band_selected=False,
    )
    thresholds = dict(primary.thresholds) | {"logistic_l1": 0.8, "half_width": 0.0}
    primary = relink_primary(primary, scores=(score,), thresholds=thresholds)
    members = secondary.rows[0]
    seeds = tuple(
        replace(seed, cascade_probability=0.6, cascade_decision=0)
        for seed in members.seeds
    )
    secondary = replace(secondary, rows=(replace(members, seeds=seeds),))
    result = module.replay_external_scores(primary, secondary, reference)
    assert result.rows[0].policy_probability == 0.6
    assert result.rows[0].policy_decision == 0
    assert result.rows[0].policy_decision != int(
        result.rows[0].policy_probability >= 0.5
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cascade_decision", 1),
        ("band_selected", True),
        ("stage1_decision", 1),
        ("transformer_decision", 0),
        ("cascade_probability", 0.99),
    ],
)
def test_replay_rejects_primary_decision_or_route_mismatch(monkeypatch, field, value):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, 1)
    primary = relink_primary(
        primary, scores=(replace(primary.scores[0], **{field: value}),)
    )
    with pytest.raises(module.ExternalReplayError, match="invalid_external_replay"):
        module.replay_external_scores(primary, secondary, reference)


@pytest.mark.parametrize(
    "fault", ["row_count", "family_order", "counts", "invalid_reference"]
)
def test_replay_rejects_incomplete_or_misaligned_evidence(monkeypatch, fault):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, 2)
    if fault == "row_count":
        secondary = replace(secondary, rows=secondary.rows[:1])
    elif fault == "family_order":
        secondary = replace(
            secondary,
            rows=(
                replace(secondary.rows[0], seeds=secondary.rows[0].seeds[::-1]),
                secondary.rows[1],
            ),
        )
    elif fault == "counts":
        secondary = replace(
            secondary,
            counts=replace(secondary.counts, reused_primary_transformer_scores=0),
        )
    else:
        reference = None
    with pytest.raises(module.ExternalReplayError, match="invalid_external_replay"):
        module.replay_external_scores(primary, secondary, reference)


def test_reused_seed42_probability_must_match_primary_observation(monkeypatch):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, 1)
    row = secondary.rows[0]
    forged = replace(row.seeds[0], transformer_probability=0.123)
    secondary = replace(secondary, rows=(replace(row, seeds=(forged, *row.seeds[1:])),))
    with pytest.raises(module.ExternalReplayError, match="invalid_external_replay"):
        module.replay_external_scores(primary, secondary, reference)


@pytest.mark.parametrize(
    "field,value", [("cascade_probability", 0.999), ("cascade_decision", 1)]
)
def test_seed_cascade_must_select_saved_component(monkeypatch, field, value):
    module = replay_module()
    primary, secondary, reference = inputs(monkeypatch, 1)
    row = secondary.rows[0]
    forged = replace(row.seeds[1], **{field: value})
    secondary = replace(
        secondary, rows=(replace(row, seeds=(row.seeds[0], forged, *row.seeds[2:])),)
    )
    with pytest.raises(module.ExternalReplayError, match="invalid_external_replay"):
        module.replay_external_scores(primary, secondary, reference)
