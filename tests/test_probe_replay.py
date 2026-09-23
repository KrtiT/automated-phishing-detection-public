"""Invented, in-memory replay inputs; no research data, fits, or calibration."""

import copy
import importlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from automated_phishing_detection import gmm_monitor as gmm
from automated_phishing_detection import secondary_drift as drift
from automated_phishing_detection.secondary_probes import PERTURBATION_OPERATORS
from automated_phishing_detection.url_features import extract_url_features


def test_replay_module_exists():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/automated_phishing_detection/probe_replay.py"
    )
    assert path.is_file(), "missing label-free secondary probe replay"


@pytest.fixture
def replay():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/automated_phishing_detection/probe_replay.py"
    )
    assert path.is_file(), "missing label-free secondary probe replay"
    return importlib.import_module("automated_phishing_detection.probe_replay")


@dataclass(frozen=True)
class PortableModel:
    probability: float = 0.25

    def score_urls(self, urls):
        assert type(urls) is tuple and len(urls) == 1
        return (self.probability,)


def artifact():
    return {
        "schema_version": 1,
        "contract_id": gmm.CONTRACT_ID,
        "features": list(gmm.GMM_FEATURE_NAMES),
        "dtype": "float64",
        "scaler": {
            "mean": [2.0] * 26,
            "scale": [3.0] * 26,
            "variance": [9.0] * 26,
            "n_samples_seen": 256,
            "n_features_in": 26,
        },
        "mixture": {
            "components": 1,
            "covariance_type": "diag",
            "weights": [1.0],
            "means": [[0.0] * 26],
            "variances": [[1.0] * 26],
            "precisions": [[1.0] * 26],
            "precisions_cholesky": [[1.0] * 26],
            "converged": True,
            "n_iter": 1,
            "lower_bound": -1.0,
        },
        "input_hashes": {},
    }


def frozen_references():
    values = tuple((float(i % 2),) + (0.0,) * 25 for i in range(256))
    mmd = drift.MMDReference(
        values,
        tuple(f"domain-{i}.example" for i in range(256)),
        tuple(f"reference-{i}" for i in range(256)),
        1.0,
    )
    feature = drift.PSIFeatureReference(
        (), 0.0, (0, 256, 0), (0.5 / 257.5, 256.5 / 257.5, 0.5 / 257.5)
    )
    return mmd, drift.PSIReference((feature,) * 26, 256)


def records(replay, count=4):
    return tuple(
        replay.AuditInput(f"row-{i}", i * 2, "https://example.com/a%2f")
        for i in range(count)
    )


def primary_scorer(replay):
    def score(row):
        index = int(row.record_id.removeprefix("row-"))
        first = (0.2, 0.8, 0.45, 0.55)[index % 4]
        if row.raw_url.startswith("HTTPS"):
            first = 1.0 - first
        second = (0.9, 0.1, 0.8, 0.2)[index % 4]
        return replay.PrimaryScores(
            row.record_id,
            row.raw_url,
            len(row.raw_url) / 100,
            first,
            second,
            '{"singleton":true}',
            '{"singleton":true}',
        )

    return score


def kwargs(replay):
    mmd, psi = frozen_references()
    return {
        "primary_scorer": primary_scorer(replay),
        "stage1_model": PortableModel(),
        "gmm_artifact": artifact(),
        "operating_points": replay.OperatingPoints(0.5, 0.5, 0.5, 0.1, -1.0),
        "mmd_reference": mmd,
        "mmd_calibration": drift.DriftCalibration(0.0, 2),
        "psi_reference": psi,
        "psi_calibration": drift.DriftCalibration(0.0, 2),
    }


def test_four_noncomposed_streams_exact_mapping_and_no_labels(replay):
    inputs = records(replay)
    result = replay.replay_probes(inputs, **kwargs(replay))
    assert tuple(stream.name for stream in result.streams) == (
        "original",
        *PERTURBATION_OPERATORS,
    )
    assert replay.DETECTOR_NAMES == (
        "length",
        "logistic_l1",
        "transformer_42",
        "fixed_cascade",
        "gmm_policy",
    )
    assert replay.MONITOR_NAMES == ("gmm", "mmd", "psi")
    expected = (
        "https://example.com/a%2f",
        "HTTPS://EXAMPLE.COM/a%2f",
        "https://example.com/a%2F",
        "https://example.com/%61%2f",
    )
    for stream, output_url in zip(result.streams, expected, strict=True):
        assert len(stream.rows) == len(inputs)
        for index, row in enumerate(stream.rows):
            mapping = row.mapping
            assert (
                mapping.record_id,
                mapping.validation_position,
                mapping.stream_position,
            ) == (f"row-{index}", index * 2, index + 1)
            assert mapping.original_url == inputs[index].raw_url
            assert mapping.output_url == output_url
            assert mapping.changed == (output_url != mapping.original_url)
            assert len(row.probabilities) == len(row.decisions) == 5
    public = result.public_summary
    assert public["contract_id"] == "secondary-seed-probe-v1"
    assert public["scope"] == "in_memory_descriptive_only"
    encoded = json.dumps(public, allow_nan=False)
    for forbidden in (
        "example.com",
        "row-",
        "is_phishing",
        "recall",
        "average_precision",
        "attack_success",
        "false_alert_gate_met",
    ):
        assert forbidden not in encoded
    assert set(asdict(inputs[0])) == {"record_id", "validation_position", "raw_url"}


def test_portable_monitor_probability_and_real_gmm_arithmetic(replay):
    options = kwargs(replay)
    result = replay.replay_probes(records(replay, 1), **options)
    row = result.streams[0].rows[0]
    features = extract_url_features(row.mapping.output_url)
    expected = (np.array((*features, 0.25)) - 2) / 3
    assert row.structural_features == features
    assert row.portable_monitor_probability == 0.25
    assert row.probabilities[1] == 0.2
    np.testing.assert_array_equal(row.standardized_monitor_features, expected)
    expected_nll = 13 * math.log(2 * math.pi) + 0.5 * float(expected @ expected)
    assert row.negative_log_likelihood == pytest.approx(expected_nll, abs=1e-12)


def test_replay_complete_windows_reset_future_only_and_full_psi(replay):
    result = replay.replay_probes(records(replay, 321), **kwargs(replay))
    for stream in result.streams:
        gmm_windows, mmd_windows, psi_windows = (
            monitor.windows for monitor in stream.monitors
        )
        assert [(w.start_position, w.end_position) for w in gmm_windows] == [
            (1, 256),
            (65, 320),
        ]
        assert all(w.alert for w in gmm_windows)
        assert not any(row.drift_override for row in stream.rows[:256])
        assert all(row.drift_override for row in stream.rows[256:])
        for row in stream.rows:
            assert row.logical_stage2_mask == (row.logical_band or row.drift_override)
            expected = (
                row.probabilities[2]
                if row.logical_stage2_mask
                else row.probabilities[1]
            )
            assert row.probabilities[4] == expected
            assert row.decisions[4] == int(expected >= 0.5)
        matrix = tuple(row.standardized_monitor_features for row in stream.rows)
        options = kwargs(replay)
        expected_mmd = drift.mmd_window_scores(options["mmd_reference"], matrix)
        expected_psi = drift.psi_window_scores(options["psi_reference"], matrix)
        assert tuple(w.score for w in mmd_windows) == expected_mmd.scores
        assert tuple(w.score for w in psi_windows) == expected_psi.scores
        assert (
            tuple(w.feature_scores for w in psi_windows) == expected_psi.feature_scores
        )
        assert all(len(w.feature_scores) == 26 for w in psi_windows)


def test_strict_monitor_equality_never_alerts(replay):
    options = kwargs(replay)
    first = replay.replay_probes(records(replay, 256), **options)
    original = first.streams[0]
    options["operating_points"] = replace(
        options["operating_points"],
        monitor_boundary=original.monitors[0].windows[0].score,
    )
    options["mmd_calibration"] = drift.DriftCalibration(
        original.monitors[1].windows[0].score, 2
    )
    options["psi_calibration"] = drift.DriftCalibration(
        original.monitors[2].windows[0].score, 2
    )
    second = replay.replay_probes(records(replay, 257), **options)
    assert all(m.windows[0].alert is False for m in second.streams[0].monitors)
    assert second.streams[0].rows[-1].drift_override is False


@pytest.mark.parametrize("count", [0, 1, 255])
def test_short_streams_explicitly_not_estimable_never_fake_zero_pass(replay, count):
    result = replay.replay_probes(records(replay, count), **kwargs(replay))
    for stream in result.public_summary["streams"]:
        for monitor in stream["monitors"].values():
            assert monitor["window_count"] == 0
            assert monitor["score_mean"] is None
            assert monitor["alert_count"] is None
            assert monitor["alert_fraction"] is None
            assert monitor["reason"] == "no_complete_256_row_window"
        for monitor in stream["paired_with_original"]["monitors"].values():
            assert monitor["score_differences"] is None
            assert monitor["alert_transitions"] is None
            assert monitor["reason"] == "no_complete_256_row_window"


def test_full_precision_pairing_counts_noops_and_means(replay):
    inputs = (
        replay.AuditInput("row-0", 0, "HTTPS://EXAMPLE.COM/%2F"),
        replay.AuditInput("row-1", 4, "https://example.com"),
        replay.AuditInput("row-2", 9, "https://example.com/a%2f"),
    )
    result = replay.replay_probes(inputs, **kwargs(replay))
    summaries = result.public_summary["streams"]
    assert summaries[1]["mapping_counts"] == {
        "eligible": 3,
        "changed": 2,
        "eligible_noop": 1,
        "ineligible": 0,
    }
    assert summaries[2]["mapping_counts"] == {
        "eligible": 2,
        "changed": 1,
        "eligible_noop": 1,
        "ineligible": 1,
    }
    assert summaries[3]["mapping_counts"] == {
        "eligible": 1,
        "changed": 1,
        "eligible_noop": 0,
        "ineligible": 2,
    }
    for stream, summary in zip(result.streams, summaries, strict=True):
        for i, name in enumerate(replay.DETECTOR_NAMES):
            pairs = list(zip(result.streams[0].rows, stream.rows, strict=True))
            deltas = [b.probabilities[i] - a.probabilities[i] for a, b in pairs]
            scores = summary["paired_with_original"]["detectors"][name]
            assert scores["score_differences"] == {
                "count": 3,
                "changed_score_count": sum(
                    b.probabilities[i] != a.probabilities[i] for a, b in pairs
                ),
                "mean_signed_delta": math.fsum(deltas) / 3,
                "mean_absolute_delta": math.fsum(abs(d) for d in deltas) / 3,
                "max_absolute_delta": max(abs(d) for d in deltas),
            }
            assert list(scores["decision_transitions"]) == ["00", "01", "10", "11"]
            assert scores["decision_transitions"] == {
                f"{a}{b}": sum(
                    x.decisions[i] == a and y.decisions[i] == b for x, y in pairs
                )
                for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
            }


def test_missing_frozen_reference_or_calibration_propagates_null_reason(replay):
    options = kwargs(replay)
    options["mmd_reference"] = drift.MMDReference(
        (), (), (), None, "fewer_than_256_training_domains"
    )
    options["psi_calibration"] = drift.DriftCalibration(
        None, 0, "no_calibration_windows"
    )
    result = replay.replay_probes(records(replay, 256), **options)
    for stream in result.streams:
        assert stream.monitors[1].windows[0].score is None
        assert stream.monitors[1].windows[0].reason == "fewer_than_256_training_domains"
        assert stream.monitors[2].windows[0].score is not None
        assert stream.monitors[2].windows[0].alert is None
    for stream in result.public_summary["streams"]:
        for name, reason in (
            ("mmd", "fewer_than_256_training_domains"),
            ("psi", "no_calibration_windows"),
        ):
            paired = stream["paired_with_original"]["monitors"][name]
            assert paired["reason"] == reason
            assert paired["score_differences"] is None
            assert paired["alert_transitions"] is None


@pytest.mark.parametrize("count", [0, 1, 255])
@pytest.mark.parametrize(
    "unavailable",
    ["mmd_reference", "psi_reference", "mmd_calibration", "psi_calibration"],
)
def test_short_stream_preserves_accepted_unavailable_reason(replay, count, unavailable):
    options = kwargs(replay)
    accepted = {
        "mmd_reference": drift.MMDReference(
            (), (), (), None, "fewer_than_256_training_domains"
        ),
        "psi_reference": drift.PSIReference((), 0, "no_training_rows"),
        "mmd_calibration": drift.DriftCalibration(None, 0, "no_calibration_windows"),
        "psi_calibration": drift.DriftCalibration(None, 0, "no_calibration_windows"),
    }[unavailable]
    options[unavailable] = accepted
    name = unavailable.split("_", maxsplit=1)[0]
    result = replay.replay_probes(records(replay, count), **options)
    for stream in result.streams:
        monitor = next(monitor for monitor in stream.monitors if monitor.name == name)
        assert monitor.reason == accepted.reason
        assert monitor.windows == ()
    for stream in result.public_summary["streams"]:
        monitor = stream["monitors"][name]
        assert monitor["reason"] == accepted.reason
        assert monitor["window_count"] == 0
        assert monitor["score_mean"] is None
        assert monitor["alert_count"] is None
        paired = stream["paired_with_original"]["monitors"][name]
        assert paired["reason"] == accepted.reason
        assert paired["score_differences"] is None
        assert paired["alert_transitions"] is None


def test_no_fit_reference_rebuild_recalibration_or_input_mutation(replay, monkeypatch):
    options = kwargs(replay)
    inputs = records(replay, 256)
    before = copy.deepcopy(
        {key: value for key, value in options.items() if key != "primary_scorer"}
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "replay attempted fitting, reference reconstruction, or calibration"
        )

    for target, name in (
        (gmm, "fit_training_mixture"),
        (gmm, "calibrate_and_audit"),
        (gmm.GaussianMixture, "fit"),
        (gmm.StandardScaler, "fit"),
        (drift, "fit_mmd_reference"),
        (drift, "fit_psi_reference"),
        (drift, "calibrate_window_scores"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    replay.replay_probes(inputs, **options)
    assert {
        key: value for key, value in options.items() if key != "primary_scorer"
    } == before
    assert inputs == records(replay, 256)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: [rows[0], rows[0]],
        lambda rows: [rows[1], rows[0]],
        lambda rows: [replace(rows[0], validation_position=True)],
        lambda rows: [replace(rows[0], validation_position=-1)],
        lambda rows: [replace(rows[0], record_id="bad id")],
        lambda rows: [replace(rows[0], raw_url="not a URL")],
        lambda rows: [asdict(rows[0])],
    ],
)
def test_invalid_input_fails_before_any_scoring(replay, mutation):
    options = kwargs(replay)

    def forbidden(row):
        pytest.fail("invalid input reached scorer")

    options["primary_scorer"] = forbidden
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(mutation(records(replay)), **options)


@pytest.mark.parametrize(
    "changes",
    [
        {"record_id": "different"},
        {"raw_url": "https://different.example"},
        {"length_probability": float("nan")},
        {"stage1_probability": float("inf")},
        {"transformer_probability": -0.1},
        {"length_probability": True},
        {"length_scoring_audit_json": "[]"},
        {"stage1_scoring_audit_json": '{"value":NaN}'},
        {"length_scoring_audit_json": '{"singleton":true,"singleton":false}'},
        {"stage1_scoring_audit_json": '{"nested":{"count":1,"count":2}}'},
    ],
)
def test_score_corruption_alignment_or_nonfinite_fails_stop(replay, changes):
    options = kwargs(replay)
    score = options["primary_scorer"]
    calls = []

    def corrupt(row):
        calls.append(row)
        return replace(score(row), **changes)

    options["primary_scorer"] = corrupt
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(records(replay), **options)
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, True])
def test_invalid_portable_scores_fail(replay, bad):
    options = kwargs(replay)
    options["stage1_model"] = PortableModel(bad)
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(records(replay, 1), **options)


def test_invalid_gmm_scaler_fails_before_primary_scoring(replay):
    options = kwargs(replay)
    options["gmm_artifact"]["scaler"]["scale"][0] = 0.0

    def forbidden(row):
        pytest.fail("invalid scaler reached scorer")

    options["primary_scorer"] = forbidden
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(records(replay, 1), **options)


def test_invalid_feature_nll_and_drift_coverage_fail_stop(replay, monkeypatch):
    options = kwargs(replay)
    monkeypatch.setattr(replay, "extract_url_features", lambda url: (0.0,) * 24)
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(records(replay, 1), **options)
    monkeypatch.setattr(replay, "extract_url_features", extract_url_features)
    monkeypatch.setattr(gmm, "score_feature_matrix", lambda *args: (float("nan"),))
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes(records(replay, 1), **options)


def test_drift_output_alignment_is_not_silently_zipped_or_skipped(replay, monkeypatch):
    monkeypatch.setattr(
        drift,
        "mmd_window_scores",
        lambda *args: drift.DriftWindowScores((255,), (1.0,)),
    )
    with pytest.raises(replay.ProbeReplayError, match="coverage"):
        replay.replay_probes(records(replay, 256), **kwargs(replay))


def test_adapter_uses_authoritative_singletons_not_portable_detector_scores(
    replay, monkeypatch
):
    from automated_phishing_detection import length_inference
    from automated_phishing_detection.selective_inference import (
        InferenceCounts,
        RequestScores,
    )

    length_model = object()
    seen = []

    def length_score(model, urls):
        assert model is length_model
        seen.append(urls)
        return (0.125,), {"singleton": True}

    def score_all(url):
        seen.append(url)
        scorer.counts = InferenceCounts(1, 1, 1, 0)
        return RequestScores(
            0.75, 0.25, 1, 1, False, False, False, True, {"singleton": True}
        )

    monkeypatch.setattr(
        length_inference, "score_length_only_authoritative", length_score
    )
    scorer = SimpleNamespace(score_all=score_all, counts=InferenceCounts(0, 0, 0, 0))
    score = replay.make_primary_scorer(length_model, scorer)
    row = records(replay, 1)[0]
    actual = score(row)
    assert actual == replay.PrimaryScores(
        row.record_id,
        row.raw_url,
        0.125,
        0.75,
        0.25,
        '{"singleton":true}',
        '{"singleton":true}',
    )
    assert seen == [(row.raw_url,), row.raw_url]


@pytest.mark.parametrize("corruption", ["flags", "count", "bool_count", "type"])
def test_adapter_rejects_selective_or_extra_scoring(replay, monkeypatch, corruption):
    from automated_phishing_detection import length_inference
    from automated_phishing_detection.selective_inference import (
        InferenceCounts,
        RequestScores,
    )

    monkeypatch.setattr(
        length_inference, "score_length_only_authoritative", lambda *args: ((0.1,), {})
    )

    def score_all(url):
        attempts = (
            True if corruption == "bool_count" else 2 if corruption == "count" else 1
        )
        scorer.counts = InferenceCounts(attempts, 1, 1, 0)
        result = RequestScores(
            0.75, 0.25, 1, 1, False, False, False, corruption != "flags", {}
        )
        return asdict(result) if corruption == "type" else result

    scorer = SimpleNamespace(score_all=score_all, counts=InferenceCounts(0, 0, 0, 0))
    score = replay.make_primary_scorer(object(), scorer)
    with pytest.raises(replay.ProbeReplayError):
        score(records(replay, 1)[0])


def test_paired_monitor_summary_reports_both_denominators_and_alert_counts(replay):
    result = replay.replay_probes(records(replay, 256), **kwargs(replay))
    original = result.streams[0]
    for stream, summary in zip(
        result.streams, result.public_summary["streams"], strict=True
    ):
        for first, second in zip(original.monitors, stream.monitors, strict=True):
            paired = summary["paired_with_original"]["monitors"][second.name]
            assert (
                paired["original_window_count"]
                == paired["transformed_window_count"]
                == 1
            )
            assert paired["original_alert_count"] == sum(w.alert for w in first.windows)
            assert paired["transformed_alert_count"] == sum(
                w.alert for w in second.windows
            )
            assert paired["original_score_mean"] == first.windows[0].score
            assert paired["transformed_score_mean"] == second.windows[0].score


def test_scoring_error_stops_without_skipping_or_retrying_a_stream(replay):
    options = kwargs(replay)
    original = options["primary_scorer"]
    calls = []

    def stop(row):
        calls.append(row)
        if len(calls) == 5:
            raise RuntimeError("invented failure")
        return original(row)

    options["primary_scorer"] = stop
    with pytest.raises(replay.ProbeReplayError, match="invented failure"):
        replay.replay_probes(records(replay), **options)
    assert len(calls) == 5


def test_distinct_detector_thresholds_are_retained_for_fixed_and_policy(replay):
    options = kwargs(replay)
    options["operating_points"] = replay.OperatingPoints(0.1, 0.7, 0.3, 0.06, -1.0)
    result = replay.replay_probes(records(replay, 257), **options)
    for stream in result.streams:
        for row in stream.rows:
            length, stage1, transformer, fixed, policy = row.probabilities
            band = abs(stage1 - 0.7) <= 0.06
            assert fixed == (transformer if band else stage1)
            assert row.decisions[:4] == (
                int(length >= 0.1),
                int(stage1 >= 0.7),
                int(transformer >= 0.3),
                int(transformer >= 0.3) if band else int(stage1 >= 0.7),
            )
            assert policy == (transformer if band or row.drift_override else stage1)
            assert row.decisions[4] == (
                int(transformer >= 0.3)
                if band or row.drift_override
                else int(stage1 >= 0.7)
            )


def test_changed_score_count_uses_no_tolerance_and_all_four_streams_are_scored(replay):
    options = kwargs(replay)
    calls = []
    original = options["primary_scorer"]

    def score(row):
        calls.append(row)
        value = np.nextafter(0.5, 1.0) if row.raw_url.startswith("HTTPS") else 0.5
        return replace(original(row), length_probability=float(value))

    options["primary_scorer"] = score
    result = replay.replay_probes(records(replay), **options)
    assert len(calls) == 16
    assert [row.record_id for row in calls] == [f"row-{i}" for i in range(4)] * 4
    paired = result.public_summary["streams"][1]["paired_with_original"]["detectors"][
        "length"
    ]
    assert paired["score_differences"]["changed_score_count"] == 4
    assert (
        paired["score_differences"]["mean_signed_delta"] == np.nextafter(0.5, 1.0) - 0.5
    )


def test_reference_nonfinite_bandwidth_is_rejected_even_when_unavailable(replay):
    options = kwargs(replay)
    options["mmd_reference"] = drift.MMDReference(
        (), (), (), float("nan"), "fewer_than_256_training_domains"
    )
    with pytest.raises(replay.ProbeReplayError):
        replay.replay_probes((), **options)


def test_callbacks_precede_next_work_and_preserve_existing_arithmetic(
    replay, monkeypatch
):
    inputs = records(replay, 257)
    options = kwargs(replay)
    baseline = replay.replay_probes(inputs, **options)
    saved_rows, saved_streams, calls = [], [], []
    score = options["primary_scorer"]
    replay_stream = replay._replay_stream
    summarize = replay._summary

    def tracked_score(row):
        assert len(saved_rows) == len(calls)
        assert len(saved_streams) == len(calls) // len(inputs)
        calls.append(row)
        return score(row)

    def tracked_replay_stream(name, rows, *args):
        assert len(saved_rows) == len(calls)
        assert saved_rows[-len(rows) :] == [(name, row) for row in rows]
        return replay_stream(name, rows, *args)

    def tracked_summary(streams):
        assert tuple(saved_streams) == streams
        return summarize(streams)

    monkeypatch.setattr(replay, "_replay_stream", tracked_replay_stream)
    monkeypatch.setattr(replay, "_summary", tracked_summary)
    options["primary_scorer"] = tracked_score
    result = replay.replay_probes(
        inputs,
        **options,
        row_callback=lambda name, row: saved_rows.append((name, row)),
        stream_callback=saved_streams.append,
    )
    assert result == baseline
    assert result.public_summary == baseline.public_summary
    assert len(calls) == len(saved_rows) == 4 * len(inputs)
    assert tuple(saved_streams) == result.streams
    assert all(not row.drift_override for _, row in saved_rows)
    for stream_index, stream in enumerate(result.streams):
        provisional = saved_rows[(stream_index + 1) * len(inputs) - 1][1]
        authoritative = stream.rows[-1]
        assert provisional.probabilities[4] == provisional.probabilities[1]
        assert authoritative.drift_override is True
        assert authoritative.probabilities[4] == authoritative.probabilities[2]
        assert (
            replace(
                provisional,
                probabilities=authoritative.probabilities,
                decisions=authoritative.decisions,
                drift_override=authoritative.drift_override,
                logical_stage2_mask=authoritative.logical_stage2_mask,
            )
            == authoritative
        )


@pytest.mark.parametrize(
    ("failure", "scored_count", "saved_row_count", "saved_stream_count"),
    [
        ("scorer", 6, 5, 1),
        ("monitor", 8, 8, 1),
        ("row_callback", 6, 6, 1),
        ("stream_callback", 8, 8, 2),
        ("aggregate", 16, 16, 4),
    ],
)
def test_callbacks_preserve_prefixes_and_fail_stop_without_extra_scoring(
    replay, monkeypatch, failure, scored_count, saved_row_count, saved_stream_count
):
    options = kwargs(replay)
    saved_rows, saved_streams, calls, monitor_calls = [], [], [], []
    score = options["primary_scorer"]
    mmd_windows = drift.mmd_window_scores

    def tracked_score(row):
        assert len(saved_rows) == len(calls)
        calls.append(row)
        if failure == "scorer" and len(calls) == 6:
            raise RuntimeError("scorer failure")
        return score(row)

    def retain_row(name, row):
        saved_rows.append((name, row))
        if failure == "row_callback" and len(saved_rows) == 6:
            raise RuntimeError("row_callback failure")

    def retain_stream(stream):
        saved_streams.append(stream)
        if failure == "stream_callback" and len(saved_streams) == 2:
            raise RuntimeError("stream_callback failure")

    def tracked_monitor(*args):
        assert len(saved_rows) == len(calls)
        monitor_calls.append(len(calls))
        if failure == "monitor" and len(monitor_calls) == 2:
            raise RuntimeError("monitor failure")
        return mmd_windows(*args)

    def fail_summary(streams):
        assert tuple(saved_streams) == streams
        raise RuntimeError("aggregate failure")

    options["primary_scorer"] = tracked_score
    monkeypatch.setattr(drift, "mmd_window_scores", tracked_monitor)
    if failure == "aggregate":
        monkeypatch.setattr(replay, "_summary", fail_summary)
    with pytest.raises(replay.ProbeReplayError, match=f"{failure} failure"):
        replay.replay_probes(
            records(replay),
            **options,
            row_callback=retain_row,
            stream_callback=retain_stream,
        )
    assert len(calls) == scored_count
    assert len(saved_rows) == saved_row_count
    assert len(saved_streams) == saved_stream_count
    assert [name for name, _ in saved_rows] == [
        name for name in replay.STREAM_NAMES for _ in range(4)
    ][:saved_row_count]
    assert [row.mapping.record_id for _, row in saved_rows] == [
        f"row-{i}" for _ in replay.STREAM_NAMES for i in range(4)
    ][:saved_row_count]
    assert [stream.name for stream in saved_streams] == list(
        replay.STREAM_NAMES[:saved_stream_count]
    )
    assert monitor_calls == list(range(4, (saved_row_count // 4) * 4 + 1, 4))


@pytest.mark.parametrize("name", ["row_callback", "stream_callback"])
def test_noncallable_retention_sink_rejected_before_scoring(replay, name):
    options = kwargs(replay)
    options[name] = False
    options["primary_scorer"] = lambda row: pytest.fail(
        "invalid callback reached scorer"
    )
    with pytest.raises(replay.ProbeReplayError, match="callback"):
        replay.replay_probes(records(replay), **options)


def test_empty_streams_still_reach_stream_callbacks_before_summary(replay, monkeypatch):
    saved = []
    summary = replay._summary

    def tracked_summary(streams):
        assert tuple(saved) == streams
        return summary(streams)

    monkeypatch.setattr(replay, "_summary", tracked_summary)
    result = replay.replay_probes(
        (),
        **kwargs(replay),
        row_callback=lambda *args: pytest.fail("empty stream produced a row"),
        stream_callback=saved.append,
    )
    assert tuple(saved) == result.streams
