"""Compose invented validation evidence; no research files are opened."""

import copy
import importlib
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from test_secondary_development import _fixture

from automated_phishing_detection import secondary_development as development


@pytest.fixture
def api():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/automated_phishing_detection/development_probes.py"
    )
    assert path.is_file(), "missing saved-reference probe composition"
    return importlib.import_module("automated_phishing_detection.development_probes")


@pytest.fixture(scope="module")
def evidence():
    from automated_phishing_detection.retained_drift import (
        load_retained_drift_reference,
    )

    fixture = _fixture(development)
    args = fixture["arguments"]
    reference = development.build_training_reference(**args)
    result = development.evaluate_validation(reference, fixture["validation_content"])
    training = result.private_outputs["training-reference.json"]
    audit = result.private_outputs["validation-audit.json"]
    retained = load_retained_drift_reference(
        training,
        audit,
        expected_reference_sha256=sha256(training).hexdigest(),
        expected_audit_sha256=sha256(audit).hexdigest(),
        pins=args["pins"],
        preparation_summary=args["preparation_summary"],
        expected_drift_summary=result.public_summary,
    )
    return fixture, retained


def arguments(evidence):
    fixture, retained = evidence
    return {
        "validation_bytes": fixture["validation_content"],
        "suffix_rules_bytes": fixture["arguments"]["suffix_rules"],
        "preparation_summary": fixture["arguments"]["preparation_summary"],
        "retained": retained,
    }


def test_module_exists(api):
    assert callable(api.prepare_audit_probe_rows)


def test_exact_saved_audit_order_without_labels(api, evidence):
    rows = api.prepare_audit_probe_rows(**arguments(evidence))
    fixture, retained = evidence
    assert tuple(row.record_id for row in rows) == retained.audit_record_ids
    assert (
        tuple(row.validation_position for row in rows)
        == retained.audit_validation_positions
    )
    assert tuple(row.raw_url for row in rows) == tuple(
        fixture["validation"][i]["raw_url"] for i in retained.audit_validation_positions
    )
    assert all(not hasattr(row, "is_phishing") for row in rows)


@pytest.mark.parametrize(
    "name", ["validation_bytes", "suffix_rules_bytes", "preparation_summary"]
)
def test_input_byte_change_rejected_before_scoring(api, evidence, name):
    supplied = arguments(evidence)
    supplied[name] += b" "
    with pytest.raises(api.DevelopmentProbeError):
        api.prepare_audit_probe_rows(**supplied)


@pytest.mark.parametrize(
    "field",
    [
        "audit_record_ids",
        "audit_validation_positions",
        "audit_domains",
        "validation_record_ids",
    ],
)
def test_saved_alignment_change_rejected(api, evidence, field):
    supplied = arguments(evidence)
    original = getattr(supplied["retained"], field)
    supplied["retained"] = replace(
        supplied["retained"], **{field: tuple(reversed(original))}
    )
    with pytest.raises(api.DevelopmentProbeError):
        api.prepare_audit_probe_rows(**supplied)


def test_retained_states_and_all_four_streams_compose_without_refits(
    api, evidence, monkeypatch
):
    from automated_phishing_detection import probe_replay

    supplied = arguments(evidence)
    fixture, retained = evidence
    model = fixture["arguments"]["logistic_l1"]
    artifact = fixture["arguments"]["gmm_state"]
    original = copy.deepcopy(artifact)

    def forbidden(*args, **kwargs):
        pytest.fail("probe composition must not fit or recalibrate")

    monkeypatch.setattr(development, "build_training_reference", forbidden)
    monkeypatch.setattr(development, "evaluate_validation", forbidden)
    seen = []

    def score(row):
        seen.append((row.record_id, row.raw_url))
        return probe_replay.PrimaryScores(
            row.record_id, row.raw_url, 0.2, 0.3, 0.4, "{}", "{}"
        )

    result = api.replay_development_probes(
        **supplied,
        primary_scorer=score,
        stage1_model=model,
        gmm_artifact=artifact,
        operating_points=probe_replay.OperatingPoints(0.5, 0.5, 0.5, 0.1, 0.0),
    )
    assert len(result.streams) == 4
    assert len(seen) == 4 * len(retained.audit_record_ids)
    assert artifact == original
    for stream in result.streams:
        assert (
            tuple(row.mapping.record_id for row in stream.rows)
            == retained.audit_record_ids
        )


@pytest.mark.parametrize(
    "field", ["scaler_mean", "scaler_scale", "portable_state_sha256"]
)
def test_model_reference_mismatch_stops_before_detector_call(api, evidence, field):
    from automated_phishing_detection import probe_replay

    supplied = arguments(evidence)
    fixture, retained = evidence
    original = getattr(retained, field)
    value = "0" * 64 if type(original) is str else tuple(x + 1 for x in original)
    supplied["retained"] = replace(retained, **{field: value})
    with pytest.raises(api.DevelopmentProbeError):
        api.replay_development_probes(
            **supplied,
            primary_scorer=lambda row: pytest.fail("mismatched state was scored"),
            stage1_model=fixture["arguments"]["logistic_l1"],
            gmm_artifact=fixture["arguments"]["gmm_state"],
            operating_points=probe_replay.OperatingPoints(0.5, 0.5, 0.5, 0.1, 0.0),
        )


def test_development_adapter_forwards_optional_callbacks_unchanged(
    api, evidence, monkeypatch
):
    from automated_phishing_detection import probe_replay

    fixture, _ = evidence
    sentinel = object()

    def row_callback(name, row):
        pass

    def stream_callback(stream):
        pass

    def replay_rows(rows, **kwargs):
        assert kwargs["row_callback"] is row_callback
        assert kwargs["stream_callback"] is stream_callback
        return sentinel

    monkeypatch.setattr(probe_replay, "replay_probes", replay_rows)
    result = api.replay_development_probes(
        **arguments(evidence),
        primary_scorer=lambda row: pytest.fail("forwarding test unexpectedly scored"),
        stage1_model=fixture["arguments"]["logistic_l1"],
        gmm_artifact=fixture["arguments"]["gmm_state"],
        operating_points=probe_replay.OperatingPoints(0.5, 0.5, 0.5, 0.1, 0.0),
        row_callback=row_callback,
        stream_callback=stream_callback,
    )
    assert result is sentinel


def test_development_callback_failure_keeps_prefix_without_later_work(api, evidence):
    from automated_phishing_detection import probe_replay

    fixture, retained = evidence
    scored, saved = [], []

    def score(row):
        assert len(scored) == len(saved)
        scored.append(row)
        return probe_replay.PrimaryScores(
            row.record_id, row.raw_url, 0.2, 0.3, 0.4, "{}", "{}"
        )

    def retain(name, row):
        saved.append((name, row))
        if len(saved) == 2:
            raise RuntimeError("invented sink failure")

    with pytest.raises(
        api.DevelopmentProbeError, match="development_probe_replay_failed"
    ):
        api.replay_development_probes(
            **arguments(evidence),
            primary_scorer=score,
            stage1_model=fixture["arguments"]["logistic_l1"],
            gmm_artifact=fixture["arguments"]["gmm_state"],
            operating_points=probe_replay.OperatingPoints(0.5, 0.5, 0.5, 0.1, 0.0),
            row_callback=retain,
            stream_callback=lambda stream: pytest.fail("failed stream was completed"),
        )
    assert len(scored) == len(saved) == 2
    assert (
        tuple(row.mapping.record_id for _, row in saved)
        == retained.audit_record_ids[:2]
    )
    assert [name for name, _ in saved] == ["original", "original"]
