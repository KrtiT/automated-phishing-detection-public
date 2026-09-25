import importlib
import importlib.util
import json
from dataclasses import asdict
from hashlib import sha256

import pytest
from external_producer_fixtures import prepared_external
from test_evaluation_producer import synthetic_session

from automated_phishing_detection import evaluation_producer
from automated_phishing_detection.bound_external_runtime import BoundExternalSession
from automated_phishing_detection.selective_inference import InferenceCounts


def primary_module():
    name = "automated_phishing_detection.external_primary"
    assert importlib.util.find_spec(name) is not None, "external primary phase missing"
    return importlib.import_module(name)


def external_session(monkeypatch):
    evaluation, *calls = synthetic_session(evaluation_producer, monkeypatch)
    return BoundExternalSession(evaluation, object()), calls


@pytest.mark.parametrize("count", [0, 1, 5])
def test_primary_retains_every_row_once_in_original_order(monkeypatch, count):
    module = primary_module()
    prepared = prepared_external(count)
    session, calls = external_session(monkeypatch)
    result = module.score_external_primary(prepared, session)
    assert result.records == prepared.retained
    assert len(result.scores) == count
    assert result.inference_counts == InferenceCounts(count, count, count, 0)
    assert session.evaluation.primary.scorer.urls == [
        row.raw_url for row in prepared.retained
    ]
    assert all(len(values) == count for values in calls)
    assert dict(result.thresholds)["logistic_l1"] == 0.5
    assert [score.monitor_probability for score in result.scores] == [0.33] * count
    assert b"host" not in repr(result).encode()


def test_primary_matches_shared_singleton_without_fabricating_tranco_label(monkeypatch):
    module = primary_module()
    prepared = prepared_external()
    session, _ = external_session(monkeypatch)
    result = module.score_external_primary(prepared, session)
    comparison, _ = external_session(monkeypatch)
    expected = tuple(
        evaluation_producer.score_primary_url(
            row.raw_url,
            comparison.evaluation.primary,
            dict(result.thresholds),
            position,
        )
        for position, row in enumerate(prepared.retained, 1)
    )
    assert result.scores == expected
    assert result.records[-1].role == "tranco"
    assert result.records[-1].is_phishing is None


def test_primary_installs_exact_canonical_checkpoint_before_receipt(monkeypatch):
    module = primary_module()
    prepared = prepared_external()
    session, _ = external_session(monkeypatch)
    retained = []
    result = module.score_external_primary(
        prepared, session, retain=lambda name, content: retained.append((name, content))
    )
    assert retained == [
        ("primary-scores.jsonl", result.checkpoint_bytes),
        ("primary-completion.json", result.receipt_bytes),
    ]
    expected = b"".join(
        evaluation_producer._json_bytes(
            {"record": asdict(row), "primary": asdict(score)}
        )
        for row, score in zip(result.records, result.scores)
    )
    assert result.checkpoint_bytes == expected
    assert json.loads(result.receipt_bytes) == {
        "schema_version": 1,
        "phase": "external_primary",
        "row_count": 5,
        "retained_test_sha256": sha256(
            prepared.private_outputs["retained-test.jsonl"]
        ).hexdigest(),
        "primary_scores_sha256": sha256(expected).hexdigest(),
        "thresholds": dict(result.thresholds),
        "inference_counts": asdict(result.inference_counts),
    }


@pytest.mark.parametrize("fail_at", [0, 1, 4])
def test_failed_primary_preserves_completed_prefix_and_actual_counters(
    monkeypatch, fail_at
):
    module = primary_module()
    prepared = prepared_external()
    session, _ = external_session(monkeypatch)
    session.evaluation.primary.scorer.fail_at = fail_at
    retained = []
    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(
            prepared, session, retain=lambda *args: retained.append(args)
        )
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "scoring"
    assert progress["expected_record_ids"] == [
        row.record_id for row in prepared.retained
    ]
    assert len(progress["completed_rows"]) == fail_at
    assert progress["started_position"] == fail_at + 1
    assert progress["unattempted_positions"] == list(range(fail_at + 2, 6))
    assert progress["inference_counts"] == asdict(
        InferenceCounts(fail_at + 1, fail_at + 1, fail_at + 1, 0)
    )
    assert len(session.evaluation.primary.scorer.urls) == fail_at + 1
    assert retained == []
    assert "host" not in str(caught.value)


@pytest.mark.parametrize(
    "fail_name", ["primary-scores.jsonl", "primary-completion.json"]
)
def test_failed_retention_never_retries_and_keeps_complete_progress(
    monkeypatch, fail_name
):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    writes = []

    def retain(name, content):
        writes.append((name, content))
        if name == fail_name:
            raise OSError("private writer path")

    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(prepared_external(), session, retain=retain)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "retention"
    assert len(progress["completed_rows"]) == 5
    assert progress["unattempted_positions"] == []
    assert [name for name, _ in writes][-1] == fail_name
    assert len(writes) == (1 if fail_name == "primary-scores.jsonl" else 2)
    assert str(caught.value) == "external_primary_retention_failed"


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_primary_preserves_interruption_type_and_prefix(monkeypatch, exception_type):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    original = evaluation_producer.score_primary_url

    def score(raw_url, primary, thresholds, position):
        if position == 3:
            raise exception_type()
        return original(raw_url, primary, thresholds, position)

    monkeypatch.setattr(evaluation_producer, "score_primary_url", score)
    with pytest.raises(exception_type) as caught:
        module.score_external_primary(prepared_external(), session)
    progress = json.loads(caught.value.progress)
    assert len(progress["completed_rows"]) == 2
    assert progress["started_position"] == 3
    assert progress["unattempted_positions"] == [4, 5]
    assert progress["inference_counts"]["transformer_forward_attempts"] == 2
