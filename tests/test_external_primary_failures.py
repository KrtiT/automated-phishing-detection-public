import asyncio
import json
from dataclasses import replace

import pytest
from external_producer_fixtures import prepared_external
from test_external_primary import external_session, primary_module

from automated_phishing_detection import evaluation_producer
from automated_phishing_detection.bound_runtime import BoundSession
from automated_phishing_detection.selective_inference import InferenceCounts


@pytest.mark.parametrize("invalid", [None, object(), (), {"retained": []}])
def test_primary_rejects_invalid_preparation_before_scoring(monkeypatch, invalid):
    module = primary_module()
    session, calls = external_session(monkeypatch)
    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(invalid, session)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "input_validation"
    assert not progress["expected_order_known"]
    assert not progress["completed_rows"]
    assert session.evaluation.primary.scorer.urls == []
    assert calls == [[], [], []]


@pytest.mark.parametrize("invalid", [1, {}, (), "retain"])
def test_invalid_callback_is_rejected_before_inference(monkeypatch, invalid):
    module = primary_module()
    session, calls = external_session(monkeypatch)
    with pytest.raises(module.ExternalPrimaryError):
        module.score_external_primary(prepared_external(), session, retain=invalid)
    assert calls == [[], [], []]


@pytest.mark.parametrize("invalid", [None, object()])
def test_invalid_session_has_unknown_not_zero_counters(invalid):
    module = primary_module()
    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(prepared_external(), invalid)
    progress = json.loads(caught.value.progress)
    assert progress["expected_order_known"]
    assert progress["unattempted_positions"] == [1, 2, 3, 4, 5]
    assert progress["inference_counts"] is None
    assert progress["inference_counts_reason"] == "physical_counts_unavailable"


@pytest.mark.parametrize("fault", ["owner", "nonzero", "threshold"])
def test_primary_preconditions_prevent_all_forward_calls(monkeypatch, fault):
    module = primary_module()
    session, calls = external_session(monkeypatch)
    primary = session.evaluation.primary
    if fault == "owner":
        primary.scorer.owner = -1
    elif fault == "nonzero":
        primary.scorer.extra_attempt = True
    else:
        primary.models.cascade.half_width = -1
    with pytest.raises(module.ExternalPrimaryError):
        module.score_external_primary(prepared_external(), session)
    assert calls == [[], [], []]
    assert primary.scorer.urls == []


def test_completed_phase_cannot_resume_or_reuse_owner(monkeypatch):
    module = primary_module()
    session, calls = external_session(monkeypatch)
    prepared = prepared_external()
    module.score_external_primary(prepared, session)
    with pytest.raises(module.ExternalPrimaryError):
        module.score_external_primary(prepared, session)
    assert len(session.evaluation.primary.scorer.urls) == 5
    assert all(len(values) == 5 for values in calls)


@pytest.mark.parametrize(
    "counts", [None, InferenceCounts(True, 0, 0, 0), InferenceCounts(-1, 0, 0, 0)]
)
def test_invalid_observed_counters_never_become_zero(monkeypatch, counts):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    scorer = session.evaluation.primary.scorer
    monkeypatch.setattr(type(scorer), "counts", property(lambda instance: counts))
    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(prepared_external(), session)
    progress = json.loads(caught.value.progress)
    assert progress["inference_counts"] is None
    assert progress["inference_counts_reason"] == "physical_counts_unavailable"


def test_counter_getter_failure_does_not_mask_phase_error(monkeypatch):
    module = primary_module()
    session, _ = external_session(monkeypatch)

    class BrokenCounts:
        def _require_owner(self):
            pass

        @property
        def counts(self):
            raise RuntimeError("private scorer state")

    evaluation = replace(
        session.evaluation, primary=BoundSession(object(), BrokenCounts())
    )
    with pytest.raises(module.ExternalPrimaryError) as caught:
        module.score_external_primary(
            prepared_external(), replace(session, evaluation=evaluation)
        )
    assert str(caught.value) == "external_primary_input_validation_failed"
    assert json.loads(caught.value.progress)["inference_counts"] is None


def test_cancelled_retention_preserves_exception_and_completed_rows(monkeypatch):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    cancellation = asyncio.CancelledError()
    writes = []

    def retain(name, content):
        writes.append(name)
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        module.score_external_primary(prepared_external(), session, retain=retain)
    assert caught.value is cancellation
    assert writes == ["primary-scores.jsonl"]
    assert len(json.loads(cancellation.progress)["completed_rows"]) == 5


def test_primary_receipt_binds_pre_scoring_input_snapshot(monkeypatch):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    prepared = prepared_external()
    expected = prepared.public_summary["private_sha256"]["retained-test.jsonl"]
    original = evaluation_producer.score_primary_url

    def mutate_input(raw_url, primary, thresholds, position):
        prepared.private_outputs["retained-test.jsonl"] = b"mutated after validation"
        return original(raw_url, primary, thresholds, position)

    monkeypatch.setattr(evaluation_producer, "score_primary_url", mutate_input)
    result = module.score_external_primary(prepared, session)
    assert json.loads(result.receipt_bytes)["retained_test_sha256"] == expected


@pytest.mark.parametrize(
    "exception_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_interruption_error_text_is_symbolic_without_changing_identity(
    monkeypatch, exception_type
):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    interruption = exception_type("private https://secret.example/path")

    def retain(name, content):
        raise interruption

    with pytest.raises(exception_type) as caught:
        module.score_external_primary(prepared_external(), session, retain=retain)
    assert caught.value is interruption
    assert str(caught.value) == "external_primary_retention_failed"
    if exception_type is SystemExit:
        assert interruption.code == "external_primary_retention_failed"


@pytest.mark.parametrize("code", [None, 0, 1, 17])
def test_system_exit_keeps_numeric_exit_semantics(monkeypatch, code):
    module = primary_module()
    session, _ = external_session(monkeypatch)
    interruption = SystemExit(code)

    def retain(name, content):
        raise interruption

    with pytest.raises(SystemExit) as caught:
        module.score_external_primary(prepared_external(), session, retain=retain)
    assert caught.value is interruption
    assert caught.value.code == code
    assert json.loads(caught.value.progress)["status"] == "failed"
