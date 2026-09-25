import asyncio
import json
from dataclasses import replace

import pytest
from external_composition_fixtures import composition_inputs, producer_module
from test_external_producer_failures import retained


@pytest.mark.parametrize("exception_type", [ValueError, asyncio.CancelledError])
def test_non_ascii_callback_progress_cannot_erase_parent_checkpoints(
    monkeypatch, exception_type
):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    error = exception_type("private callback message")
    error.progress = b"\xff"

    def retain(name, content):
        raise error

    expected_type = (
        module.ExternalProducerError if exception_type is ValueError else exception_type
    )
    with pytest.raises(expected_type) as caught:
        module.produce_external_evidence(prepared, session, retain=retain)
    progress = json.loads(caught.value.progress)
    assert retained(progress) == {
        "retained-test.jsonl": prepared.private_outputs["retained-test.jsonl"]
    }
    assert progress["phase_progress_json"] is None
    if exception_type is asyncio.CancelledError:
        assert caught.value is error


def test_callback_progress_property_is_not_evaluated(monkeypatch):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)

    class PrivateProgressError(ValueError):
        @property
        def progress(self):
            raise RuntimeError("private-progress-getter-message")

    def retain(name, content):
        raise PrivateProgressError("private")

    with pytest.raises(module.ExternalProducerError) as caught:
        module.produce_external_evidence(prepared, session, retain=retain)
    assert json.loads(caught.value.progress)["phase_progress_json"] is None
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("fault", ["tabular_hash", "seed_type", "stage1_threshold"])
def test_all_secondary_binding_checks_precede_primary_inference(monkeypatch, fault):
    module = producer_module()
    prepared, session, primary_calls, _ = composition_inputs(monkeypatch)
    bound = session.evaluation.secondary
    if fault == "tabular_hash":
        member = replace(bound.tabular[0], artifact_sha256="f" * 64)
        bound = replace(bound, tabular=(member, *bound.tabular[1:]))
    elif fault == "seed_type":
        bound = replace(
            bound, seeds=(replace(bound.seeds[0], seed=42.0), *bound.seeds[1:])
        )
    else:
        bound = replace(bound, stage1_threshold=0.4)
    session = replace(session, evaluation=replace(session.evaluation, secondary=bound))
    with pytest.raises(module.ExternalProducerError):
        module.produce_external_evidence(prepared, session)
    assert session.evaluation.primary.scorer.urls == []
    assert primary_calls == [[], [], []]
