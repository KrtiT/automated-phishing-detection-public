"""The durable callback matches the existing producer, without rescoring retries."""

import json

import pytest
from external_composition_fixtures import composition_inputs, producer_module
from test_external_source_checkpoints import checkpoint_module, provenance
from test_external_source_checkpoints import writer as writer


@pytest.mark.parametrize("count", [0, 1, 255, 256, 319, 320])
def test_existing_producer_retains_exact_thirty_scientific_payloads(
    writer, monkeypatch, count
):
    module = producer_module()
    prepared, session, primary_calls, secondary_calls = composition_inputs(
        monkeypatch, count
    )
    writer.begin(provenance())
    result = module.produce_external_evidence(prepared, session, retain=writer)
    outputs = writer.complete(result.private_outputs)
    assert outputs == provenance() | result.private_outputs
    assert all(len(calls) == count for calls in primary_calls)
    assert len(secondary_calls) == (11 if count else 0)
    assert json.loads(writer.snapshot())["status"] == "complete"
    assert result.public_summary["protected_evaluation_authorized"] is False


def test_failed_first_scientific_install_stops_before_any_forward(writer, monkeypatch):
    module, checkpoint = producer_module(), checkpoint_module()
    prepared, session, primary_calls, secondary_calls = composition_inputs(monkeypatch)
    writer.begin(provenance())
    installed, original = [], checkpoint.checkpoint_io._install_record

    def fail(directory, name, content):
        installed.append(name)
        original(directory, name, content)
        raise OSError("private-canary")

    monkeypatch.setattr(checkpoint.checkpoint_io, "_install_record", fail)
    with pytest.raises(module.ExternalProducerError) as caught:
        module.produce_external_evidence(prepared, session, retain=writer)
    assert all(not calls for calls in primary_calls)
    assert secondary_calls == []
    assert installed == [checkpoint.SCIENTIFIC_ORDER[0]]
    progress = json.loads(caught.value.progress)
    assert progress["failed_checkpoint"] == checkpoint.SCIENTIFIC_ORDER[0]
    assert progress["retention_status"] == "failed_or_ambiguous"
    assert json.loads(writer.snapshot())["status"] == "failed"
