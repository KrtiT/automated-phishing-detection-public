"""Completed evidence survives later teardown without another count probe."""

import base64
import json

import pytest
from external_composition_fixtures import composition_inputs, producer_module
from test_external_source_checkpoints import science
from test_external_source_checkpoints import writer as writer
from test_external_source_failure import state as state


@pytest.mark.parametrize("count", [0, 1])
def test_completed_producer_survives_session_cleanup(state, writer, monkeypatch, count):
    prepared, session, primary_calls, secondary_calls = composition_inputs(
        monkeypatch, count=count
    )
    with state.capture_body():
        state.produced = producer_module().produce_external_evidence(prepared, session)
    calls_before = (tuple(primary_calls), tuple(secondary_calls))

    def unavailable_counts(unused):
        pytest.fail("failure snapshots must not observe numerical counts")

    monkeypatch.setattr(
        type(session.evaluation.primary.scorer), "counts", property(unavailable_counts)
    )
    record = json.loads(
        state.snapshot(
            writer.attempt, writer.identity, "checkpoint_completion", OSError("secret")
        )
    )
    assert record["cleanup_failed"] is True
    assert record["producer_progress_base64"] is None
    assert record["completed"]["composition"] == json.loads(
        json.dumps(state.produced.public_summary)
    )
    assert {
        name: base64.b64decode(value)
        for name, value in record["completed"]["private_outputs_base64"].items()
    } == state.produced.private_outputs
    assert calls_before == (tuple(primary_calls), tuple(secondary_calls))


def test_known_zero_counts_and_tuple_composition_are_preserved(state, writer):
    state.produced = producer_module().ProducedExternal(
        None, science(), {"counts": {"completed": 0, "members": (0, 0)}}
    )
    with state.capture_body():
        pass
    state.session_closed = True
    record = json.loads(
        state.snapshot(writer.attempt, writer.identity, "final_binding", OSError())
    )
    assert record["completed"]["composition"] == {
        "counts": {"completed": 0, "members": [0, 0]}
    }
    assert record["cleanup_failed"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "not_bytes"])
def test_completed_inventory_is_exactly_thirty(state, writer, mutation):
    outputs = science()
    if mutation == "missing":
        outputs.pop(next(iter(outputs)))
    else:
        outputs["unknown" if mutation == "extra" else next(iter(outputs))] = "private"
    state.produced = producer_module().ProducedExternal(None, outputs, {"counts": 0})
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        state.snapshot(writer.attempt, writer.identity, "final_binding", OSError())
