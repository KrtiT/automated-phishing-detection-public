"""Complete run codecs use retained records only and keep failures symbolic."""

import json
import socket
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from shift_run_codec_fixtures import shift_case
from test_shift_run_checkpoints import wire
from test_shift_run_codec import module
from test_shift_run_codec import sample as sample

from automated_phishing_detection import http_replay, shift_replay


def test_roundtrip_and_checkpoint_validation_perform_no_io(sample, monkeypatch):
    api = module()
    run, warmup, measured = sample

    def forbidden(*args, **kwargs):
        pytest.fail("pure codec attempted a request, source read or replay")

    with monkeypatch.context() as guard:
        for name in ("open", "read_bytes", "read_text"):
            guard.setattr(Path, name, forbidden)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(httpx, "AsyncClient", forbidden)
        for name in ("_scan", "_drain", "replay_run"):
            guard.setattr(http_replay, name, forbidden)
        for name in ("_control", "replay_shift_run"):
            guard.setattr(shift_replay, name, forbidden)
        content = api.encode_shift_run(run)
        restored = api.decode_shift_run(content, expected_plan=run.plan)
        assert api.verify_shift_checkpoints(warmup, measured, run=restored) is None


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**400])
@pytest.mark.parametrize(
    "field", ["measured_elapsed_ms", "measured_drain_ms", "measured_timeout_drain_ms"]
)
def test_nonfinite_and_overflowing_intervals_are_symbolic(sample, field, value):
    api = module()
    with pytest.raises(api.ShiftRunCodecError) as rejected:
        api.encode_shift_run(replace(sample[0], **{field: value}))
    assert str(rejected.value) == "invalid_shift_run_record"


def test_shortened_complete_policy_is_rejected_even_when_generic_plan_is_valid(sample):
    api = module()
    run = sample[0]
    plan = replace(run.plan, warmup_count=999)
    with pytest.raises(api.ShiftRunCodecError):
        api.encode_shift_run(replace(run, plan=plan))


def test_outcome_errors_and_latency_cannot_disappear_from_checkpoints(sample):
    api = module()
    run, warmup, measured = sample
    value = json.loads(measured)
    value["measured"][0]["elapsed_ms"] = 1.0
    value["measured"][0]["error"] = None
    with pytest.raises(api.ShiftRunCodecError) as rejected:
        api.verify_shift_checkpoints(warmup, wire(value), run=run)
    assert str(rejected.value) == "invalid_shift_run_checkpoints"


def test_forward_claim_at_earlier_timeout_cannot_regress_in_later_drain(sample):
    api = module()
    run, warmup, measured = sample
    first, second = json.loads(warmup), json.loads(measured)
    for value in (first, second):
        counts = value["occurrence_drains"][0]["counts"]
        counts["transformer_forward_attempts"] = 1
        counts["successful_transformer_scores"] = 1
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(wire(first), wire(second), run=run)


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_finite_intervals_cannot_overflow_complete_summary_rates(operation):
    api = module()
    run = shift_case()[0]
    measured = tuple(replace(row, elapsed_ms=0.0) for row in run.measured)
    changed = replace(run, measured=measured, measured_elapsed_ms=1e-308)
    with pytest.raises(api.ShiftRunCodecError):
        if operation == "encode":
            api.encode_shift_run(changed)
        else:
            content = api.codec.dump(api.codec.run_wire(changed))
            api.decode_shift_run(content, expected_plan=run.plan)
