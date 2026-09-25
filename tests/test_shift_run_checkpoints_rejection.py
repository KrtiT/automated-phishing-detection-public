"""Progress type, stage and cross-checkpoint mutations cannot claim completeness."""

import json

import pytest
from test_shift_run_checkpoints import module, wire
from test_shift_run_checkpoints import sample as sample
from test_shift_run_codec_rejection import target


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize(
    "path",
    [(), ("initial_state",), ("initial_state", "counts"), ("occurrence_drains", 0)],
)
def test_all_nested_inventory_fields_are_required(sample, phase, path):
    api = module()
    run, warmup, measured = sample
    original = warmup if phase == "warmup" else measured
    for name in [*target(json.loads(original), path), "extra"]:
        value = json.loads(original)
        nested = target(value, path)
        if name == "extra":
            nested[name] = None
        else:
            del nested[name]
        pair = (wire(value), measured) if phase == "warmup" else (warmup, wire(value))
        with pytest.raises(api.ShiftRunCodecError):
            api.verify_shift_checkpoints(*pair, run=run)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("run_index",), True),
        (("warmup_count",), 1),
        (("concurrency",), True),
        (("manifest_sha256",), "b" * 64),
        (("phase",), "warmup"),
        (("stage",), "validation"),
        (("measured_started", 0), 1),
        (("warmup_started", 0), False),
        (("measured", 0), None),
        (("warmup", 0), None),
        (("initial_state", "complete"), True),
        (("initial_state", "broken"), True),
        (("initial_state", "phase"), "measured"),
        (("reset_state", "phase"), "warmup"),
        (("reset_state", "complete"), True),
        (("reset_state", "counts", "completed_requests"), 999),
        (("reset_state", "counts", "admitted_requests"), True),
        (("measured_timeout_drain_ms",), 99.0),
        (("measured_elapsed_ms",), 999999.0),
        (("occurrence_drains", 1, "elapsed_ms"), -1),
        (("occurrence_drains", 1, "counts", "admitted_requests"), True),
    ],
)
def test_checkpoint_values_must_link_to_actual_run(sample, path, value):
    api = module()
    run, warmup, measured = sample
    payload = json.loads(measured)
    target(payload, path[:-1])[path[-1]] = value
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(warmup, wire(payload), run=run)


@pytest.mark.parametrize(
    "name",
    ["after_warmup", "reset_state", "measured_elapsed_ms", "measured_timeout_drain_ms"],
)
def test_warmup_checkpoint_cannot_claim_unperformed_stages(sample, name):
    api = module()
    run, warmup, measured = sample
    payload = json.loads(warmup)
    payload[name] = json.loads(measured)[name]
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(wire(payload), measured, run=run)


@pytest.mark.parametrize(
    "content", [b"{}", b"{}\n", b'{"stage":1,"stage":2}\n', None, "{}\n"]
)
@pytest.mark.parametrize("phase", ["warmup", "measured"])
def test_invalid_checkpoint_bytes_are_rejected(sample, content, phase):
    api = module()
    run, warmup, measured = sample
    pair = (content, measured) if phase == "warmup" else (warmup, content)
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(*pair, run=run)
