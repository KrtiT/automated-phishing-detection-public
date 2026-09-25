"""Closed shape, identity and numerical rejection before accepting saved shift runs."""

import json
from dataclasses import replace

import pytest
from test_shift_run_checkpoints import wire
from test_shift_run_codec import module
from test_shift_run_codec import sample as sample

from automated_phishing_detection.http_schema import DrainResponse

CONTAINERS = (
    (),
    ("plan",),
    ("plan", "requests", 0),
    ("warmup", 0),
    ("measured", 1),
    ("warmup", 0, "response"),
    ("initial",),
    ("trace",),
    ("trace", "rows", 0),
    ("trace", "rows", 255, "window"),
)


def target(value, path):
    for name in path:
        value = value[name]
    return value


@pytest.mark.parametrize("path", CONTAINERS)
def test_every_missing_or_extra_nested_field_is_rejected(sample, path):
    api = module()
    run = sample[0]
    original = api.encode_shift_run(run)
    names = list(target(json.loads(original), path)) + ["undeclared"]
    for name in names:
        value = json.loads(original)
        container = target(value, path)
        if name == "undeclared":
            container[name] = None
        else:
            del container[name]
        with pytest.raises(api.ShiftRunCodecError):
            api.decode_shift_run(wire(value), expected_plan=run.plan)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("concurrency",), True),
        (("concurrency",), 2),
        (("workload",), "fixed_cascade"),
        (("plan", "run_index"), True),
        (("plan", "warmup_count"), 999),
        (("plan", "manifest_sha256"), "invalid"),
        (("plan", "requests", 0, "raw_url"), "https://different.test/"),
        (("measured_elapsed_ms",), None),
        (("measured_drain_ms",), None),
        (("measured_drain_ms",), True),
        (("measured_drain_ms",), 0),
        (("measured_timeout_drain_ms",), -1),
        (("measured_timeout_drain_ms",), True),
        (("measured", 1, "elapsed_ms"), True),
        (("measured", 1, "elapsed_ms"), -1),
        (("measured", 1, "elapsed_ms"), "1"),
        (("measured", 1, "status_code"), True),
        (("measured", 0, "error"), "unknown"),
        (("measured", 1, "record_id"), "different"),
        (("measured", 1, "request_id"), "different"),
        (("measured", 1, "response", "probability"), True),
        (("initial", "completed_requests"), False),
        (("trace", "complete"), 1),
        (("trace", "complete"), False),
        (("trace", "rows", 1, "position"), 1),
        (("trace", "rows", 255, "window", "score"), 2.0),
        (("trace", "rows", 0, "stage2_invoked"), True),
    ],
)
def test_known_field_forgeries_are_rejected(sample, path, value):
    api = module()
    run = sample[0]
    payload = json.loads(api.encode_shift_run(run))
    target(payload, path[:-1])[path[-1]] = value
    with pytest.raises(api.ShiftRunCodecError):
        api.decode_shift_run(wire(payload), expected_plan=run.plan)


@pytest.mark.parametrize(
    "kind", ["duplicate", "spacing", "newline", "utf8", "nan", "overflow"]
)
def test_noncanonical_and_nonfinite_json_is_rejected(sample, kind):
    api = module()
    run = sample[0]
    content = api.encode_shift_run(run)
    mutations = {
        "duplicate": b'{"concurrency":1,' + content[1:],
        "spacing": b" " + content,
        "newline": content[:-1],
        "utf8": content.replace(b"invented.test", "inventé.test".encode(), 1),
        "nan": content.replace(b'"measured_drain_ms":0.5', b'"measured_drain_ms":NaN'),
        "overflow": content.replace(
            b'"measured_drain_ms":0.5', b'"measured_drain_ms":1e999'
        ),
    }
    with pytest.raises(api.ShiftRunCodecError):
        api.decode_shift_run(mutations[kind], expected_plan=run.plan)


@pytest.mark.parametrize("field", ["warmup", "measured"])
def test_shortened_or_reordered_phase_is_not_complete(sample, field):
    api = module()
    run = sample[0]
    for rows in (getattr(run, field)[:-1], tuple(reversed(getattr(run, field)))):
        with pytest.raises(api.ShiftRunCodecError):
            api.encode_shift_run(replace(run, **{field: rows}))


def test_missing_defaulted_trace_field_is_rejected_before_model_construction(
    sample, monkeypatch
):
    api = module()
    run = sample[0]
    value = json.loads(api.encode_shift_run(run))
    del value["trace"]["workload"]
    monkeypatch.setattr(
        DrainResponse,
        "model_validate",
        lambda *args, **kwargs: pytest.fail("constructed before closed shape"),
    )
    with pytest.raises(api.ShiftRunCodecError):
        api.decode_shift_run(wire(value), expected_plan=run.plan)


@pytest.mark.parametrize("value", [None, {}, "{}\n", bytearray(b"{}\n")])
def test_content_must_be_exact_bytes(sample, value):
    api = module()
    with pytest.raises(api.ShiftRunCodecError):
        api.decode_shift_run(value, expected_plan=sample[0].plan)
