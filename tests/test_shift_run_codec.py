"""Complete shift record encoding is consistency evidence, never execution proof."""

import importlib
import importlib.util
import json
from dataclasses import replace

import pytest
from shift_run_codec_fixtures import shift_case

from automated_phishing_detection import shift_replay
from automated_phishing_detection.http_replay import ERRORS, ReplayRequest


def module():
    name = "automated_phishing_detection.shift_run_codec"
    assert importlib.util.find_spec(name), "missing strict complete shift codec"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def sample():
    return shift_case(error="timeout")


@pytest.mark.parametrize("error", [None, *sorted(ERRORS)])
def test_roundtrip_preserves_terminal_errors_and_existing_summary(error):
    api = module()
    run, unused_warmup, unused_measured = shift_case(error=error)
    content = api.encode_shift_run(run)
    restored = api.decode_shift_run(content, expected_plan=run.plan)
    assert restored == run
    assert api.encode_shift_run(restored) == content
    assert shift_replay.summarize_shift_run(
        restored
    ) == shift_replay.summarize_shift_run(run)
    assert content.endswith(b"\n") and content.isascii()
    assert restored.measured[0].error == error


@pytest.mark.parametrize("run_index", [1, 2, 3, 4, 5])
def test_full_manifest_is_not_truncated_to_warmup(run_index):
    api = module()
    run, unused_warmup, unused_measured = shift_case(
        measured_count=1001, run_index=run_index
    )
    restored = api.decode_shift_run(api.encode_shift_run(run), expected_plan=run.plan)
    assert restored == run and len(restored.measured) == 1001


@pytest.mark.parametrize("change", ["hash", "repeat", "url", "order", "warmup"])
def test_independent_plan_must_match_every_field(sample, change):
    api = module()
    run = sample[0]
    expected = run.plan
    if change == "hash":
        expected = replace(expected, manifest_sha256="b" * 64)
    elif change == "repeat":
        expected = replace(expected, run_index=2)
    elif change == "warmup":
        expected = replace(expected, warmup_count=999)
    else:
        requests = list(expected.requests)
        if change == "url":
            requests[0] = ReplayRequest(
                requests[0].record_id, "https://different.test/"
            )
        else:
            requests[0], requests[1] = requests[1], requests[0]
        expected = replace(expected, requests=tuple(requests))
    with pytest.raises(api.ShiftRunCodecError):
        api.decode_shift_run(api.encode_shift_run(run), expected_plan=expected)


def test_decoding_returns_fresh_nested_models(sample):
    api = module()
    run = sample[0]
    content = api.encode_shift_run(run)
    restored = api.decode_shift_run(content, expected_plan=run.plan)
    restored.trace.rows[0].monitor_nll = 99.0
    restored.after_measured.completed_requests = 9
    assert api.decode_shift_run(content, expected_plan=run.plan) == run


def test_wire_is_closed_complete_run_not_progress(sample):
    api = module()
    run, warmup, measured = sample
    payload = json.loads(api.encode_shift_run(run))
    assert payload["schema_version"] == 1
    assert payload["workload"] == "shift_period" and payload["concurrency"] == 1
    assert set(payload) == set(run.__dataclass_fields__) | {
        "schema_version",
        "workload",
        "concurrency",
    }
    for partial in (warmup, measured):
        with pytest.raises(api.ShiftRunCodecError):
            api.decode_shift_run(partial, expected_plan=run.plan)


def test_decoded_trace_retains_existing_offline_kernel_compatibility(sample):
    from automated_phishing_detection.policy_replay import (
        MonitorScore,
        PairedProbabilities,
    )

    api = module()
    run = sample[0]
    restored = api.decode_shift_run(api.encode_shift_run(run), expected_plan=run.plan)
    pairs = tuple(
        PairedProbabilities(row.record_id, 0.2, 0.7) for row in run.plan.requests
    )
    monitors = tuple(MonitorScore(row.record_id, 1.0) for row in run.plan.requests)
    shift_replay.verify_offline_trace(
        restored,
        pairs,
        monitors,
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.1,
        monitor_boundary=2.0,
    )
