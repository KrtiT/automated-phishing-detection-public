"""Individual drains constrain forwards which aggregate counters cannot identify."""

import json
from dataclasses import replace

import pytest
from shift_run_codec_fixtures import checkpoints, shift_case
from test_shift_run_checkpoints import wire
from test_shift_run_codec import module


def forwards(counts, total):
    return counts.model_copy(
        update={
            "transformer_forward_attempts": total,
            "successful_transformer_scores": total,
        }
    )


def _physical_run(*, later_warmup_success=False):
    run = shift_case(error="timeout")[0]
    warmup = list(run.warmup)
    if later_warmup_success:
        response = warmup[2].response.model_copy(update={"stage2_invoked": True})
        warmup[2] = replace(warmup[2], response=response)
    after_warmup = forwards(run.after_warmup, 1)
    after_measured = forwards(run.after_measured, 1)
    trace = run.trace.model_copy(update={"counts": after_measured})
    return replace(
        run,
        warmup=tuple(warmup),
        after_warmup=after_warmup,
        after_measured=after_measured,
        trace=trace,
    )


def _physical_checkpoints(run, warmup_forward):
    warmup, measured = map(json.loads, checkpoints(run))
    for value in (warmup, measured):
        for drain in value["occurrence_drains"]:
            total = warmup_forward if drain["phase"] == "warmup" else 1
            drain["counts"].update(
                transformer_forward_attempts=total, successful_transformer_scores=total
            )
    measured["reset_state"]["counts"] = run.after_warmup.model_dump()
    return wire(warmup), wire(measured)


@pytest.mark.parametrize("later_success", [False, True])
def test_completed_timeout_forward_and_later_success_have_valid_distinct_drains(
    later_success,
):
    api = module()
    run = _physical_run(later_warmup_success=later_success)
    warmup, measured = _physical_checkpoints(run, 0 if later_success else 1)
    assert (
        api.decode_shift_run(api.encode_shift_run(run), expected_plan=run.plan) == run
    )
    assert api.verify_shift_checkpoints(warmup, measured, run=run) is None


def test_consistently_rehashed_aggregate_cannot_hide_missing_later_forward():
    api = module()
    run = _physical_run(later_warmup_success=True)
    warmup, measured = _physical_checkpoints(run, 1)
    api.encode_shift_run(run)
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(warmup, measured, run=run)


def test_measured_timeout_drain_is_bound_to_trace_not_only_error_budget():
    api = module()
    run, warmup, measured = shift_case(error="timeout")
    value = json.loads(measured)
    counts = value["occurrence_drains"][1]["counts"]
    counts.update(transformer_forward_attempts=1, successful_transformer_scores=1)
    with pytest.raises(api.ShiftRunCodecError):
        api.verify_shift_checkpoints(warmup, wire(value), run=run)


def test_successfully_drained_measured_timeout_can_include_a_trace_forward():
    api = module()
    run = _physical_run()
    after = forwards(run.after_measured, 2)
    first = run.trace.rows[0].model_copy(
        update={
            "stage1_probability": 0.5,
            "transformer_probability": 0.8,
            "fixed_decision": 1,
            "decision": 1,
            "band_selected": True,
            "stage2_invoked": True,
        }
    )
    trace = run.trace.model_copy(
        update={"counts": after, "rows": [first, *run.trace.rows[1:]]}
    )
    run = replace(run, after_measured=after, trace=trace)
    warmup, measured = _physical_checkpoints(run, 1)
    saved = json.loads(measured)
    for drain in saved["occurrence_drains"]:
        if drain["phase"] == "measured":
            drain["counts"].update(
                transformer_forward_attempts=2, successful_transformer_scores=2
            )
    restored = api.decode_shift_run(api.encode_shift_run(run), expected_plan=run.plan)
    assert api.verify_shift_checkpoints(warmup, wire(saved), run=restored) is None
