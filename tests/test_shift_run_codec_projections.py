"""Mutable model copies must not create an encoder-only or untyped wire form."""

import warnings
from dataclasses import replace

import pytest
from shift_run_codec_fixtures import shift_case
from test_shift_run_codec import module


@pytest.fixture(scope="module")
def run():
    return shift_case(error="timeout")[0]


def _selected_first(run):
    counts = run.after_measured.model_copy(
        update={"transformer_forward_attempts": 1, "successful_transformer_scores": 1}
    )
    first = run.trace.rows[0].model_copy(
        update={
            "band_selected": True,
            "stage2_invoked": True,
            "transformer_probability": 0.0,
        }
    )
    trace = run.trace.model_copy(
        update={"counts": counts, "rows": [first, *run.trace.rows[1:]]}
    )
    return replace(run, after_measured=counts, trace=trace)


def _integer_float(run, field):
    if field == "response":
        response = run.warmup[0].response.model_copy(update={"probability": 0})
        return replace(
            run, warmup=(replace(run.warmup[0], response=response), *run.warmup[1:])
        )
    if field == "transformer_probability":
        run = _selected_first(run)
    rows = list(run.trace.rows)
    if field == "window":
        window = rows[255].window.model_copy(update={"score": 1})
        rows[255] = rows[255].model_copy(update={"window": window})
    else:
        rows[0] = rows[0].model_copy(update={field: 1 if field == "monitor_nll" else 0})
    return replace(run, trace=run.trace.model_copy(update={"rows": rows}))


@pytest.mark.parametrize(
    "field",
    [
        "response",
        "stage1_probability",
        "transformer_probability",
        "monitor_nll",
        "window",
    ],
)
@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_integer_mutations_of_model_float_fields_are_rejected(run, field, operation):
    api = module()
    changed = _integer_float(run, field)
    with pytest.raises(api.ShiftRunCodecError):
        if operation == "encode":
            api.encode_shift_run(changed)
        else:
            content = api.codec.dump(api.codec.run_wire(changed))
            api.decode_shift_run(content, expected_plan=run.plan)


def _untyped_trace(run, field):
    if field == "counts":
        return run.trace.model_copy(update={"counts": run.trace.counts.model_dump()})
    if field == "rows":
        return run.trace.model_copy(update={"rows": tuple(run.trace.rows)})
    rows = list(run.trace.rows)
    if field == "row":
        rows[0] = rows[0].model_dump()
    else:
        rows[255] = rows[255].model_copy(
            update={"window": rows[255].window.model_dump()}
        )
    return run.trace.model_copy(update={"rows": rows})


@pytest.mark.parametrize("field", ["counts", "rows", "row", "window"])
def test_encoder_requires_exact_typed_nested_trace_models(run, field):
    api = module()
    changed = replace(run, trace=_untyped_trace(run, field))
    with warnings.catch_warnings(), pytest.raises(api.ShiftRunCodecError):
        warnings.simplefilter("ignore", UserWarning)
        api.encode_shift_run(changed)


def test_typed_selected_trace_roundtrips_without_coercion(run):
    api = module()
    changed = _selected_first(run)
    content = api.encode_shift_run(changed)
    restored = api.decode_shift_run(content, expected_plan=run.plan)
    assert restored == changed
    assert api.encode_shift_run(restored) == content
