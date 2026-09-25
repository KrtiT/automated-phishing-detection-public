"""Link original shift phase progress to a separate completed run, without I/O."""

from dataclasses import fields, replace

from . import _shift_run_codec as codec
from . import http_replay, shift_replay
from .http_schema import DrainResponse
from .shift_schema import ShiftStateResponse

ShiftRunCodecError = codec.ShiftRunCodecError
_FIELDS = {member.name for member in fields(shift_replay._ShiftProgress)} | {
    "schema_version",
    "workload",
    "concurrency",
}


def _state(value, plan, phase, counts):
    codec.model_shape(value, ShiftStateResponse)
    state = ShiftStateResponse.model_validate(value)
    shift_replay._identity(state, plan)
    codec.require(state.phase == phase and not state.broken and not state.complete)
    codec.require(not state.rows and codec.dump(state.counts) == codec.dump(counts))


def _expected(run, wire, phase):
    measured = phase == "measured"
    return {
        "schema_version": 1,
        "workload": "shift_period",
        "concurrency": 1,
        "manifest_sha256": run.manifest_sha256,
        "run_index": run.run_index,
        "warmup_count": 1000,
        "measured_count": len(run.measured),
        "warmup_started": [True] * 1000,
        "measured_started": [measured] * len(run.measured),
        "warmup": wire["warmup"],
        "measured": wire["measured"] if measured else [None] * len(run.measured),
        "phase": phase,
        "stage": f"{phase}_checkpoint",
        "initial": wire["initial"],
        "after_warmup": wire["after_warmup"] if measured else None,
        "after_measured": None,
        "trace": None,
        "measured_drain_ms": None,
        "measured_elapsed_ms": run.measured_elapsed_ms if measured else None,
        "measured_timeout_drain_ms": run.measured_timeout_drain_ms
        if measured
        else None,
    }


def _checkpoint(content, run, wire, phase):
    value = codec.load(content)
    codec.keys(value, _FIELDS)
    expected = _expected(run, wire, phase)
    codec.require(
        codec.dump({name: value[name] for name in expected}) == codec.dump(expected)
    )
    _state(value["initial_state"], run.plan, "warmup", run.initial)
    if phase == "measured":
        _state(value["reset_state"], run.plan, "measured", run.after_warmup)
    else:
        codec.require(value["reset_state"] is None)
    return value


def _occurrences(values, run, phases):
    codec.require(type(values) is list)
    expected = [
        (phase, position)
        for phase in phases
        for position, row in enumerate(getattr(run, phase))
        if row.error is not None
    ]
    codec.require(len(values) == len(expected))
    for value, identity in zip(values, expected, strict=True):
        codec.keys(value, {"phase", "position", "counts", "elapsed_ms"})
        codec.require(type(value["position"]) is int)
        codec.require((value["phase"], value["position"]) == identity)
        codec.model_shape(value["counts"], DrainResponse)
        DrainResponse.model_validate(value["counts"])
        codec.number(value["elapsed_ms"])


def _span(run, phase, before, previous, counts, start, stop):
    prefix = replace(run, **{phase: getattr(run, phase)[: stop + 1]})
    codec.require(
        shift_replay._complete_counts(counts, before.admitted_requests + stop + 1)
    )
    http_replay._check_phase(prefix, phase, before, counts)
    codec.require(
        all(
            getattr(counts, name) >= value
            for name, value in previous.model_dump().items()
        )
    )
    rows = getattr(run, phase)[start : stop + 1]
    invoked = sum(
        row.response.stage2_invoked for row in rows if row.response is not None
    )
    errors = sum(row.error is not None for row in rows)
    forwards = (
        counts.successful_transformer_scores - previous.successful_transformer_scores
    )
    codec.require(invoked <= forwards <= invoked + errors)
    if phase == "measured":
        expected = sum(row.stage2_invoked for row in run.trace.rows[: stop + 1])
        codec.require(
            counts.successful_transformer_scores
            == before.successful_transformer_scores + expected
        )


def _drains(run, values, phase):
    before = run.initial if phase == "warmup" else run.after_warmup
    after = run.after_warmup if phase == "warmup" else run.after_measured
    points = [
        (value["position"], DrainResponse.model_validate(value["counts"]))
        for value in values
        if value["phase"] == phase
    ]
    points.append((len(getattr(run, phase)) - 1, after))
    previous, start = before, 0
    for position, counts in points:
        _span(run, phase, before, previous, counts, start, position)
        previous, start = counts, position + 1


def _verify(warmup_bytes, measured_bytes, run):
    from .shift_run_codec import encode_shift_run

    wire = codec.load(encode_shift_run(run))
    warmup = _checkpoint(warmup_bytes, run, wire, "warmup")
    measured = _checkpoint(measured_bytes, run, wire, "measured")
    _occurrences(warmup["occurrence_drains"], run, ("warmup",))
    values = measured["occurrence_drains"]
    _occurrences(values, run, ("warmup", "measured"))
    prefix = values[: len(warmup["occurrence_drains"])]
    codec.require(codec.dump(prefix) == codec.dump(warmup["occurrence_drains"]))
    codec.require(
        codec.dump(warmup["initial_state"]) == codec.dump(measured["initial_state"])
    )
    elapsed = 0.0
    for value in values:
        if value["phase"] == "measured":
            elapsed += value["elapsed_ms"]
    codec.require(elapsed == run.measured_timeout_drain_ms)
    for phase in ("warmup", "measured"):
        _drains(run, values, phase)


def verify_shift_checkpoints(
    warmup_bytes: bytes, measured_bytes: bytes, *, run: shift_replay.ShiftRun
) -> None:
    """Verify actual phase-stage bytes; infer neither final progress nor process proof."""
    try:
        _verify(warmup_bytes, measured_bytes, run)
    except Exception:
        raise ShiftRunCodecError("invalid_shift_run_checkpoints") from None
