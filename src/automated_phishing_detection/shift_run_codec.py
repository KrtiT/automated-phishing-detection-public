"""Complete saved shift runs establish consistency, not execution or access proof."""

from . import _shift_run_codec as codec
from .http_schema import DrainResponse
from .shift_replay import ShiftRun, summarize_shift_run
from .shift_run_checkpoints import verify_shift_checkpoints
from .shift_schema import ShiftPlan, ShiftStateResponse

ShiftRunCodecError = codec.ShiftRunCodecError
__all__ = [
    "ShiftRunCodecError",
    "encode_shift_run",
    "decode_shift_run",
    "verify_shift_checkpoints",
]


def encode_shift_run(run: ShiftRun) -> bytes:
    """Encode complete frozen-workload evidence without executing requests."""
    try:
        content = codec.dump(codec.run_wire(run))
        codec.run_shape(codec.load(content), run.plan)
        codec.dump(summarize_shift_run(run))
        return content
    except Exception:
        raise ShiftRunCodecError("invalid_shift_run_record") from None


def decode_shift_run(content: bytes, *, expected_plan: ShiftPlan) -> ShiftRun:
    """Bind the exact independently supplied plan before constructing typed rows."""
    try:
        codec.validate_plan(expected_plan)
        value = codec.load(content)
        codec.run_shape(value, expected_plan)
        run = ShiftRun(
            expected_plan,
            codec.outcomes(value["warmup"]),
            codec.outcomes(value["measured"]),
            *(
                DrainResponse.model_validate(value[name])
                for name in ("initial", "after_warmup", "after_measured")
            ),
            ShiftStateResponse.model_validate(value["trace"]),
            *(
                value[name]
                for name in (
                    "measured_elapsed_ms",
                    "measured_drain_ms",
                    "measured_timeout_drain_ms",
                )
            ),
        )
        codec.dump(summarize_shift_run(run))
        codec.require(encode_shift_run(run) == content)
        return run
    except Exception:
        raise ShiftRunCodecError("invalid_shift_run_record") from None
