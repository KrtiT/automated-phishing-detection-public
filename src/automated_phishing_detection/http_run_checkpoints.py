"""Join original HTTP phase checkpoints without inventing later drain state."""

from .http_replay import HttpRun, _ReplayProgress
from .http_run_codec import HttpRunCodecError, encode_http_run


def _checkpoint(run, *, measured):
    phase = "measured" if measured else "warmup"
    return _ReplayProgress(
        manifest_sha256=run.manifest_sha256,
        prevalence_basis_points=run.prevalence_basis_points,
        concurrency=run.concurrency,
        run_index=run.run_index,
        workload=run.workload,
        warmup=list(run.warmup),
        measured=list(run.measured) if measured else [None] * 10000,
        warmup_started=[True] * 1000,
        measured_started=[measured] * 10000,
        stage=f"{phase}_checkpoint",
        initial=run.initial,
        after_warmup=run.after_warmup if measured else None,
        measured_elapsed_ms=run.measured_elapsed_ms if measured else None,
    ).snapshot()


def verify_http_checkpoints(
    warmup_bytes: bytes, measured_bytes: bytes, *, run: HttpRun
) -> None:
    """Require complete-run consistency; establish neither process proof nor access."""
    try:
        encode_http_run(run)
        if (
            type(warmup_bytes) is not bytes
            or type(measured_bytes) is not bytes
            or warmup_bytes != _checkpoint(run, measured=False)
            or measured_bytes != _checkpoint(run, measured=True)
        ):
            raise ValueError("checkpoint_mismatch")
    except Exception:
        raise HttpRunCodecError("invalid_http_run_checkpoints") from None
