"""The historical adapter also preserves the original reference workload."""

from operational_cell_acceptance_fixtures import cell_case, http_run
from study_history_cell_fixtures import (
    candidates,
    case,
    make_history,
    manifests,
    restore,
)

from automated_phishing_detection import operational_cell_acceptance as original
from automated_phishing_detection.http_run_checkpoints import _checkpoint
from automated_phishing_detection.http_run_codec import encode_http_run

__all__ = ["candidates", "case", "manifests"]


def test_full_size_transformer_reference_retains_old_errors_and_invocations(case):
    prepared = cell_case(original, case, ordinal=111)
    run = http_run(prepared.inputs)
    prepared.payloads.update(
        {
            "run.json": encode_http_run(run),
            "warmup.json": _checkpoint(run, measured=False),
            "measured.json": _checkpoint(run, measured=True),
        }
    )
    history = make_history(case, "transformer_only", prepared=prepared)
    result = restore(history)
    assert result.inputs.cell.ordinal == 111
    assert result.summary == history.working.summary
    assert result.summary["workload"] == "transformer_only"
    assert result.summary["request_errors"] == 6
    assert result.run.after_measured.transformer_forward_attempts == 11000
