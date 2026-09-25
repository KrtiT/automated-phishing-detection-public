"""Reconstruct the complete external composition from retained bytes only.

Consistency and portable arithmetic do not authenticate source provenance,
physical model observations, numerical-owner cleanup, or a successful process.
Those remain requirements of the separately reviewed official execution wrapper.
"""

from dataclasses import asdict

from . import external_metrics, external_replay, saved_evidence
from ._external_public_summary import public_summary as compose_summary
from ._saved_external_arithmetic import verify_retained_arithmetic
from ._saved_external_bindings import (
    restore_bindings,
    restore_reference,
    snapshot_outputs,
)
from ._saved_external_columns import restore_secondary
from ._saved_external_inputs import restore_preparation, restore_primary
from .bound_secondary import SecondaryScoring
from .external_primary import ExternalPrimaryScores
from .external_replay import ReplayedExternal


class SavedExternalEvidenceError(ValueError):
    """Symbolic rejection of incomplete, malformed or inconsistent saved bytes."""


def _score_rows(
    primary: ExternalPrimaryScores, secondary: SecondaryScoring, bindings: dict
) -> tuple[dict, ...]:
    rows = []
    for record, score, members in zip(
        primary.records, primary.scores, secondary.rows, strict=True
    ):
        row = {
            "record": asdict(record),
            **asdict(score),
            "secondary_tabular": [asdict(value) for value in members.tabular],
            "secondary_seeds": [asdict(value) for value in members.seeds],
        }
        row["features"] = list(score.features)
        saved_evidence._validate_score_row(row, bindings)
        rows.append(row)
    return tuple(rows)


def _compare_derived(outputs: dict[str, bytes], replay: ReplayedExternal) -> None:
    expected = {
        "routing.json": saved_evidence._json_bytes(asdict(replay.evidence.replay)),
        "monitors.json": saved_evidence._json_bytes(
            [asdict(monitor) for monitor in replay.monitors]
        ),
        "predictions.jsonl": b"".join(
            saved_evidence._json_bytes(asdict(row)) for row in replay.rows
        ),
        "secondary.json": saved_evidence._json_bytes(
            external_metrics.summarize_external(replay.rows)
        ),
    }
    saved_evidence._require(
        all(outputs[name] == content for name, content in expected.items()),
        "saved_external_derived_outputs_differ",
    )


def _reconstruct(outputs: dict[str, bytes], public: bytes) -> ReplayedExternal:
    snapshot, unused_summary = snapshot_outputs(outputs, public)
    bindings = restore_bindings(snapshot)
    reference = restore_reference(snapshot, bindings)
    prepared = restore_preparation(snapshot)
    primary = restore_primary(snapshot, prepared, bindings)
    secondary = restore_secondary(snapshot, primary, bindings)
    rows = _score_rows(primary, secondary, bindings)
    verify_retained_arithmetic(rows, bindings, reference)
    replay = external_replay.replay_external_scores(primary, secondary, reference)
    _compare_derived(snapshot, replay)
    summary = compose_summary(
        prepared, replay, primary.inference_counts, secondary.counts, snapshot
    )
    saved_evidence._require(
        saved_evidence._json_bytes(summary) == public,
        "saved_external_summary_differs",
    )
    return replay


def reconstruct_external_evidence(
    private_outputs: dict[str, bytes], public_summary: bytes
) -> ReplayedExternal:
    """Rebuild one complete stream, with no source reads, fitting or forwards."""
    try:
        return _reconstruct(private_outputs, public_summary)
    except Exception:
        raise SavedExternalEvidenceError("invalid_saved_external_evidence") from None
