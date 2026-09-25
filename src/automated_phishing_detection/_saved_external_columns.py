"""Restore twelve completed external columns without constructing model members."""

from hashlib import sha256

from . import bound_secondary as secondary
from ._external_secondary_checkpoints import (
    SECONDARY_CHECKPOINTS,
    CompletedColumn,
    column_bytes,
    completion_bytes,
    project_member_bindings,
)
from ._external_secondary_inputs import validate_primary_phase
from ._external_secondary_validation import (
    TABULAR_NAMES,
    validate_column,
    validate_secondary_scoring,
)
from ._saved_external_inputs import _array, _fields, _require
from .external_primary import ExternalPrimaryScores
from .external_producer import _all_scores
from .saved_evidence import SavedEvidenceError, _loads


def _column(value: object, index: int) -> CompletedColumn:
    if index < len(TABULAR_NAMES):
        column_type = secondary.CompletedTabularColumn
        score_type = secondary.SecondaryTabularScore
    else:
        column_type = secondary.CompletedSeedColumn
        score_type = secondary.SecondarySeedScore
    values = _fields(value, column_type.__dataclass_fields__)
    values["scores"] = tuple(
        score_type(**_fields(score, score_type.__dataclass_fields__))
        for score in _array(values["scores"])
    )
    return column_type(**values)


def _restore_columns(
    outputs: dict[str, bytes], primary: ExternalPrimaryScores, bindings: dict
) -> tuple[CompletedColumn, ...]:
    members = project_member_bindings(bindings["secondary"])
    _require(len(members) == len(SECONDARY_CHECKPOINTS))
    primary_hash = sha256(primary.checkpoint_bytes).hexdigest()
    record_ids = tuple(record.record_id for record in primary.records)
    columns = []
    for index, (name, member) in enumerate(
        zip(SECONDARY_CHECKPOINTS, members, strict=True)
    ):
        content = outputs[name]
        envelope = _fields(
            _loads(content, "secondary_column"),
            (
                "schema_version",
                "primary_scores_sha256",
                "record_ids",
                "binding",
                "column",
            ),
        )
        column = _column(envelope["column"], index)
        validate_column(column, index, len(record_ids))
        _require(content == column_bytes(primary_hash, record_ids, member, column))
        columns.append(column)
    return tuple(columns)


def restore_secondary(
    outputs: dict[str, bytes], primary: ExternalPrimaryScores, bindings: dict
) -> secondary.SecondaryScoring:
    """Restore structural phase agreement, not physical execution authenticity."""
    try:
        _require(type(outputs) is dict)
        validate_primary_phase(primary)
        columns = _restore_columns(outputs, primary, bindings)
        scoring = secondary._scoring_result(columns, len(primary.records))
        validate_secondary_scoring(scoring, len(primary.records))
        primary_hash = sha256(primary.checkpoint_bytes).hexdigest()
        _require(
            type(outputs["secondary-completion.json"]) is bytes
            and outputs["secondary-completion.json"]
            == completion_bytes(primary_hash, scoring, outputs)
        )
        _require(
            type(outputs["all-scores.jsonl"]) is bytes
            and outputs["all-scores.jsonl"] == _all_scores(primary, scoring)
        )
        return scoring
    except Exception:
        raise SavedEvidenceError("invalid_saved_external_columns") from None
