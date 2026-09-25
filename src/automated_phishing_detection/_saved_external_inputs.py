"""Restore caller-bound external preparation and primary observations from bytes."""

from collections.abc import Iterable
from dataclasses import asdict

from . import fixed_cascade, phishvn
from ._external_inputs import validate_prepared_external
from ._external_secondary_inputs import _THRESHOLDS, validate_primary_phase
from .external_primary import ExternalPrimaryScores
from .primary_scores import PrimaryURLScores
from .saved_evidence import SavedEvidenceError, _loads
from .selective_inference import InferenceCounts


def _require(condition: bool) -> None:
    if not condition:
        raise SavedEvidenceError("invalid_saved_external_inputs")


def _fields(value: object, names: Iterable[str]) -> dict:
    _require(type(value) is dict and set(value) == set(names))
    return dict(value)


def _array(value: object) -> tuple:
    _require(type(value) is list)
    return tuple(value)


def _rows(content: bytes) -> tuple:
    _require(type(content) is bytes)
    return tuple(_loads(line, "external_row") for line in content.splitlines(True))


def _quarantine(value: object) -> phishvn.ExternalQuarantine:
    values = _fields(value, phishvn.ExternalQuarantine.__dataclass_fields__)
    values["reason_codes"] = _array(values["reason_codes"])
    return phishvn.ExternalQuarantine(**values)


def restore_preparation(outputs: dict[str, bytes]) -> phishvn.PreparedExternal:
    """Restore exact retained order, without claiming publisher authentication."""
    try:
        _require(type(outputs) is dict)
        retained = tuple(
            phishvn.PreparedExternalRow(
                **_fields(value, phishvn.PreparedExternalRow.__dataclass_fields__)
            )
            for value in _rows(outputs["retained-test.jsonl"])
        )
        quarantine = tuple(
            _quarantine(value) for value in _rows(outputs["quarantine.jsonl"])
        )
        private = {
            name: outputs[name]
            for name in ("retained-test.jsonl", "quarantine.jsonl", "inventory.json")
        }
        prepared = phishvn.PreparedExternal(
            retained,
            quarantine,
            private,
            _loads(outputs["preparation-summary.json"], "external_preparation"),
        )
        validate_prepared_external(prepared)
        return prepared
    except Exception:
        raise SavedEvidenceError("invalid_saved_external_preparation") from None


def _counts(value: object) -> InferenceCounts:
    return InferenceCounts(**_fields(value, InferenceCounts.__dataclass_fields__))


def _score(value: object) -> PrimaryURLScores:
    values = _fields(value, PrimaryURLScores.__dataclass_fields__)
    values["features"] = _array(values["features"])
    values["inference_counts"] = _counts(values["inference_counts"])
    return PrimaryURLScores(**values)


def _primary_scores(
    content: bytes, records: tuple[phishvn.PreparedExternalRow, ...]
) -> tuple[PrimaryURLScores, ...]:
    rows = _rows(content)
    _require(len(rows) == len(records))
    scores = []
    for value, record in zip(rows, records, strict=True):
        row = _fields(value, ("record", "primary"))
        _require(fixed_cascade._matches_exactly(row["record"], asdict(record)))
        scores.append(_score(row["primary"]))
    return tuple(scores)


def restore_primary(
    outputs: dict[str, bytes], prepared: phishvn.PreparedExternal, bindings: dict
) -> ExternalPrimaryScores:
    """Restore observations against prepared records and already verified bindings."""
    try:
        _require(type(outputs) is dict and type(prepared) is phishvn.PreparedExternal)
        receipt = _loads(outputs["primary-completion.json"], "primary_completion")
        thresholds = _fields(bindings["thresholds"], _THRESHOLDS)
        primary = ExternalPrimaryScores(
            prepared.retained,
            _primary_scores(outputs["primary-scores.jsonl"], prepared.retained),
            tuple((name, thresholds[name]) for name in _THRESHOLDS),
            _counts(receipt["inference_counts"]),
            outputs["primary-scores.jsonl"],
            outputs["primary-completion.json"],
        )
        validate_primary_phase(primary)
        return primary
    except Exception:
        raise SavedEvidenceError("invalid_saved_external_primary") from None
