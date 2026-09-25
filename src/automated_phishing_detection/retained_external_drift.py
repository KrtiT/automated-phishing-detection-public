"""Restore the fixed accepted drift chain from retained bytes without I/O.

This validates the accepted public chain and retained snapshots, not execution
authority. The caller still binds primary artifact/contract identities, portable
state and scaler agreement before using the restored references.
"""

from hashlib import sha256

from . import bound_drift, bound_secondary, retained_drift, secondary_development
from .bound_drift import BoundDrift, BoundDriftError
from .secondary_development import DevelopmentPins


def _public_inputs(public: tuple[tuple[str, bytes], ...]) -> dict[str, bytes]:
    names = (bound_drift._REPORT, bound_drift._PREPARATION, bound_drift._SOURCE)
    if (
        type(public) is not tuple
        or len(public) != len(names)
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not bytes
            for pair in public
        )
        or tuple(pair[0] for pair in public) != names
    ):
        raise BoundDriftError("invalid_retained_public_inputs")
    contents = dict(public)
    accepted_name, accepted_hash = bound_secondary.PUBLIC_REPORTS["tabular"]
    bound_drift._require(
        accepted_name == names[0]
        and sha256(contents[names[0]]).hexdigest() == accepted_hash,
        "public_report_not_accepted",
    )
    return contents


def _preparation_chain(contents: dict[str, bytes], pins: DevelopmentPins) -> int:
    preparation = contents[bound_drift._PREPARATION]
    bound_drift._require(
        sha256(preparation).hexdigest() == pins.preparation_summary_sha256,
        "preparation_hash_mismatch",
    )
    source = contents[bound_drift._SOURCE]
    source_hash = sha256(source).hexdigest()
    bound_drift._require(
        secondary_development._json(preparation)["source_spec_sha256"] == source_hash,
        "source_hash_mismatch",
    )
    return bound_drift._preparation(source, preparation, source_hash, pins)


def _reference(
    training: bytes,
    audit: bytes,
    preparation: bytes,
    report: dict,
    pins: DevelopmentPins,
) -> retained_drift.RetainedDriftReference:
    summary, reference_hash, audit_hash = (
        bound_drift.seed_probe_execution._retained_drift(report, pins)
    )
    return retained_drift.load_retained_drift_reference(
        training,
        audit,
        expected_reference_sha256=reference_hash,
        expected_audit_sha256=audit_hash,
        pins=pins,
        preparation_summary=preparation,
        expected_drift_summary=secondary_development._json(summary),
    )


def _restore_bound(
    training_reference: bytes,
    validation_audit: bytes,
    public_inputs: tuple[tuple[str, bytes], ...],
) -> BoundDrift:
    contents = _public_inputs(public_inputs)
    report = secondary_development._json(contents[bound_drift._REPORT])
    pins = bound_drift._declared_pins(report)
    count = _preparation_chain(contents, pins)
    reference = _reference(
        training_reference,
        validation_audit,
        contents[bound_drift._PREPARATION],
        report,
        pins,
    )
    bound_drift._require(
        reference.psi.training_row_count == count, "drift_training_count_mismatch"
    )
    return BoundDrift(reference, training_reference, validation_audit, public_inputs)


def restore_external_drift(
    training_reference: bytes,
    validation_audit: bytes,
    public_inputs: tuple[tuple[str, bytes], ...],
) -> BoundDrift:
    """Authenticate exact retained public bytes before restoring drift state."""
    try:
        return _restore_bound(training_reference, validation_audit, public_inputs)
    except BoundDriftError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        raise BoundDriftError("invalid_retained_external_drift") from None
