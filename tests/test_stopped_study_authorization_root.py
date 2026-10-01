"""Failed-root inventory, independent pins and original finalization stay closed."""

import json
from dataclasses import replace

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import (
    make_stopped,
    refresh_accounting,
    refresh_snapshot,
    verify,
)
from study_run_record_fixtures import prepared

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._adopted_study_ledger_validation import (
    validate_ledger,
)
from automated_phishing_detection._adopted_study_verification import (
    verify_saved_adopted_authorization,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "list", "member_list", "bytearray"]
)
def test_snapshot_inventory_is_exact(prepared, manifests, change):
    case = make_stopped(prepared, manifests)
    payloads = case.snapshot.payloads
    if change == "missing":
        payloads = payloads[:-1]
    elif change == "extra":
        payloads += (("public-summary.json", b"{}"),)
    elif change == "duplicate":
        payloads += (payloads[0],)
    elif change == "list":
        payloads = list(payloads)
    elif change == "member_list":
        payloads = (list(payloads[0]),) + payloads[1:]
    else:
        payloads = ((payloads[0][0], bytearray(payloads[0][1])),) + payloads[1:]
    case.snapshot = replace(case.snapshot, payloads=payloads)
    with pytest.raises(ValueError, match="^invalid_stopped_study_authorization$"):
        verify(case)


@pytest.mark.parametrize("pin", ["root", "profile", "envelope", "snapshot"])
def test_independent_pins_are_required(prepared, manifests, pin):
    case = make_stopped(prepared, manifests)
    overrides = {}
    if pin == "root":
        case.snapshot = replace(case.snapshot, reservation_sha256="0" * 64)
    elif pin == "snapshot":
        case.pins["attempt/study-accounting.json"] = "0" * 64
    else:
        overrides[f"expected_{pin}_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify(case, **overrides)


@pytest.mark.parametrize("change", ["missing", "extra", "invalid", "not_mapping"])
def test_independent_hash_inventory_is_closed(prepared, manifests, change):
    case = make_stopped(prepared, manifests)
    if change == "missing":
        case.pins.pop("attempt/study-accounting.json")
    elif change == "extra":
        case.pins["attempt/evidence/study-accounting.json"] = "0" * 64
    elif change == "invalid":
        case.pins["attempt/study-accounting.json"] = b"0" * 64
    else:
        case.pins = tuple(case.pins.items())
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "member,field,value",
    [
        ("finalize.claim", "operation", "completion"),
        ("finalize.claim", "reservation_sha256", "0" * 64),
        ("outcome.json", "status", "whole_study_hold"),
        ("outcome.json", "error_type", "exception"),
        ("outcome.json", "stage", "reduction"),
        ("outcome.json", "reservation_sha256", "0" * 64),
        ("outcome.json", "extra", True),
    ],
)
def test_rehashed_finalization_must_be_exact(prepared, manifests, member, field, value):
    case = make_stopped(prepared, manifests)
    name = f"attempt/{member}"
    record = json.loads(case.payloads[name]) | {field: value}
    case.payloads[name] = receipt._json_bytes(record, "fixture")
    refresh_snapshot(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "member", ["finalize.claim", "outcome.json", "reservation.json"]
)
def test_receipt_codec_rejects_trailing_newline(prepared, manifests, member):
    case = make_stopped(prepared, manifests)
    case.payloads[f"attempt/{member}"] += b"\n"
    refresh_snapshot(case)
    with pytest.raises(ValueError):
        verify(case)


def test_rehashed_joins_do_not_replace_independent_pin(prepared, manifests):
    case = make_stopped(prepared, manifests)
    original_pins = case.pins.copy()
    case.scientific["cells"][1]["progress_sha256"] = "0" * 64
    refresh_accounting(case)
    verify(case)
    with pytest.raises(ValueError):
        verify(case, expected_snapshot_sha256=original_pins)


def test_legacy_routes_still_reject_failed_prefix(prepared, manifests):
    case = make_stopped(prepared, manifests)
    with pytest.raises(ValueError):
        verify_saved_adopted_authorization(
            case.snapshot,
            expected_profile_sha256=case.authorization.profile_sha256,
            expected_envelope_sha256=case.authorization.envelope_sha256,
        )
    with pytest.raises(ValueError):
        validate_ledger(
            case.accounting["authorization_ledger"],
            case.execution,
            case.contents,
            False,
        )


@pytest.mark.parametrize("member", ["prediction-barrier.json", "source-results.json"])
def test_wrong_source_or_barrier_status_fails(prepared, manifests, member):
    case = make_stopped(prepared, manifests)
    value = json.loads(case.contents[member]) | {"status": "failed"}
    case.payloads[f"attempt/{member}"] = canonical_bytes(value)
    refresh_snapshot(case)
    with pytest.raises(ValueError):
        verify(case)
