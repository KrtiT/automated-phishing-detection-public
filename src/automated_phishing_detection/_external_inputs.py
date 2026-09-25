"""Check prepared external byte consistency, not original publisher provenance."""

import json
from collections import Counter
from dataclasses import asdict
from hashlib import sha256

from . import fixed_cascade, phishvn
from ._external_input_records import ExternalInputError, _require, validate_positions

_FILES = {"retained-test.jsonl", "quarantine.jsonl", "inventory.json"}
_ROLES = ("gold", "certified", "secondary", "tranco")


def _private(prepared: phishvn.PreparedExternal) -> dict:
    private = prepared.private_outputs
    _require(
        type(private) is dict and set(private) == _FILES, "private_inventory_mismatch"
    )
    _require(
        all(type(value) is bytes for value in private.values()), "invalid_private_bytes"
    )
    hashes = {name: sha256(value).hexdigest() for name, value in private.items()}
    _require(
        hashes == prepared.public_summary["private_sha256"], "private_hash_mismatch"
    )
    for name, rows, row_type in (
        ("retained-test.jsonl", prepared.retained, phishvn.PreparedExternalRow),
        ("quarantine.jsonl", prepared.quarantine, phishvn.ExternalQuarantine),
    ):
        _require(
            type(rows) is tuple and all(type(row) is row_type for row in rows),
            "invalid_prepared_rows",
        )
        content = b"".join(phishvn._json_bytes(asdict(row)) for row in rows)
        _require(content == private[name], "prepared_row_bytes_mismatch")
    return private


def _inventory(content: bytes) -> dict[str, int]:
    value = json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    _require(
        type(value) is dict
        and set(value) == {"declared_split_counts", "test_split", "inventory_binding"},
        "invalid_inventory",
    )
    _require(phishvn._json_bytes(value) == content, "noncanonical_inventory")
    counts = value["declared_split_counts"]
    _require(
        type(counts) is dict
        and set(counts) == {"train", "val", "test"}
        and all(type(count) is int and count >= 0 for count in counts.values())
        and value["test_split"] == "test"
        and value["inventory_binding"] == "caller_supplied_claim_only",
        "invalid_inventory",
    )
    return counts


def _summary_counts(prepared: phishvn.PreparedExternal, counts: dict[str, int]) -> dict:
    rows, quarantine = prepared.retained, prepared.quarantine
    valid_non_test = sum(counts.values()) - len(rows) - len(quarantine)
    _require(valid_non_test >= 0, "invalid_preparation_counts")
    roles = Counter(row.role for row in rows)
    reasons = Counter(reason for row in quarantine for reason in row.reason_codes)
    return {
        "input_row_count": sum(counts.values()),
        "input_test_rows": counts["test"],
        "retained_test_rows": len(rows),
        "valid_non_test_rows": valid_non_test,
        "quarantined_rows": len(quarantine),
        "quarantined_test_rows": sum(row.source_split == "test" for row in quarantine),
        "retained_test_domain_count": len({row.registrable_domain for row in rows}),
        "role_counts": {role: roles[role] for role in _ROLES},
        "quarantine_reason_counts": dict(sorted(reasons.items())),
    }


def _summary(prepared: phishvn.PreparedExternal, counts: dict[str, int]) -> None:
    expected = {
        "schema_version": 1,
        "scope": "normalized_inputs_only",
        "schema_verified": False,
        "protected_evaluation_authorized": False,
        "inventory_binding": "caller_supplied_claim_only",
        "declared_split_count": 3,
        **_summary_counts(prepared, counts),
        "quarantine_count_basis": "reason_incidences_not_disjoint_categories",
        "private_sha256": {
            name: sha256(content).hexdigest()
            for name, content in prepared.private_outputs.items()
        },
    }
    _require(
        fixed_cascade._matches_exactly(prepared.public_summary, expected),
        "preparation_summary_mismatch",
    )


def validate_prepared_external(
    prepared: phishvn.PreparedExternal,
) -> tuple[phishvn.PreparedExternalRow, ...]:
    """Require canonical preparation agreement before scoring; infer no provenance."""
    try:
        _require(
            type(prepared) is phishvn.PreparedExternal, "invalid_prepared_external"
        )
        private = _private(prepared)
        counts = _inventory(private["inventory.json"])
        validate_positions(prepared, counts)
        _summary(prepared, counts)
        return prepared.retained
    except ExternalInputError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        raise ExternalInputError("invalid_external_preparation") from None
