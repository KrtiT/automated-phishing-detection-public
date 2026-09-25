"""Closed operational metadata checks confer consistency, never authority."""

import json
import math
import re
from hashlib import sha256

from . import _internal_handoff_validation as internal
from . import fixed_cascade
from ._checkpoint_codec import canonical_bytes
from ._external_checkpoint_protocol import PROTOCOL
from ._external_completion_records import _LOGICAL_NAMES

EXECUTION_FIELDS = frozenset(
    ("revision", "execution_contract_sha256", "runtime_sha256", "source_spec_sha256")
)
PREPARATION_FIELDS = frozenset(
    ("study_preparation_reservation_sha256", "study_preparation_complete_sha256")
)
ARTIFACT_NAMES = frozenset(
    (
        "cascade.json",
        "gmm.json",
        "length-only.json",
        "logistic-l1.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    )
)
THRESHOLD_NAMES = frozenset(
    ("length_only", "logistic_l1", "transformer", "half_width", "monitor_boundary")
)
_EXTERNAL_HASHES = frozenset(
    (
        "source_profile_sha256",
        "archive_sha256",
        "suffix_rules_sha256",
        "internal_handoff_sha256",
        "internal_overlap_sha256",
        "internal_reservation_sha256",
        "reservation_sha256",
    )
)
digest = internal.digest


class OperationalInputError(ValueError):
    """Symbolic rejection without private source values or diagnostics."""


def require(condition):
    if not condition:
        raise OperationalInputError("invalid_operational_inputs")


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected))


def loads(content, *, canonical=True):
    require(type(content) is bytes)
    value = json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    require(type(value) is dict)
    require(not canonical or canonical_bytes(value) == content)
    return value


def authenticated(content, expected):
    require(type(content) is bytes)
    digest(expected)
    require(sha256(content).hexdigest() == expected)
    return loads(content)


def validate_primary(value) -> None:
    keys(value, {"artifact_hashes", "thresholds"})
    keys(value["artifact_hashes"], ARTIFACT_NAMES)
    for value_hash in value["artifact_hashes"].values():
        digest(value_hash)
    thresholds = value["thresholds"]
    keys(thresholds, THRESHOLD_NAMES)
    require(
        all(
            type(number) in (int, float) and math.isfinite(number)
            for number in thresholds.values()
        )
    )
    require(
        all(
            0 <= thresholds[name] <= 1
            for name in ("length_only", "logistic_l1", "transformer")
        )
    )
    require(thresholds["half_width"] >= 0)


def same(first, second):
    require(canonical_bytes(first) == canonical_bytes(second))


def shared_execution(value):
    keys(value, EXECUTION_FIELDS)
    require(
        type(value["revision"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is not None
    )
    for name in EXECUTION_FIELDS - {"revision"}:
        digest(value[name])


def _external_execution(value, original, hashes):
    constants = {"kind": "external_evaluation", "checkpoint_protocol": PROTOCOL}
    prepared = original["source_interface"] == "retained_study_preparation_v1"
    constants["source_interface"] = (
        "retained_study_preparation_v1"
        if prepared
        else "publisher_archive_reconstruction_v1"
    )
    names = _EXTERNAL_HASHES | (PREPARATION_FIELDS if prepared else set())
    keys(value, {*constants, *EXECUTION_FIELDS, *names, "archive_size_bytes"})
    same({name: value[name] for name in constants}, constants)
    for name in names:
        digest(value[name])
    require(
        type(value["archive_size_bytes"]) is int and value["archive_size_bytes"] > 0
    )
    same(
        {name: value[name] for name in EXECUTION_FIELDS},
        {name: original[name] for name in EXECUTION_FIELDS},
    )
    same(value["suffix_rules_sha256"], original["suffix_rules_sha256"])
    require(value["reservation_sha256"] == hashes["attempt/reservation.json"])
    if prepared:
        same(
            {name: value[name] for name in PREPARATION_FIELDS},
            {name: original[name] for name in PREPARATION_FIELDS},
        )


def _source_links(value):
    first, second = value["internal"], value["external"]
    internal.envelope(first)
    keys(second, {"execution", "worker", "snapshot_sha256"})
    internal.worker(second["worker"])
    internal.hash_map(second["snapshot_sha256"], _LOGICAL_NAMES)
    _external_execution(
        second["execution"], first["execution"], second["snapshot_sha256"]
    )
    expected = {
        "internal_handoff_sha256": sha256(canonical_bytes(first)).hexdigest(),
        "internal_overlap_sha256": first["snapshot_sha256"][internal.OVERLAP_NAME],
        "internal_reservation_sha256": first["execution"]["reservation_sha256"],
    }
    same({name: second["execution"][name] for name in expected}, expected)
    for directory in ("checkpoints", "evidence"):
        for name, field in (
            ("internal-source-handoff.json", "internal_handoff_sha256"),
            ("internal-source-overlap.json", "internal_overlap_sha256"),
        ):
            require(
                second["snapshot_sha256"][f"attempt/{directory}/{name}"]
                == expected[field]
            )


def validate_metadata(value):
    keys(
        value,
        {
            "schema_version",
            "kind",
            "root_reservation_sha256",
            "execution",
            "operational_profile_sha256",
            "primary",
            "internal",
            "external",
        },
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "same-parent-operational-inputs-v1")
    digest(value["root_reservation_sha256"])
    digest(value["operational_profile_sha256"])
    shared_execution(value["execution"])
    validate_primary(value["primary"])
    _source_links(value)
    same(
        value["execution"],
        {name: value["internal"]["execution"][name] for name in EXECUTION_FIELDS},
    )
