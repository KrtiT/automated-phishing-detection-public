"""Closed retained-preparation envelopes and independently pinned public buffers."""

import json
import re
from hashlib import sha256

from . import baselines, fixed_cascade, phiusiil
from ._checkpoint_codec import canonical_bytes
from ._study_preparation_records import PreparedStudySnapshot
from .study_preparation_retention import PREPARATION_ORDER

_HASH_FIELDS = frozenset(
    {
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
        "source_profile_sha256",
        "preparation_summary_sha256",
        "source_csv_sha256",
        "suffix_rules_sha256",
        "partition_sha256",
        "archive_sha256",
    }
)
_IDENTITY_FIELDS = _HASH_FIELDS | {"kind", "protocol", "revision", "archive_size_bytes"}
_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "protected_evaluation_authorized",
        "scoring_authorized",
        "execution",
        "reservation_sha256",
        "input_sha256",
    }
)


class StudyPreparationRestoreError(ValueError):
    """Fixed symbolic rejection without private parser diagnostics."""


def require(condition):
    if not condition:
        raise StudyPreparationRestoreError("invalid_retained_study_preparation")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def load(content):
    require(type(content) is bytes)
    return json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )


def identity(value):
    require(type(value) is dict and set(value) == _IDENTITY_FIELDS)
    require(value["kind"] == "study_preparation_only")
    require(value["protocol"] == "study-preparation-v1")
    require(type(value["kind"]) is str and type(value["protocol"]) is str)
    require(type(value["revision"]) is str)
    require(re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is not None)
    require(
        type(value["archive_size_bytes"]) is int and value["archive_size_bytes"] > 0
    )
    for name in _HASH_FIELDS:
        digest(value[name])
    return load(canonical_bytes(value))


def _payloads(snapshot, reservation):
    require(type(snapshot) is PreparedStudySnapshot)
    require(type(snapshot.reservation_sha256) is str)
    require(snapshot.reservation_sha256 == reservation)
    payloads = snapshot.payloads
    require(type(payloads) is tuple and len(payloads) == len(PREPARATION_ORDER))
    require(all(type(member) is tuple and len(member) == 2 for member in payloads))
    require(
        all(type(name) is str and type(content) is bytes for name, content in payloads)
    )
    require(tuple(name for name, unused in payloads) == PREPARATION_ORDER)
    return dict(payloads)


def authenticate(snapshot, expected_identity, reservation, completion):
    digest(reservation)
    digest(completion)
    execution = identity(expected_identity)
    outputs = _payloads(snapshot, reservation)
    content = outputs["preparation-complete.json"]
    require(sha256(content).hexdigest() == completion)
    envelope = load(content)
    require(type(envelope) is dict and set(envelope) == _COMPLETION_FIELDS)
    expected = {
        "schema_version": 1,
        "protocol": "study-preparation-v1",
        "status": "preparation_only",
        "protected_evaluation_authorized": False,
        "scoring_authorized": False,
        "execution": execution,
        "reservation_sha256": reservation,
        "input_sha256": {
            name: sha256(outputs[name]).hexdigest() for name in PREPARATION_ORDER[:-1]
        },
    }
    require(content == canonical_bytes(expected))
    return outputs, execution


def public_inputs(outputs, execution, source_bytes, report_bytes):
    for content, name in (
        (source_bytes, "source_spec_sha256"),
        (report_bytes, "preparation_summary_sha256"),
        (outputs["suffix-rules.dat"], "suffix_rules_sha256"),
    ):
        require(
            type(content) is bytes and sha256(content).hexdigest() == execution[name]
        )
    source = phiusiil._load_source_spec(source_bytes)
    report = load(report_bytes)
    preparation = baselines._validate_preparation_summary(report)
    require(report["source_spec_sha256"] == execution["source_spec_sha256"])
    require(phiusiil._matches_exactly(report["declared_sources"], source))
    require(source["phiusiil"]["csv_sha256"] == execution["source_csv_sha256"])
    require(source["public_suffix_list"]["sha256"] == execution["suffix_rules_sha256"])
    require(
        preparation["output_hashes"]["group_test.jsonl"]
        == execution["partition_sha256"]
    )
    split = preparation["splits"]["group_test"]
    return {
        "expected_sha256": execution["partition_sha256"],
        "source_csv_sha256": execution["source_csv_sha256"],
        "suffix_rules_sha256": execution["suffix_rules_sha256"],
        "expected_row_count": split["row_count"],
        "expected_domain_count": split["domain_count"],
        "expected_class_counts": dict(split["class_counts"]),
    }, report
