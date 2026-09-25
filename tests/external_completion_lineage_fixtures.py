"""Invented unit lineage, never proof of an internal research execution."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

from retained_external_drift_fixtures import snapshot_chain
from test_retained_drift import _public_bytes, _rebind

from automated_phishing_detection import _internal_handoff_validation as internal
from automated_phishing_detection import phiusiil, source_overlap
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.internal_external_handoff import (
    InternalHandoffPayloads,
)

OVERLAP_DOMAINS = frozenset({"internal-only.com"})


def digest(content):
    return sha256(content).hexdigest()


def rebound_snapshot_chain(suffix):
    data, source_bytes = snapshot_chain()
    source = json.loads(source_bytes)
    source["public_suffix_list"]["sha256"] = digest(suffix)
    source_bytes = _public_bytes(source)
    preparation = json.loads(data["preparation_summary"])
    preparation["declared_sources"] = source
    preparation["source_spec_sha256"] = digest(source_bytes)
    preparation_bytes = _public_bytes(preparation)
    pins = replace(
        data["pins"],
        suffix_rules_sha256=digest(suffix),
        preparation_summary_sha256=digest(preparation_bytes),
    )
    reference = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    public = data["expected_drift_summary"]
    for document in (reference, audit, public):
        document["input_hashes"] = asdict(pins)
    return _rebind(reference, audit, public, preparation_bytes, pins), source_bytes


def fixture_binding(root, public_inputs):
    return ExecutionBinding(
        root,
        "c" * 40,
        "d" * 64,
        tuple((name, digest(content)) for name, content in public_inputs),
        "{}",
    )


def _execution(binding, suffix, source_csv_sha256):
    pins = dict(binding.source_hashes)
    return {
        **internal.EXECUTION_CONSTANTS,
        **{name: digest(name.encode()) for name in internal.EXECUTION_HASHES},
        "revision": binding.revision,
        "execution_contract_sha256": binding.contract_sha256,
        "runtime_sha256": digest(binding.runtime_json.encode()),
        "source_spec_sha256": pins["data/sources.json"],
        "preparation_summary_sha256": pins["reports/phiusiil-preparation-summary.json"],
        "source_csv_sha256": source_csv_sha256,
        "suffix_rules_sha256": digest(suffix),
    }


def _overlap(execution):
    outputs = {name: digest(name.encode()) for name in internal.PREPARATION_NAMES}
    outputs["group_test.jsonl"] = execution["partition_sha256"]
    row = {
        "source_ordinal": 1,
        "record_id": phiusiil.record_id_for_row(execution["source_csv_sha256"], 1),
        "canonical_url_sha256": digest(b"https://internal-only.com/invented"),
        "registrable_domain": "internal-only.com",
        "status": "valid_url_and_domain",
    }
    return canonical_bytes(
        {
            "schema_version": 1,
            "algorithm_id": source_overlap._ALGORITHM_ID,
            "scope": source_overlap._SCOPE,
            "input_hashes": {name: execution[name] for name in internal.OVERLAP_PINS},
            "reconstructed_output_sha256": outputs,
            "rows": [row],
            "domains": sorted(OVERLAP_DOMAINS),
        }
    )


def unit_handoff(binding, suffix, source_csv_sha256):
    """Declare unit-only internal lineage; no actual internal producer is claimed."""
    execution = _execution(binding, suffix, source_csv_sha256)
    overlap = _overlap(execution)
    hashes = {name: digest(name.encode()) for name in internal.SNAPSHOT_NAMES}
    hashes.update(
        {path: execution[name] for name, path in internal.SOURCE_LINKS.items()}
    )
    hashes[internal.OVERLAP_NAME] = digest(overlap)
    return InternalHandoffPayloads(
        canonical_bytes(
            {
                "schema_version": 1,
                "kind": "same-parent-internal-handoff-v1",
                "execution": execution,
                "snapshot_sha256": hashes,
                "worker": {
                    "command_sha256": digest(b"invented internal unit worker"),
                    "stdout_sha256": digest(b""),
                    "stderr_sha256": digest(b""),
                    "exit": {"pid": 1, "exit_observed": True, "exit_code": 0},
                },
            }
        ),
        overlap,
    )


def fixture_profile(binding, archive_pins, suffix):
    return CandidateExternalProfile(
        canonical_bytes(
            {
                "schema_version": 1,
                "profile_id": "external-source-candidate-v1",
                "status": "specified_closed_candidate",
                "protected_evaluation_ready": False,
                "protected_evaluation_authorized": False,
                "execution": {
                    "revision": binding.revision,
                    "execution_contract_sha256": binding.contract_sha256,
                    "runtime_sha256": digest(binding.runtime_json.encode()),
                    "source_spec_sha256": dict(binding.source_hashes)[
                        "data/sources.json"
                    ],
                },
                "publisher": {"expected_format": archive_pins},
                "public_suffix_list": {"sha256": digest(suffix)},
            }
        )
    )
