"""Byte-only consistency checks for external identity and public projections."""

import json
import re
from hashlib import sha256

from . import _external_source_profile as profiles
from ._checkpoint_codec import canonical_bytes
from ._external_provenance_payloads import PROVENANCE_NAMES
from ._saved_external_bindings import PRIVATE_OUTPUTS
from .execution_preflight import ExecutionBinding

_SOURCE = "data/sources.json"
PREPARATION = "reports/phiusiil-preparation-summary.json"
_CANDIDATE = {
    "schema_version": 1,
    "profile_id": "external-source-candidate-v1",
    "status": "specified_closed_candidate",
    "protected_evaluation_ready": False,
    "protected_evaluation_authorized": False,
}


class ExternalSourceExecutionError(ValueError):
    """A symbolic external source failure without private values or diagnostics."""


def require(condition):
    if not condition:
        raise ExternalSourceExecutionError("invalid_external_source_execution")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def loads(content):
    require(type(content) is bytes)
    result = json.loads(content)
    require(type(result) is dict and canonical_bytes(result) == content)
    return result


def pins(binding):
    require(type(binding.source_hashes) is tuple)
    result = {}
    for entry in binding.source_hashes:
        require(type(entry) is tuple and len(entry) == 2)
        name, value = entry
        require(type(name) is str and name not in result)
        digest(value)
        result[name] = value
    require({_SOURCE, PREPARATION} <= set(result))
    return result


def context(binding, profile):
    require(type(binding) is ExecutionBinding)
    require(type(profile) is profiles.CandidateExternalProfile)
    projection = loads(profile.canonical_bytes)
    metadata = {name: projection[name] for name in _CANDIDATE}
    require(canonical_bytes(metadata) == canonical_bytes(_CANDIDATE))
    bound = pins(binding)
    require(
        canonical_bytes(projection["execution"])
        == canonical_bytes(profiles._execution(binding, bound))
    )
    archive = profile.archive_pins
    digest(archive.archive_sha256)
    require(type(archive.archive_size_bytes) is int and archive.archive_size_bytes > 0)
    digest(profile.suffix_rules_sha256)
    return projection, bound


def snapshot(outputs):
    require(type(outputs) is dict)
    result = outputs.copy()
    require(set(result) == PROVENANCE_NAMES | PRIVATE_OUTPUTS)
    require(
        all(
            type(name) is str and type(content) is bytes
            for name, content in result.items()
        )
    )
    return result


def hashes(outputs):
    return {name: sha256(content).hexdigest() for name, content in outputs.items()}


def composition(value, outputs):
    require(type(value) is dict)
    copied = json.loads(canonical_bytes(value))
    require(copied["protected_evaluation_authorized"] is False)
    require(copied["source_binding"] == "caller_supplied_preparation_only")
    expected = hashes({name: outputs[name] for name in PRIVATE_OUTPUTS})
    require(canonical_bytes(copied["private_sha256"]) == canonical_bytes(expected))
    return copied
