"""Authenticated closed candidate metadata, never protected execution permission."""

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from . import _external_checkpoint_protocol as checkpoints
from . import execution_preflight as preflight
from . import phiusiil
from ._checkpoint_codec import canonical_bytes
from ._external_provenance_payloads import PROVENANCE_NAMES
from ._phishvn_archive import (
    COPIED_MEMBERS,
    EXPECTED_FORMAT,
    EXPECTED_MEMBERS,
    PhishVNSourcePins,
)
from ._saved_external_bindings import PRIVATE_OUTPUTS

_SOURCE = "data/sources.json"
_IMPLEMENTATIONS = tuple(
    f"src/automated_phishing_detection/{name}.py"
    for name in (
        "_external_source_profile",
        "execution_preflight",
        "phiusiil",
        "protocol_preflight",
        "_phishvn_archive",
        "phishvn_source",
        "saved_phishvn_source",
        "phishvn",
        "internal_external_handoff",
        "_internal_handoff_validation",
        "external_source_provenance",
        "_external_provenance_payloads",
        "_external_source_checkpoints",
        "_external_checkpoint_protocol",
        "_external_checkpoint_io",
        "_external_secondary_checkpoints",
        "bound_external_runtime",
        "external_producer",
        "saved_external_evidence",
        "_saved_external_bindings",
        "_checkpoint_codec",
    )
)


class ExternalSourceProfileError(ValueError):
    """Candidate metadata could not be authenticated without granting access."""


@dataclass(frozen=True)
class CandidateExternalProfile:
    """Immutable metadata; caller construction establishes no access authority."""

    canonical_bytes: bytes = field(repr=False)

    def projection(self) -> dict:
        return json.loads(self.canonical_bytes)

    @property
    def profile_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    @property
    def archive_pins(self) -> PhishVNSourcePins:
        publisher = self.projection()["publisher"]["expected_format"]
        return PhishVNSourcePins(
            publisher["archive_sha256"], publisher["archive_size_bytes"]
        )

    @property
    def suffix_rules_sha256(self) -> str:
        return self.projection()["public_suffix_list"]["sha256"]

    @property
    def protected_evaluation_ready(self) -> bool:
        return False


def _require(condition):
    if not condition:
        raise ExternalSourceProfileError("invalid_external_source_profile")


def _hex(value, size=64):
    return (
        type(value) is str and re.fullmatch(rf"[0-9a-f]{{{size}}}", value) is not None
    )


def _pins(binding):
    _require(type(binding.source_hashes) is tuple)
    pins = {}
    for entry in binding.source_hashes:
        _require(type(entry) is tuple and len(entry) == 2)
        name, digest = entry
        _require(type(name) is str and name not in pins and _hex(digest))
        pins[name] = digest
    _require(set((_SOURCE, *_IMPLEMENTATIONS)) <= set(pins))
    return pins


def _execution(binding, pins):
    _require(_hex(binding.revision, 40) and _hex(binding.contract_sha256))
    _require(type(binding.runtime_json) is str)
    runtime = json.loads(binding.runtime_json)
    _require(
        type(runtime) is dict
        and preflight._canonical_json(runtime) == binding.runtime_json
    )
    return {
        "revision": binding.revision,
        "execution_contract_sha256": binding.contract_sha256,
        "runtime_sha256": sha256(binding.runtime_json.encode()).hexdigest(),
        "source_spec_sha256": pins[_SOURCE],
    }


def _suffix(binding, pins):
    content = preflight._read_regular(binding.root, _SOURCE)
    _require(type(content) is bytes and sha256(content).hexdigest() == pins[_SOURCE])
    return phiusiil._load_source_spec(content)["public_suffix_list"]


def _publisher():
    return {
        "expected_format": dict(EXPECTED_FORMAT),
        "archive_members": sorted(EXPECTED_MEMBERS),
        "manifest_copied_order": COPIED_MEMBERS,
        "table_members": (
            "data/dataset_url.csv",
            *(f"data/splits/url_{split}.csv" for split in ("train", "val", "test")),
        ),
    }


def _retention():
    provenance, scientific = checkpoints.PROVENANCE_ORDER, checkpoints.SCIENTIFIC_ORDER
    _require(len(provenance) == 6 and set(provenance) == PROVENANCE_NAMES)
    _require(len(scientific) == 30 and set(scientific) == PRIVATE_OUTPUTS)
    return {
        "protocol": checkpoints.PROTOCOL,
        "directory": checkpoints.DIRECTORY,
        "provenance_order": provenance,
        "scientific_order": scientific,
    }


def _projection(binding):
    pins = _pins(binding)
    return {
        "schema_version": 1,
        "profile_id": "external-source-candidate-v1",
        "status": "specified_closed_candidate",
        "protected_evaluation_ready": False,
        "protected_evaluation_authorized": False,
        "execution": _execution(binding, pins),
        "publisher": _publisher(),
        "public_suffix_list": _suffix(binding, pins),
        "retention": _retention(),
        "implementation_sha256": {name: pins[name] for name in _IMPLEMENTATIONS},
    }


def resolve_external_source_profile(
    binding: preflight.ExecutionBinding,
) -> CandidateExternalProfile:
    """Resolve fixed public metadata and recheck identity; access remains closed."""
    try:
        _require(type(binding) is preflight.ExecutionBinding)
        preflight.recheck_binding(binding)
        result = CandidateExternalProfile(canonical_bytes(_projection(binding)))
        preflight.recheck_binding(binding)
        return result
    except (ValueError, TypeError, KeyError, OSError, RecursionError):
        raise ExternalSourceProfileError("invalid_external_source_profile") from None
