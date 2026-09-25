"""Pure completion-link authentication; metadata consistency grants no authority.

The parent establishes observed successful exit before supplying these bytes.
Scientific and provenance reconstruction remain independent subsequent checks.
"""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from . import _external_source_records as records
from . import execution_receipt as receipt
from ._external_completion_files import PAYLOAD_NAMES, ExternalFileSnapshot
from ._external_source_profile import CandidateExternalProfile
from ._prepared_external_records import match_prepared_outputs
from .execution_preflight import ExecutionBinding
from .internal_external_handoff import InternalHandoffPayloads

_LOGICAL_NAMES = frozenset(
    {
        "attempt/reservation.json",
        "attempt/finalize.claim",
        "attempt/outcome.json",
        "public-summary.json",
        *(
            f"attempt/{directory}/{name}"
            for directory in ("checkpoints", "evidence")
            for name in PAYLOAD_NAMES
        ),
    }
)


class ExternalCompletionVerificationError(ValueError):
    """A symbolic retained-completion rejection without private diagnostics."""


@dataclass(frozen=True)
class ExternalCompletionRecords:
    identity: dict
    reservation_sha256: str
    public: dict = field(repr=False)
    private_outputs: dict[str, bytes] = field(repr=False)


def _require(condition):
    if not condition:
        raise ExternalCompletionVerificationError("invalid_external_completion_records")


def _encoded(value):
    return receipt._json_bytes(value, "external_completion")


def _contents(files):
    _require(type(files) is ExternalFileSnapshot and type(files.payloads) is tuple)
    result = {}
    for entry in files.payloads:
        _require(type(entry) is tuple and len(entry) == 2)
        name, content = entry
        _require(type(name) is str and type(content) is bytes and name not in result)
        result[name] = content
    _require(
        len(PAYLOAD_NAMES) == 36 and len(result) == 76 and set(result) == _LOGICAL_NAMES
    )
    return result


def _reservation(contents, attempt, identity):
    expected = {
        "schema_version": 1,
        "status": "reserved",
        "directory": str(receipt._absolute_path(attempt)),
        "identity": identity,
    }
    content = contents["attempt/reservation.json"]
    _require(content == _encoded(expected))
    return sha256(content).hexdigest()


def _claim(contents, reservation):
    _require(
        contents["attempt/finalize.claim"]
        == _encoded(
            {
                "schema_version": 1,
                "reservation_sha256": reservation,
                "operation": "completion",
            }
        )
    )


def _private(contents):
    outputs = {}
    for name in PAYLOAD_NAMES:
        content = contents[f"attempt/evidence/{name}"]
        _require(content == contents[f"attempt/checkpoints/{name}"])
        outputs[name] = content
    return outputs


def _public(contents, binding, profile, identity, reservation, outputs, preparation):
    saved = json.loads(contents["public-summary.json"])
    _require(type(saved) is dict)
    public = records.build_external_public(
        binding,
        profile,
        identity,
        reservation,
        outputs,
        saved["composition"],
        preparation=preparation,
    )
    _require(_encoded(public) == contents["public-summary.json"])
    return public


def _outcome(contents, reservation, outputs):
    expected = {
        "schema_version": 1,
        "status": "completion_prepared",
        "reservation_sha256": reservation,
        "public_summary_sha256": sha256(contents["public-summary.json"]).hexdigest(),
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in outputs.items()
        },
    }
    _require(contents["attempt/outcome.json"] == _encoded(expected))


def _authenticate(files, attempt, binding, profile, handoff, preparation):
    contents = _contents(files)
    identity = records.external_identity(
        binding, profile, handoff, preparation=preparation
    )
    reservation = _reservation(contents, attempt, identity)
    _claim(contents, reservation)
    outputs = _private(contents)
    if preparation is not None:
        match_prepared_outputs(outputs, preparation)
    public = _public(
        contents, binding, profile, identity, reservation, outputs, preparation
    )
    _outcome(contents, reservation, outputs)
    return ExternalCompletionRecords(identity, reservation, public, outputs)


def authenticate_external_records(
    files: ExternalFileSnapshot,
    attempt: Path,
    *,
    binding: ExecutionBinding,
    profile: CandidateExternalProfile,
    handoff: InternalHandoffPayloads,
    preparation=None,
) -> ExternalCompletionRecords:
    """Check exact saved links against independently retained parent expectations."""
    try:
        return _authenticate(files, attempt, binding, profile, handoff, preparation)
    except Exception:
        raise ExternalCompletionVerificationError(
            "invalid_external_completion_records"
        ) from None
