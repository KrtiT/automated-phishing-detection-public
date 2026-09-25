"""Closed root publication joins; metadata consistency never grants authority."""

import json
from dataclasses import dataclass, field
from hashlib import sha256

from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes

ORDER = (
    "study-intent.json",
    "prediction-barrier.json",
    "source-results.json",
    "study-accounting.json",
)
HOLD_ORDER = (ORDER[0], ORDER[1], ORDER[3])
EXTRA_NAMES = ("operational-summary.json", "study-evidence.json")


class StudyRootRetentionError(ValueError):
    """Symbolic root retention failure without private diagnostics."""


@dataclass(frozen=True)
class StudyRootSnapshot:
    reservation_sha256: str
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)

    def payload(self, name):
        for retained, content in self.payloads:
            if retained == name:
                return content
        raise KeyError(name)


def require(condition):
    if not condition:
        raise StudyRootRetentionError("invalid_study_root_retention")


def decode(content):
    require(type(content) is bytes)
    result = json.loads(content)
    require(type(result) is dict and canonical_bytes(result) == content)
    receipt._json_value(result)
    return result


def reservation(attempt, identity):
    require(type(attempt) is receipt.Attempt)
    content = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(receipt._absolute_path(attempt.directory)),
            "identity": json.loads(receipt._json_bytes(identity, "identity")),
        },
        "study_reservation",
    )
    require(sha256(content).hexdigest() == attempt.reservation_sha256)
    return content


def next_name(contents, name):
    require(type(name) is str and name not in contents)
    require(ORDER[-1] not in contents)
    if name == ORDER[-1]:
        return
    names = (*contents, name)
    require(any(names == order[: len(names)] for order in (ORDER, HOLD_ORDER)))


def private_outputs(contents, extra):
    require(type(extra) is dict)
    success = tuple(contents) == ORDER
    require(tuple(contents) in (ORDER, HOLD_ORDER))
    require(set(extra) == (set(EXTRA_NAMES) if success else set()))
    for content in (*contents.values(), *extra.values()):
        decode(content)
    return contents | extra, success


def _barrier(contents, execution, success):
    barrier = decode(contents[ORDER[1]])
    require(barrier["predictions_started"] is False)
    require(
        barrier["status"]
        == ("necessary_capacity_present" if success else "whole_study_hold")
    )
    require(canonical_bytes(barrier["execution"]) == canonical_bytes(execution))
    feasibility = barrier["feasibility"]
    require(type(feasibility) is dict and type(feasibility["shortages"]) is list)
    require(bool(feasibility["shortages"]) is not success)
    require(
        barrier["feasibility_sha256"]
        == sha256(canonical_bytes(feasibility)).hexdigest()
    )
    return feasibility


def public_bytes(attempt, identity, contents, outputs, success, public):
    execution = identity | {"reservation_sha256": attempt.reservation_sha256}
    feasibility = _barrier(contents, execution, success)
    expected = {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "study_evidence_published" if success else "whole_study_hold",
        "execution": execution,
        "accounting_sha256": sha256(contents[ORDER[3]]).hexdigest(),
        "private_sha256": hashes(outputs),
    }
    expected.update(
        {
            "operational": decode(outputs[EXTRA_NAMES[0]]),
            "study": decode(outputs[EXTRA_NAMES[1]]),
        }
        if success
        else {"feasibility": feasibility}
    )
    content = receipt._json_bytes(public, "study_public")
    require(content == receipt._json_bytes(expected, "study_public"))
    return content


def hashes(outputs):
    return {name: sha256(content).hexdigest() for name, content in outputs.items()}


def publication(attempt, outputs, public):
    reservation_hash = attempt.reservation_sha256
    return {
        "finalize.claim": receipt._json_bytes(
            {
                "schema_version": 1,
                "reservation_sha256": reservation_hash,
                "operation": "completion",
            },
            "claim",
        ),
        "outcome.json": receipt._json_bytes(
            {
                "schema_version": 1,
                "status": "completion_prepared",
                "reservation_sha256": reservation_hash,
                "public_summary_sha256": sha256(public).hexdigest(),
                "private_sha256": hashes(outputs),
            },
            "outcome",
        ),
    }
