"""Closed canonical identity for the four live adopted-study continuations."""

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256


class StudyAdmissionError(ValueError):
    def __init__(self, check_id):
        self.check_id = check_id
        super().__init__(check_id)


@dataclass(frozen=True)
class AdmissionFrame:
    role: str
    profile_sha256: str
    envelope_sha256: str
    parent_pid: int
    command_sha256: str
    root_reservation_sha256: str
    intent_sha256: str
    barrier_sha256: str
    preparation_reservation_sha256: str
    preparation_completion_sha256: str
    predecessor_sha256: str | None
    accepted_inputs_sha256: str | None
    cell_binding_sha256: str | None

    def __post_init__(self):
        validate_admission_frame(self)

    @property
    def canonical_bytes(self):
        value = dict(schema_version="study-admission-v1", **asdict(self))
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")

    @property
    def sha256(self):
        return sha256(self.canonical_bytes).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_admission_frame(frame):
    if type(frame) is not AdmissionFrame:
        raise StudyAdmissionError("invalid_admission_frame")
    if frame.role not in ("internal", "external", "service", "client"):
        raise StudyAdmissionError("invalid_admission_role")
    if type(frame.parent_pid) is not int or frame.parent_pid <= 0:
        raise StudyAdmissionError("invalid_admission_parent")
    nullable = {
        "predecessor_sha256": frame.role == "internal",
        "accepted_inputs_sha256": frame.role in ("internal", "external"),
        "cell_binding_sha256": frame.role in ("internal", "external"),
    }
    for name, value in asdict(frame).items():
        if not name.endswith("_sha256"):
            continue
        valid = value is None if nullable.get(name, False) else _digest(value)
        if not valid:
            raise StudyAdmissionError("invalid_admission_digest")


def decode_admission_frame(content):
    if type(content) is not bytes or not 0 < len(content) <= 4096:
        raise StudyAdmissionError("invalid_admission_frame")
    try:
        value = json.loads(content)
        if (
            type(value) is not dict
            or value.pop("schema_version") != "study-admission-v1"
        ):
            raise ValueError()
        frame = AdmissionFrame(**value)
        if frame.canonical_bytes != content:
            raise ValueError()
        return frame
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise StudyAdmissionError("invalid_admission_frame") from None
