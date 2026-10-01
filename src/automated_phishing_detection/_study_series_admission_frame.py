"""Separate fixed segment/role wire identity, never execution authorization."""

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from ._study_admission_frame import StudyAdmissionError


@dataclass(frozen=True)
class SeriesAdmissionFrame:
    role: str
    profile_sha256: str
    envelope_sha256: str
    parent_pid: int
    command_sha256: str
    series_reservation_sha256: str
    segment_reservation_sha256: str
    origin_reservation_sha256: str
    history_index_sha256: str
    intent_sha256: str
    predecessor_sha256: str
    accepted_inputs_sha256: str
    cell_binding_sha256: str
    segment_ordinal: int
    cell_ordinal: int

    def __post_init__(self):
        validate_series_frame(self)

    @property
    def canonical_bytes(self):
        value = dict(schema_version="study-series-admission-v1", **asdict(self))
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")

    @property
    def sha256(self):
        return sha256(self.canonical_bytes).hexdigest()


def require(condition, reason="invalid_series_admission"):
    if not condition:
        raise StudyAdmissionError(reason)


def validate_series_frame(frame):
    require(type(frame) is SeriesAdmissionFrame)
    require(type(frame.role) is str and frame.role in ("service", "client"))
    require(type(frame.parent_pid) is int and frame.parent_pid > 0)
    require(type(frame.segment_ordinal) is int and frame.segment_ordinal == 2)
    require(type(frame.cell_ordinal) is int and 2 <= frame.cell_ordinal <= 125)
    for name, value in asdict(frame).items():
        if name.endswith("_sha256"):
            require(
                type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None
            )


def decode_series_admission(content):
    require(type(content) is bytes and 0 < len(content) <= 4096)
    try:
        value = json.loads(content)
        require(type(value) is dict)
        require(value.pop("schema_version") == "study-series-admission-v1")
        frame = SeriesAdmissionFrame(**value)
        require(frame.canonical_bytes == content)
        return frame
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise StudyAdmissionError("invalid_series_admission_frame") from None
