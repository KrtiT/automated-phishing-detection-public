"""Private fixed-role admission transport for adopted study execution."""

from ._study_admission_child import ChildAdmission, consume_child_admission
from ._study_admission_frame import (
    AdmissionFrame,
    StudyAdmissionError,
    decode_admission_frame,
    validate_admission_frame,
)
from ._study_admission_parent import ParentAdmission

__all__ = [
    "AdmissionFrame",
    "ChildAdmission",
    "ParentAdmission",
    "StudyAdmissionError",
    "decode_admission_frame",
    "consume_child_admission",
    "validate_admission_frame",
]
