"""Separate series transport; no ordered ledger, binder or access route is exposed."""

from ._study_series_admission_child import (
    SeriesChildAdmission,
    consume_series_admission,
)
from ._study_series_admission_frame import (
    SeriesAdmissionFrame,
    decode_series_admission,
    validate_series_frame,
)
from ._study_series_admission_parent import (
    SeriesParentAdmission,
    validate_series_launch,
)

__all__ = [
    "SeriesAdmissionFrame",
    "SeriesChildAdmission",
    "SeriesParentAdmission",
    "consume_series_admission",
    "decode_series_admission",
    "validate_series_frame",
    "validate_series_launch",
]
