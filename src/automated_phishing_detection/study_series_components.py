"""Pure public component preservation, never preflight or execution authority."""

from dataclasses import dataclass, field

from . import _study_series_components_context as context
from . import _study_series_components_science as science

_REJECTIONS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RecursionError,
    OverflowError,
)


class SeriesComponentError(ValueError):
    """Public component preservation could not be authenticated."""


@dataclass(frozen=True)
class SeriesComponentTransition:
    """Immutable comparison facts; upstream dataclass construction proves nothing."""

    original_external_bytes: bytes = field(repr=False)
    original_operational_bytes: bytes = field(repr=False)
    current_external_bytes: bytes = field(repr=False)
    current_operational_bytes: bytes = field(repr=False)
    profile_sha256: str

    @property
    def authorizes_execution(self):
        return False


def verify_series_component_transition(
    base,
    external,
    operational,
    profile_bytes,
    audit_summary_bytes,
    *,
    expected_profile_sha256,
):
    """Compare supplied current public metadata; the caller proves genuine preflight."""
    try:
        profile, current_external, current_operational = context.joined(
            base, external, operational, profile_bytes, expected_profile_sha256
        )
        original_external, original_operational = science.components(
            profile, current_external, current_operational
        )
        science.audit(profile, audit_summary_bytes)
        return SeriesComponentTransition(
            original_external,
            original_operational,
            external.canonical_bytes,
            operational.canonical_bytes,
            expected_profile_sha256,
        )
    except _REJECTIONS:
        raise SeriesComponentError("invalid_series_components") from None
