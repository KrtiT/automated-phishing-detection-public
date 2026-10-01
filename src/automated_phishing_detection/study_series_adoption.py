"""Pure pre-access series metadata authentication, never execution authority."""

from dataclasses import dataclass, field

from . import _study_execution_schema as legacy
from . import _study_series_adoption_authority as authority
from . import _study_series_adoption_profile as profile_schema
from . import _study_series_policy as policy


class SeriesAdoptionError(ValueError):
    """Pinned metadata fails the separate closed series header contract."""


@dataclass(frozen=True)
class SeriesAdoptionHeader:
    """Authenticated declarations only; construction supplies no capability."""

    profile_bytes: bytes = field(repr=False)
    policy_sha256: str
    profile_sha256: str
    envelope_sha256: str
    history_index_sha256: str
    execution_revision: str
    segment_ordinal: int
    start_ordinal: int
    end_ordinal: int
    scope: str = field(default="closed_series_adoption_metadata_only", init=False)
    authorizes_execution: bool = field(default=False, init=False)


def _facts(content, value, profile_pin, envelope_pin):
    segment = value["segment"]
    return SeriesAdoptionHeader(
        content,
        value["policy_sha256"],
        profile_pin,
        envelope_pin,
        value["history"]["index_sha256"],
        value["execution"]["revision"],
        segment["ordinal"],
        segment["start_ordinal"],
        segment["end_ordinal"],
    )


def validate_series_adoption_header(
    policy_bytes,
    profile_bytes,
    envelope_bytes,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
):
    """Authenticate original pinned bytes without reading any referenced path."""
    try:
        value = legacy.parse(profile_bytes, expected_profile_sha256)
        envelope = legacy.parse(envelope_bytes, expected_envelope_sha256)
        legacy.require(type(value) is dict)
        selected_policy = legacy.parse(policy_bytes, value["policy_sha256"])
        legacy.require(policy_bytes == policy.policy_bytes())
        profile_schema.profile(value, selected_policy)
        authority.envelope(envelope, value)
        return _facts(
            profile_bytes, value, expected_profile_sha256, expected_envelope_sha256
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        raise SeriesAdoptionError("invalid_series_adoption_header") from None
