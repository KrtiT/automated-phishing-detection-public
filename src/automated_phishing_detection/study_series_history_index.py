"""Authenticate pre-access history locators, never catalog or execution authority."""

from dataclasses import dataclass, field

from . import _study_execution_schema as schema
from . import _study_series_adoption_profile as profiles
from . import _study_series_index_records as records
from . import _study_series_index_refs as refs
from . import _study_series_policy as policy


class SeriesHistoryIndexError(ValueError):
    """Pinned history locator metadata fails its closed consistency contract."""


@dataclass(frozen=True)
class SeriesHistoryIndex:
    """Declared file identities only; construction grants no access capability."""

    index_bytes: bytes = field(repr=False)
    index_sha256: str
    profile_sha256: str
    file_refs: tuple[tuple[str, str], ...] = field(repr=False)
    selected_attempt_ordinal: int
    accepted_ordinals: tuple[int, ...]
    stopped_ordinal: int
    scope: str = field(
        default="closed_series_history_locator_metadata_only", init=False
    )
    authorizes_execution: bool = field(default=False, init=False)


def _result(content, index_pin, profile_pin, value, profile):
    return SeriesHistoryIndex(
        content,
        index_pin,
        profile_pin,
        refs.flatten(value, profile),
        value["selected_attempt_ordinal"],
        tuple(member["ordinal"] for member in value["accepted_cells"]),
        value["stopped_cell"]["ordinal"],
    )


def validate_series_history_index(
    index_bytes,
    profile_bytes,
    *,
    expected_index_sha256,
    expected_profile_sha256,
):
    """Validate original bytes before any referenced file or catalog is accessed."""
    try:
        value = schema.parse(index_bytes, expected_index_sha256)
        profile = schema.parse(profile_bytes, expected_profile_sha256)
        profiles.profile(profile, policy.policy_projection())
        schema.require(profile["history"]["index_sha256"] == expected_index_sha256)
        records.index(value, profile)
        return _result(
            index_bytes, expected_index_sha256, expected_profile_sha256, value, profile
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        raise SeriesHistoryIndexError("invalid_series_history_index") from None
