"""Closed metadata checks use only invented declarations, never real authority."""

import copy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest
from study_series_adoption_fixtures import (
    api,
    digest,
    make_case,
    refresh,
    selected,
    validate,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

CLOSED = (
    "profile",
    "profile.execution",
    "profile.origin",
    "profile.components",
    "profile.transition",
    "profile.scientific_pins",
    "profile.history",
    "profile.paths",
    "profile.segment",
    "profile.invocation",
    "profile.invocation.arguments",
    "operator_directive",
    "technical_rebind",
    "decisions",
    "decisions.amendment",
    "decisions.historical_access",
    "decisions.segment_execution",
)


def test_pinned_header_returns_immutable_facts_not_authority():
    case = make_case()
    result = validate(case)
    assert result.profile_bytes == canonical_bytes(case.profile)
    assert result.profile_sha256 == digest(case.profile)
    assert result.envelope_sha256 == digest(case.envelope)
    assert result.policy_sha256 == digest(case.policy)
    assert result.history_index_sha256 == case.profile["history"]["index_sha256"]
    assert (result.segment_ordinal, result.start_ordinal, result.end_ordinal) == (
        2,
        73,
        125,
    )
    assert result.execution_revision == "b" * 40
    assert result.scope == "closed_series_adoption_metadata_only"
    assert result.authorizes_execution is False
    assert "Invented" not in repr(result) and "profile_bytes" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.start_ordinal = 1
    with pytest.raises(ValueError):
        replace(result, authorizes_execution=True)


@pytest.mark.parametrize("path", ("", *CLOSED))
@pytest.mark.parametrize("operation", ("missing", "extra"))
def test_every_closed_record_rejects_missing_or_extra_fields(path, operation):
    case = make_case()
    target = selected(case.envelope, path) if path else case.envelope
    if operation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unreviewed_extension"] = True
    with pytest.raises(
        api().SeriesAdoptionError, match="^invalid_series_adoption_header$"
    ):
        validate(case)


@pytest.mark.parametrize(
    "member", ("expected_profile_sha256", "expected_envelope_sha256")
)
@pytest.mark.parametrize("pin", ("0" * 64, "A" * 64, None, 1))
def test_independent_pins_authenticate_original_bytes(member, pin):
    with pytest.raises(api().SeriesAdoptionError):
        validate(make_case(), **{member: pin})


@pytest.mark.parametrize(
    "encoding", ("space", "duplicate", "nan", "wrong_type", "invalid")
)
def test_noncanonical_or_invalid_content_rejects_after_rehash(encoding):
    case = make_case()
    content = canonical_bytes(case.profile)
    content = {
        "space": b" " + content,
        "duplicate": content.replace(
            b'{"amendment_sha256":', b'{"schema_version":1,"amendment_sha256":', 1
        ),
        "nan": content.replace(b'"schema_version":1', b'"schema_version":NaN', 1),
        "wrong_type": b"[]\n",
        "invalid": b"\xff",
    }[encoding]
    with pytest.raises(api().SeriesAdoptionError):
        validate(
            case,
            profile_bytes=content,
            expected_profile_sha256=sha256(content).hexdigest(),
        )


def test_swapped_embedded_profile_rejects():
    case = make_case()
    case.envelope["profile"] = copy.deepcopy(case.profile)
    case.envelope["profile"]["series_id"] = "another-series"
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)
