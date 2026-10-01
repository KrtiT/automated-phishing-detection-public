"""Complete opaque history verification never authorizes a later execution."""

import pytest
from operational_input_fixtures import candidates, manifests
from study_history_snapshot_fixtures import api, arguments, complete_history, verify
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


def test_complete_opaque_history_is_separate_from_live_authority(prepared, manifests):
    module = api()
    case = complete_history(prepared, manifests)
    result = verify(case)
    assert type(result) is module.StudyHistorySnapshots
    assert result.authorization.accepted_ordinals == (1,)
    assert result.scope == "accepted_sources_and_complete_prefix_opaque_bytes_only"


@pytest.mark.parametrize("role", ["internal", "external"])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "changed"])
def test_all_source_snapshot_bytes_are_bound(prepared, manifests, role, change):
    case = complete_history(prepared, manifests)
    payloads = tuple(getattr(case, role).items())
    altered = {
        "missing": payloads[:-1],
        "extra": (*payloads, ("unexpected", b"opaque")),
        "duplicate": (*payloads, payloads[0]),
        "changed": ((payloads[0][0], b"changed"), *payloads[1:]),
    }[change]
    with pytest.raises(ValueError):
        verify(case, **{f"{role}_payloads": altered})


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "changed"])
def test_all_cell_snapshot_bytes_are_bound(prepared, manifests, change):
    case = complete_history(prepared, manifests)
    ordinal, reservation, payloads = case.cells[0]
    altered = {
        "missing": payloads[:-1],
        "extra": (*payloads, ("unexpected", b"x")),
        "duplicate": (*payloads, payloads[0]),
        "changed": ((payloads[0][0], b"changed"), *payloads[1:]),
    }[change]
    with pytest.raises(ValueError):
        verify(case, cell_snapshots=((ordinal, reservation, altered),))


def test_original_root_is_reauthenticated(prepared, manifests):
    from stopped_study_authorization_fixtures import verify as stage_one

    case = complete_history(prepared, manifests)
    certificate = stage_one(case)
    with pytest.raises(ValueError):
        api().verify_study_history_snapshots(certificate, **arguments(case))


@pytest.mark.parametrize("role", ["internal", "external"])
def test_independent_source_pin_is_required(prepared, manifests, role):
    case = complete_history(prepared, manifests)
    with pytest.raises(ValueError):
        verify(case, **{f"expected_{role}_sha256": {}})


def test_complete_seventy_two_cell_prefix_stays_explicit(prepared, manifests):
    case = complete_history(prepared, manifests, prefix=72)
    result = verify(case)
    assert result.authorization.accepted_ordinals == tuple(range(1, 73))
    assert result.authorization.stopped_ordinal == 73
    assert tuple(ordinal for ordinal, hashes in result.cells_sha256) == tuple(
        range(1, 73)
    )
