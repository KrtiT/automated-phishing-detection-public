"""A factual candidate header never crosses existing execution boundaries."""

from pathlib import Path

import pytest
from study_series_adoption_fixtures import api, digest, make_case, refresh, validate

from automated_phishing_detection import _study_execution_schema as legacy
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def test_no_io_or_runtime_binding_is_attempted(monkeypatch):
    from automated_phishing_detection import execution_preflight, study_execution
    from automated_phishing_detection import stopped_study_authorization as history

    case = make_case()

    def forbidden(*args, **kwargs):
        raise AssertionError("header attempted a forbidden access")

    for name in ("_read_regular", "_git", "bind_execution", "recheck_binding"):
        monkeypatch.setattr(execution_preflight, name, forbidden)
    monkeypatch.setattr(study_execution, "bind_study_execution", forbidden)
    monkeypatch.setattr(history, "verify_stopped_study_authorization", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert validate(case).authorizes_execution is False


def test_new_envelope_and_result_cannot_enter_legacy_schema():
    case = make_case()
    result = validate(case)
    with pytest.raises(legacy.StudyExecutionError):
        legacy.profile(result)
    with pytest.raises(legacy.StudyExecutionError):
        legacy.envelope(canonical_bytes(case.envelope), digest(case.envelope))
    assert not hasattr(api(), "bind_study_series_execution")


@pytest.mark.parametrize(
    "name", ("profile_sha256", "envelope_sha256", "authorizes_execution")
)
def test_profile_has_no_self_pin_or_authority_extension(name):
    case = make_case()
    case.profile[name] = "0" * 64
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


def test_fixed_policy_file_matches_final_closed_projection():
    from automated_phishing_detection import _study_series_policy as policy

    content = (Path(__file__).parents[1] / policy.POLICY_PATH).read_bytes()
    assert content == policy.policy_bytes()
    value = policy.policy_projection()
    assert value["status"] == "specified_for_explicit_adoption"
    assert value["legacy_readiness"] is False
    assert value["permitted_added_paths"] == sorted(set(value["permitted_added_paths"]))
    assert len(value["permitted_added_paths"]) == 98
    assert set(value["scripts"].values()) <= set(value["permitted_added_paths"])
    assert "no automatic retry" in value["amendment"]["text"]
