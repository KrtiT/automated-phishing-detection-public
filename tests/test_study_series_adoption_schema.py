"""Recursive closed-schema, canonical pin and non-authority boundary coverage."""

import copy

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


def closed_paths():
    case = make_case()
    from automated_phishing_detection._study_series_adoption_schema import FIELDS

    names = {
        "profile": "profile",
        "profile.execution": "execution",
        "profile.origin": "origin",
        "profile.components": "components",
        "profile.transition": "transition",
        "profile.scientific_pins": "scientific_pins",
        "profile.history": "history",
        "profile.paths": "paths",
        "profile.segment": "segment",
        "profile.invocation": "invocation",
        "profile.invocation.arguments": "arguments",
        "operator_directive": "directive",
        "technical_rebind": "rebind",
        "decisions": "decisions",
        "decisions.amendment": "decision",
        "decisions.historical_access": "decision",
        "decisions.segment_execution": "decision",
    }
    assert all(
        set(selected(case.envelope, path)) == set(FIELDS[kind].split())
        for path, kind in names.items()
    )
    return tuple(
        (path, name) for path in names for name in selected(case.envelope, path)
    )


@pytest.mark.parametrize(("path", "name"), closed_paths())
def test_each_required_field_is_required(path, name):
    case = make_case()
    selected(case.envelope, path).pop(name)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize(
    ("path", "name"),
    (
        ("profile.execution", "runtime_sha256"),
        ("profile.origin", "preparation_complete_sha256"),
        ("profile.scientific_pins", "original_h2_audit_sha256"),
        ("profile.history", "exposure_record_sha256"),
        ("profile.segment", "predecessor_accounting_sha256"),
        ("operator_directive", "development_directive_sha256"),
        ("technical_rebind", "implementation_review_sha256"),
        ("technical_rebind", "verification_evidence_sha256"),
        ("profile.components", "current_external"),
        ("profile.components", "current_operational"),
    ),
)
@pytest.mark.parametrize("invalid", ("A" * 64, "a" * 63, None, True, {}, []))
def test_all_pin_families_require_canonical_digests(path, name, invalid):
    case = make_case()
    selected(case.envelope, path)[name] = invalid
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize("kind", ("external", "operational"))
def test_revision_sensitive_current_component_is_not_original_component(kind):
    case = make_case()
    components = case.profile["components"]
    components[f"current_{kind}"] = components[f"original_{kind}"]
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize("start", (2, 125))
def test_header_reports_declared_start_without_history_eligibility(start):
    case = make_case()
    case.profile["segment"]["start_ordinal"] = start
    refresh(case)
    result = validate(case)
    assert result.start_ordinal == start and result.authorizes_execution is False


@pytest.mark.parametrize("member", ("policy_bytes", "profile_bytes", "envelope_bytes"))
def test_only_original_bytes_inputs_not_constructed_results_are_accepted(member):
    case = make_case()
    result = validate(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case, **{member: result})


def test_origin_scope_mutation_cannot_hide_behind_refreshing_profile_pin():
    case = make_case()
    prior = copy.deepcopy(case.profile["origin"]["profile"])
    name = next(iter(prior["source_artifact_scope"]))
    prior["source_artifact_scope"][name] = "0" * 64
    case.profile["origin"].update(profile=prior, profile_sha256=digest(prior))
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


def test_rejection_does_not_echo_private_record_text():
    case = make_case()
    case.envelope["operator_directive"]["decision_text"] = "private marker\0"
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError) as captured:
        validate(case)
    assert str(captured.value) == "invalid_series_adoption_header"
    assert captured.value.__suppress_context__ is True
    assert "private marker" in canonical_bytes(case.envelope).decode()
