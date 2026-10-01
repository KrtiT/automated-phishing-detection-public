"""Independent public-only binding checks with explicitly stubbed base preflight."""

from dataclasses import replace
from importlib import import_module

import pytest
from study_series_execution_fixtures import api, bind, public_case

from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._study_execution_policy import CONTRACT_SHA256

__all__ = ["public_case"]


def forbidden(*arguments, **keywords):
    pytest.fail("public binding crossed an unauthenticated or historical boundary")


@pytest.mark.parametrize("change", ("revoked", "development"))
def test_complete_header_precedes_every_public_binding_stage(
    public_case, monkeypatch, change
):
    if change == "revoked":
        public_case.envelope["revoked"] = True
    else:
        public_case.envelope["operator_directive"]["scope"] = "development_only"
    for owner, name in (
        (preflight, "bind_execution"),
        (preflight, "recheck_binding"),
        (api().admission_io, "committed_policy"),
        (api(), "_components"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)


def test_original_preflight_receives_exact_root_revision_and_v3_contract(
    public_case, monkeypatch
):
    calls = []

    def bound(root, **keywords):
        calls.append((root, keywords))
        return public_case.base

    monkeypatch.setattr(preflight, "bind_execution", bound)
    result = bind(public_case)
    assert calls == [
        (
            public_case.base.root,
            {
                "expected_revision": public_case.base.revision,
                "expected_contract_sha256": CONTRACT_SHA256,
            },
        )
    ]
    assert result.authorizes_execution is False


def test_wrong_committed_policy_stops_before_component_resolution(
    public_case, monkeypatch
):
    monkeypatch.setattr(
        api().admission_io, "committed_policy", lambda *arguments: b"other policy"
    )
    monkeypatch.setattr(api(), "_components", forbidden)
    monkeypatch.setattr(preflight, "recheck_binding", forbidden)
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)


@pytest.mark.parametrize("field", ("components", "external", "operational"))
def test_recheck_compares_retained_comparison_and_candidates(public_case, field):
    original = bind(public_case)
    with pytest.raises(api().SeriesPublicBindingError):
        api().recheck_series_public_execution(replace(original, **{field: object()}))
    assert public_case.events == ["preflight", "recheck", "preflight", "recheck"]


def test_public_recheck_does_not_reconstruct_history_or_load_models(
    public_case, monkeypatch
):
    for module_name, name in (
        ("study_history_internal", "verify_historical_internal_science"),
        ("study_history_external", "verify_historical_external_science"),
        ("study_history_cell", "verify_historical_cell_science"),
        ("study_series_cell", "verify_series_cell_science"),
        ("bound_runtime", "open_bound_session"),
        ("bound_models", "load_bound_models"),
    ):
        module = import_module(f"automated_phishing_detection.{module_name}")
        monkeypatch.setattr(module, name, forbidden)
    result = bind(public_case)
    api().recheck_series_public_execution(result)
    assert result.authorizes_execution is False
