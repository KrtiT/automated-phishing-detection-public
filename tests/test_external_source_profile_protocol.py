"""Candidate declarations close retention inventory and numerical access."""

import inspect
from dataclasses import FrozenInstanceError

import pytest
import test_external_source_profile as profile_fixtures
from test_external_source_profile import api, resolve

from automated_phishing_detection import _external_checkpoint_protocol as protocol
from automated_phishing_detection import phishvn, phishvn_source
from automated_phishing_detection._phishvn_archive import EXPECTED_FORMAT

profile_case = profile_fixtures.profile_case


def test_profile_is_immutable_and_nested_projections_are_fresh(profile_case):
    profile = resolve(profile_case)
    original = profile.canonical_bytes
    first = profile.projection()
    first["publisher"]["expected_format"]["archive_sha256"] = "0" * 64
    first["retention"]["scientific_order"].clear()
    assert profile.canonical_bytes == original
    assert profile.projection()["publisher"]["expected_format"] == EXPECTED_FORMAT
    assert len(profile.projection()["retention"]["scientific_order"]) == 30
    with pytest.raises(FrozenInstanceError):
        profile.canonical_bytes = b"changed"


@pytest.mark.parametrize("name", ["PROVENANCE_ORDER", "SCIENTIFIC_ORDER"])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "wrong_name"])
def test_checkpoint_orders_must_match_exact_existing_inventories(
    profile_case, monkeypatch, name, change
):
    order = getattr(protocol, name)
    if change == "missing":
        altered = order[:-1]
    elif change == "extra":
        altered = (*order, "unexpected")
    elif change == "duplicate":
        altered = (*order[:-1], order[0])
    else:
        altered = (*order[:-1], "unexpected")
    monkeypatch.setattr(protocol, name, altered)
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case)


def test_resolver_exposes_no_path_profile_pin_or_readiness_overrides(profile_case):
    signature = inspect.signature(api().resolve_external_source_profile)
    assert tuple(signature.parameters) == ("binding",)
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )
    for name in (
        "archive_pins",
        "profile_path",
        "profile",
        "protected_evaluation_ready",
    ):
        with pytest.raises(TypeError):
            api().resolve_external_source_profile(profile_case.binding, **{name: True})
    assert profile_case.events == []


def test_resolution_never_decodes_sources_prepares_rows_or_starts_scoring(
    profile_case, monkeypatch
):
    from automated_phishing_detection import bound_external_runtime, external_producer

    def forbidden(*args, **kwargs):
        pytest.fail("candidate resolution accessed source records or numerical work")

    for owner, name in (
        (phishvn_source, "decode_phishvn_archive"),
        (phishvn_source, "_decode_rows"),
        (phishvn, "prepare_external_rows"),
        (bound_external_runtime, "open_bound_external_session"),
        (external_producer, "produce_external_evidence"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    profile = resolve(profile_case)
    assert profile.protected_evaluation_ready is False
    assert profile_case.events == ["recheck", "public_source_read", "recheck"]
