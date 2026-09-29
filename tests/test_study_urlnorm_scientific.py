"""Genuine scientific composition on invented data, not process or access proof."""

import json

from study_urlnorm_scientific_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
    scientific_case,
    score_and_verify,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "scientific_case"]


def _assert_preparation(case):
    completion = json.loads(case.preparation.payload("preparation-complete.json"))
    assert completion["schema_version"] == 2
    assert not case.prior.external.retained
    assert len(case.preparation.external.retained) == 5


def _assert_models_unchanged(internal, external):
    internal_bindings = json.loads(internal.payload("attempt/evidence/bindings.json"))
    external_bindings = json.loads(external.payload("attempt/evidence/bindings.json"))
    for name in ("artifact_hashes", "thresholds", "secondary", "gmm_audit"):
        assert internal_bindings[name] == external_bindings[name]


def test_derived_urls_reach_original_models_and_saved_scientific_acceptance(
    scientific_case, monkeypatch
):
    case = scientific_case
    _assert_preparation(case)
    internal, external, supplied = score_and_verify(case, monkeypatch)
    assert len(internal.payloads) == 35
    assert len(external.payloads) == 76
    _assert_models_unchanged(internal, external)
    expected = [row.raw_url for row in case.preparation.external.retained]
    assert expected == list(case.expected_urls)
    assert case.session.evaluation.primary.scorer.urls == expected
    assert case.feature_urls == expected
    assert supplied and all(urls == tuple(expected) for urls in supplied)
    assert all(url.startswith("HTTPS://Host") and "/PaTh?" in url for url in expected)
    assert external.payload("attempt/evidence/publisher-source.json") == (
        case.preparation.payload("publisher-source.json")
    )
