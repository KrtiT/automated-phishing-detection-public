"""Saved outer hashes cannot replace source or scientific reconstruction."""

import json
from dataclasses import replace

import pytest
from external_completion_fixtures import external_completion_case
from external_completion_mutation_fixtures import replace_private, rewrite
from test_external_source_completion import bind_fixture, module, verify

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_preparation_outputs import _PUBLIC_INPUTS


def forbidden(*args, **kwargs):
    pytest.fail("invalid retained predecessor reached later scientific work")


@pytest.mark.parametrize("relative,retained", _PUBLIC_INPUTS)
def test_all_saved_drift_public_inputs_match_independent_binding_before_preparation(
    tmp_path, monkeypatch, relative, retained
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    changed = json.loads(case.filesinputs[retained])
    changed["invented_canary"] = relative
    replace_private(case, retained, canonical_bytes(changed))
    monkeypatch.setattr(api, "verify_external_provenance", forbidden)
    monkeypatch.setattr(api, "reconstruct_external_evidence", forbidden)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)


@pytest.mark.parametrize(
    "name", ["publisher-source.json", "suffix-rules.dat", "retained-test.jsonl"]
)
def test_relinked_provenance_cannot_replace_exact_reconstruction(
    tmp_path, monkeypatch, name
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    replace_private(case, name, case.filesinputs[name] + b"\n", link_provenance=True)
    monkeypatch.setattr(api, "reconstruct_external_evidence", forbidden)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)


@pytest.mark.parametrize(
    "name", ["all-scores.jsonl", "routing.json", "secondary.json", "monitors.json"]
)
def test_coherently_rehashed_science_still_requires_reconstruction(
    tmp_path, monkeypatch, name
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    replace_private(case, name, b"invalid-science-canary\n")
    with pytest.raises(api.ExternalCompletionVerificationError) as rejected:
        verify(api, case)
    assert str(rejected.value) == "invalid_external_completion"


def test_coherently_rehashed_composition_aggregate_cannot_claim_extra_rows(
    tmp_path, monkeypatch
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    public = json.loads(case.paths.public_summary.read_bytes())
    public["composition"]["row_count"] += 1
    rewrite(case, case.filesinputs, public)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)


@pytest.mark.parametrize("name", ["reservation.json", "finalize.claim", "outcome.json"])
def test_bad_outer_records_stop_before_any_scientific_reconstruction(
    tmp_path, monkeypatch, name
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    (case.paths.attempt / name).write_bytes(b"private-canary")
    monkeypatch.setattr(api, "verify_external_provenance", forbidden)
    monkeypatch.setattr(api, "reconstruct_external_evidence", forbidden)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)


@pytest.mark.parametrize("name", ["attempt", "public_summary"])
def test_output_checkout_rejection_precedes_saved_file_reads(
    tmp_path, monkeypatch, name
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    case.paths = replace(case.paths, **{name: case.binding.root / "forbidden"})
    monkeypatch.setattr(api, "snapshot_external_files", forbidden)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)
