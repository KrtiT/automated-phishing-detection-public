"""Historical reconstruction retains the unchanged complete external science."""

import json
from dataclasses import asdict

import pytest
from study_history_external_fixtures import (
    api,
    make_case,
    mutate_output,
    repin,
    rewrite_public,
    verify,
)


@pytest.fixture
def history(tmp_path, monkeypatch):
    return make_case(tmp_path, monkeypatch)


def test_restores_complete_science_without_new_live_ownership(history):
    result = verify(history)
    assert asdict(result.replay) == asdict(history.case.produced.replay)
    assert result.public_summary == history.case.expectedpublic
    assert dict(result.payloads) == history.payloads
    assert result.profile_bytes == history.case.profile.canonical_bytes
    assert not hasattr(result, "worker")
    assert not hasattr(result, "protected_evaluation_ready")


@pytest.mark.parametrize("change", ["missing", "extra", "altered", "mutable"])
def test_exact_independent_byte_inventory_before_any_replay(
    history, monkeypatch, change
):
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid inventory reached science")

    monkeypatch.setattr(module, "reconstruct_external_evidence", forbidden)
    name = "attempt/evidence/secondary.json"
    if change == "missing":
        history.payloads.pop(name)
    elif change == "extra":
        history.payloads["extra"] = b"secret"
    elif change == "altered":
        history.payloads[name] += b" "
    else:
        history.payloads[name] = bytearray(history.payloads[name])
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history)


@pytest.mark.parametrize(
    "field", ["execution", "publisher", "source_profile", "composition"]
)
def test_rehashed_public_substitution_rejects(history, field):
    public = json.loads(history.payloads["public-summary.json"])
    public[field]["invented_extension"] = 1
    rewrite_public(history, public)
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history)


def test_rehashed_private_secondary_science_rejects(history):
    mutate_output(history, "secondary.json", lambda value: value.update(invented=1))
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history)


def test_exact_inventory_rejects_duplicate_original_copy(history):
    history.payloads["attempt/checkpoints/secondary.json"] += b" "
    repin(history)
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history)


@pytest.mark.parametrize(
    "field",
    ["expected_profile_bytes", "expected_handoff_bytes", "expected_overlap_bytes"],
)
def test_independent_original_expectations_cannot_be_replaced(history, field):
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history, **{field: b"{}\n"})
