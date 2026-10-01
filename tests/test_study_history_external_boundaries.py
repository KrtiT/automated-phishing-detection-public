"""No IO, live capability, new forwards or mutable results from saved science."""

import builtins
import json
import subprocess
from pathlib import Path

import pytest
from study_history_external_fixtures import api, make_case, verify

from automated_phishing_detection import (
    character_transformer,
    execution_preflight,
    external_source_completion,
    owned_worker,
)
from automated_phishing_detection._external_preparation_outputs import _PUBLIC_INPUTS


@pytest.fixture
def history(tmp_path, monkeypatch):
    return make_case(tmp_path, monkeypatch)


def forbid_live(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved science attempted IO, live authority or new forwards")

    for target, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (Path, "read_text"),
        (subprocess, "Popen"),
        (execution_preflight.ExecutionBinding, "__init__"),
        (owned_worker.WorkerObservation, "__init__"),
        (external_source_completion, "verify_external_completion_snapshot"),
        (character_transformer.CharacterTransformer, "forward"),
    ):
        monkeypatch.setattr(target, name, forbidden)


def test_no_source_model_file_or_live_owner_access(history, monkeypatch):
    api()
    forbid_live(monkeypatch)
    assert len(verify(history).rows) == 5


def test_result_has_independent_fresh_views(history):
    result = verify(history)
    public = result.public_summary
    original = json.loads(history.payloads["public-summary.json"])
    history.payloads.clear()
    public.clear()
    replay = result.replay
    replay.evidence.populations.clear()
    assert result.public_summary == original
    assert len(result.payloads) == 76
    assert result.replay.evidence.populations


@pytest.mark.parametrize("name", [name for name, unused in _PUBLIC_INPUTS])
def test_each_original_public_source_pin_is_required(history, name):
    pins = history.arguments["expected_source_sha256"].copy()
    pins[name] = "0" * 64
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history, expected_source_sha256=pins)


@pytest.mark.parametrize("change", ["missing", "extra", "boolean"])
def test_source_pin_inventory_is_closed(history, change):
    pins = history.arguments["expected_source_sha256"].copy()
    if change == "missing":
        pins.pop("data/sources.json")
    elif change == "extra":
        pins["other"] = "0" * 64
    else:
        pins["data/sources.json"] = True
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history, expected_source_sha256=pins)


@pytest.mark.parametrize(
    "directory", ["relative", "/unit/../attempt", "/unit//attempt", "/wrong"]
)
def test_original_attempt_locator_cannot_be_substituted(history, directory):
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history, expected_attempt_directory=directory)


@pytest.mark.parametrize("payloads", [None, [], (), {"private": bytearray(b"secret")}])
def test_malformed_payloads_produce_only_symbolic_diagnostics(payloads):
    with pytest.raises(
        ValueError, match="^invalid_historical_external_science$"
    ) as caught:
        api().verify_historical_external_science(
            payloads,
            expected_snapshot_sha256={},
            expected_execution={},
            expected_source_sha256={},
            expected_profile_bytes=b"",
            expected_handoff_bytes=b"",
            expected_overlap_bytes=b"",
            expected_attempt_directory="/unit",
        )
    assert caught.value.__suppress_context__ is True
