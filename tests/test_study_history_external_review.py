"""Independent specification checks use invented bytes and original expectations."""

import json

import external_completion_fixtures
import pytest
from study_history_external_fixtures import (
    api,
    digest,
    make_case,
    repin,
    replace_output,
    verify,
)

from automated_phishing_detection import _external_source_profile as profiles
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_preparation_outputs import _PUBLIC_INPUTS
from automated_phishing_detection._external_provenance_payloads import (
    PREPARED_NAMES,
    PROVENANCE_NAMES,
)


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setattr(
        external_completion_fixtures, "observe_worker", lambda *unused: None
    )
    return make_case(tmp_path, monkeypatch)


def test_original_candidate_is_used_and_each_kernel_runs_once(history, monkeypatch):
    module = api()
    calls = []
    provenance, science = (
        module.verify_external_provenance,
        module.reconstruct_external_evidence,
    )

    def verify_provenance(*args, **kwargs):
        calls.append("provenance")
        return provenance(*args, **kwargs)

    def verify_science(*args, **kwargs):
        calls.append("science")
        return science(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("historical candidate must not be freshly resolved")

    monkeypatch.setattr(profiles, "resolve_external_source_profile", forbidden)
    monkeypatch.setattr(module, "verify_external_provenance", verify_provenance)
    monkeypatch.setattr(module, "reconstruct_external_evidence", verify_science)
    result = verify(history)
    assert result.profile_bytes == history.arguments["expected_profile_bytes"]
    assert calls == ["provenance", "science"]


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "runtime_sha256",
        "source_profile_sha256",
        "archive_sha256",
        "suffix_rules_sha256",
        "internal_handoff_sha256",
        "internal_overlap_sha256",
        "internal_reservation_sha256",
        "reservation_sha256",
    ],
)
def test_original_execution_expectations_are_not_derived_from_public(history, field):
    execution = history.arguments["expected_execution"].copy()
    execution[field] = "0" * (40 if field == "revision" else 64)
    with pytest.raises(ValueError, match="^invalid_historical_external_science$"):
        verify(history, expected_execution=execution)


@pytest.mark.parametrize("name", [name for name, unused in _PUBLIC_INPUTS])
def test_each_original_public_source_pin_is_mandatory(history, name):
    expected = history.arguments["expected_source_sha256"].copy()
    expected[name] = "0" * 64
    with pytest.raises(ValueError):
        verify(history, expected_source_sha256=expected)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_public_source_inventory_is_exact(history, change):
    expected = history.arguments["expected_source_sha256"].copy()
    if change == "missing":
        expected.pop(next(iter(expected)))
    else:
        expected["unexpected"] = "0" * 64
    with pytest.raises(ValueError):
        verify(history, expected_source_sha256=expected)


def _repin_provenance(history):
    name = "external-source-reconstruction.json"
    value = json.loads(history.payloads[f"attempt/evidence/{name}"])
    for field, names in (
        ("provenance_sha256", PROVENANCE_NAMES - {name}),
        ("prepared_sha256", PREPARED_NAMES),
    ):
        value[field] = {
            retained: digest(history.payloads[f"attempt/evidence/{retained}"])
            for retained in names
        }
    replace_output(history, name, canonical_bytes(value))


@pytest.mark.parametrize(
    "name",
    [
        "publisher-source.json",
        "suffix-rules.dat",
        "internal-source-handoff.json",
        "internal-source-overlap.json",
        "retained-test.jsonl",
        "quarantine.jsonl",
        "inventory.json",
        "preparation-summary.json",
    ],
)
def test_rehashed_provenance_retains_original_inputs_and_preparation(history, name):
    replace_output(history, name, b"{}\n")
    _repin_provenance(history)
    with pytest.raises(ValueError):
        verify(history)


@pytest.mark.parametrize(
    "name",
    [
        "primary-completion.json",
        "all-scores.jsonl",
        "routing.json",
        "monitors.json",
        "predictions.jsonl",
        "secondary.json",
    ],
)
def test_rehashed_science_reaches_and_fails_unchanged_kernel(
    history, monkeypatch, name
):
    replace_output(history, name, b"{}\n")
    module, calls = api(), []
    original = module.reconstruct_external_evidence

    def reconstruct(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "reconstruct_external_evidence", reconstruct)
    with pytest.raises(ValueError):
        verify(history)
    assert calls == [True]


@pytest.mark.parametrize("name", ["attempt/finalize.claim", "attempt/outcome.json"])
def test_completion_receipts_keep_exact_original_codec(history, name):
    history.payloads[name] += b"\n"
    repin(history)
    with pytest.raises(ValueError):
        verify(history)


@pytest.mark.parametrize(
    "directory", ["relative/attempt", "/wrong/attempt", "/path/../attempt"]
)
def test_original_attempt_directory_is_required(history, directory):
    with pytest.raises(ValueError):
        verify(history, expected_attempt_directory=directory)
