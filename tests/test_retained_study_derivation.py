"""Derived preparation uses authenticated invented retained buffers only."""

import importlib
import importlib.util
import json
from hashlib import sha256

import pytest
from retained_study_derivation_fixtures import (
    derivation_case,
    inputs,
    preparation_api,
    preparation_case,
    run,
    runner,
)

from automated_phishing_detection import _study_preparation_body as preparation_body
from automated_phishing_detection import (
    phishvn_source,
    source_overlap,
    source_runner,
    study_preparation_inputs,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_preparation_retention import PREPARATION_ORDER

__all__ = ["derivation_case", "inputs", "preparation_api", "preparation_case", "runner"]


def test_retained_derivation_has_no_raw_acquisition_entrypoint():
    name = "automated_phishing_detection.retained_study_derivation"
    assert importlib.util.find_spec(name), (
        "missing retained-only preparation derivation"
    )
    module = importlib.import_module(name)
    assert callable(module.run_retained_study_preparation)
    assert not hasattr(module, "run_study_preparation")


def test_derivation_never_acquires_original_sources_or_reconstructs_internal(
    derivation_case, monkeypatch
):
    def forbidden(*arguments, **keywords):
        pytest.fail("retained derivation reached an original acquisition API")

    public_read = source_runner._read_file_once

    def metadata_only(path):
        assert path in {
            derivation_case.binding.root / "data/sources.json",
            derivation_case.binding.root / "reports/phiusiil-preparation-summary.json",
        }
        return public_read(path)

    monkeypatch.setattr(source_runner, "_read_file_once", metadata_only)
    for module, names in (
        (preparation_body, ("_read_once",)),
        (
            study_preparation_inputs,
            ("prepare_internal_inputs", "decode_external_inputs"),
        ),
        (phishvn_source, ("decode_phishvn_archive",)),
        (source_overlap, ("reconstruct_source_overlap",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    snapshot = run(derivation_case)
    assert tuple(name for name, unused in snapshot.payloads) == PREPARATION_ORDER
    assert snapshot.reservation_sha256 != derivation_case.prior.reservation_sha256


def test_derivation_copies_internal_bytes_and_rewraps_only_execution(derivation_case):
    snapshot = run(derivation_case)
    for name in ("suffix-rules.dat", "group_test.jsonl", "source-overlap.json"):
        assert snapshot.payload(name) == derivation_case.prior.payload(name)
    reconstruction = json.loads(snapshot.payload("source-reconstruction.json"))
    prior = json.loads(derivation_case.prior.payload("source-reconstruction.json"))
    assert reconstruction["reconstruction"] == prior["reconstruction"]
    assert reconstruction["checkpoint_sha256"] == prior["checkpoint_sha256"]
    assert reconstruction["execution"]["revision"] == derivation_case.binding.revision
    assert reconstruction["reservation_sha256"] == snapshot.reservation_sha256


def test_derived_completion_binds_exact_lineage_and_publisher_parents(derivation_case):
    snapshot = run(derivation_case)
    complete = json.loads(snapshot.payload("preparation-complete.json"))
    assert complete["schema_version"] == 2
    assert complete["protocol"] == "study-preparation-derived-v1"
    assert complete["derivation"] == {
        key: value
        for key, value in derivation_case.continuation.items()
        if key != "prior_profile"
    }
    assert complete["execution"]["protocol"] == "study-preparation-v1"
    for name, content in snapshot.payloads[:-1]:
        assert complete["input_sha256"][name] == sha256(content).hexdigest()
    for name in ("publisher-source.json", "publisher-summary.json"):
        record = json.loads(snapshot.payload(name))
        assert record["schema_version"] == 2
        assert (
            record["parent_source_sha256"]
            == derivation_case.continuation["publisher_source_sha256"]
        )
        assert (
            record["parent_summary_sha256"]
            == derivation_case.continuation["publisher_summary_sha256"]
        )
    assert snapshot.payload("preparation-complete.json") == canonical_bytes(complete)


@pytest.mark.parametrize(
    "pin",
    (
        "prior_preparation_reservation_sha256",
        "prior_preparation_complete_sha256",
        "publisher_source_sha256",
        "publisher_summary_sha256",
    ),
)
def test_wrong_prior_preparation_pin_rejects_without_reserving(derivation_case, pin):
    with pytest.raises(ValueError):
        run(
            derivation_case, continuation=derivation_case.continuation | {pin: "0" * 64}
        )
    assert not derivation_case.paths.attempt.exists()


def test_occupied_derived_output_is_preserved_and_not_reused(derivation_case):
    derivation_case.paths.attempt.mkdir()
    marker = derivation_case.paths.attempt / "existing.txt"
    marker.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError):
        run(derivation_case)
    assert marker.read_bytes() == b"do not overwrite"


def test_failed_derivation_preserves_partial_receipt_without_retry(
    derivation_case, monkeypatch
):
    calls = []

    def fail(*arguments, **keywords):
        calls.append(True)
        raise ValueError("invented failure")

    monkeypatch.setattr(study_preparation_inputs, "prepare_external_inputs", fail)
    with pytest.raises(ValueError) as caught:
        run(derivation_case)
    assert calls == [True]
    assert type(caught.value.preparation_progress) is bytes
    attempt = derivation_case.paths.attempt
    assert (attempt / "publisher-source.json").is_file()
    assert not (attempt / "preparation-complete.json").exists()
    assert (attempt / "outcome.json").is_file()
