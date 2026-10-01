"""Actual new scientific validators over invented HTTP/shift bytes on temporary disk."""

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from operational_cell_process_fixtures import process_records
from study_series_acceptance_fixtures import arguments
from study_series_cell_fixtures import candidates, make_cell, manifests, series_case

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_attempt_io import held_attempt_writer
from automated_phishing_detection._operational_cell_protocol import (
    PRIVATE_NAMES,
    SNAPSHOT_NAMES,
)
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor
from automated_phishing_detection.study_series_cell import SeriesCellScience
from automated_phishing_detection.study_series_cell_completion import (
    OperationalCellCompletionError,
    hold_series_cell,
)
from automated_phishing_detection.study_series_inputs import restore_series_cell_inputs

__all__ = ["candidates", "manifests", "series_case"]


def _rebind(original, reservation):
    inputs, expected = original.inputs.computational, original.arguments
    binding = bind_cell_descriptor(
        inputs.descriptor_bytes, cell_reservation_sha256=reservation
    )
    return restore_series_cell_inputs(
        inputs.accepted_bytes,
        expected["profile_bytes"],
        expected["internal_snapshot"],
        expected["external_snapshot"],
        inputs.descriptor_bytes,
        binding,
        inputs.manifest_bytes,
        expected_metadata_sha256=expected["expected_metadata_sha256"],
        expected_profile_sha256=expected["expected_profile_sha256"],
        expected_binding_sha256=sha256(binding).hexdigest(),
        expected_cell_reservation_sha256=reservation,
    )


def _disk_case(tmp_path, source, ordinal):
    original = make_cell(source, ordinal)
    expected = arguments(original)
    identity = expected.pop("expected_identity")
    expected.pop("attempt")
    attempt = receipt.reserve_attempt(tmp_path.resolve() / "cell", identity=identity)
    inputs = _rebind(original, attempt.reservation_sha256)
    commands = expected["service_command"], expected["client_command"]
    payloads, observation = process_records(
        inputs.computational,
        attempt.reservation_sha256,
        commands,
        expected["expected_deadlines"],
    )
    payloads.update(
        {
            name: original.values[f"attempt/{name}"]
            for name in ("run.json", "warmup.json", "measured.json")
        }
    )
    return SimpleNamespace(
        attempt=attempt,
        identity=identity,
        payloads=payloads,
        public=tmp_path.resolve() / "public.json",
        original=original,
        options=expected | {"inputs": inputs, "observation": observation},
    )


def _write(case):
    with held_attempt_writer(case.attempt, names=PRIVATE_NAMES) as writer:
        for name in PRIVATE_NAMES:
            writer.retain(name, case.payloads[name])


@pytest.mark.parametrize("ordinal", [21, 121])
def test_real_series_science_and_held_publication_preserve_full_http_and_shift(
    tmp_path, series_case, ordinal
):
    case = _disk_case(tmp_path, series_case, ordinal)
    with hold_series_cell(
        case.attempt, case.public, expected_identity=case.identity
    ) as completer:
        _write(case)
        result = completer.complete(**case.options)
        assert type(result) is SeriesCellScience
        assert len(result.payloads) == len(SNAPSHOT_NAMES) == 36
        assert completer.candidate is result
    public = json.loads(case.public.read_bytes())
    assert public["status"] == "operational_evidence_published"
    assert public["summary"] == result.summary == case.original.summary
    assert result.run == case.original.run
    assert (
        result.inputs.origin_metadata_bytes
        == case.original.inputs.origin_metadata_bytes
    )
    assert (
        result.inputs.computational.accepted_bytes
        == case.original.inputs.computational.accepted_bytes
    )
    assert result.authorizes_execution is False
    for name in PRIVATE_NAMES:
        assert dict(result.payloads)[f"attempt/{name}"] == case.payloads[name]
        assert dict(result.payloads)[f"attempt/evidence/{name}"] == case.payloads[name]


def test_real_scientific_rejection_never_enters_publication(tmp_path, series_case):
    case = _disk_case(tmp_path, series_case, 21)
    case.payloads["run.json"] = b"{}"
    with pytest.raises(OperationalCellCompletionError):
        with hold_series_cell(
            case.attempt, case.public, expected_identity=case.identity
        ) as completer:
            _write(case)
            completer.complete(**case.options)
    assert completer.working is completer.candidate is None
    assert not completer.publishing and not case.public.exists()
    assert not (case.attempt.directory / "finalize.claim").exists()
    with pytest.raises(OperationalCellCompletionError):
        completer.complete(**case.options)
