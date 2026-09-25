"""Real private input transport and reservation, using invented pure snapshots."""

from hashlib import sha256
from types import SimpleNamespace

from operational_transport_integration_fixtures import retain_inputs

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_cell_inputs as cells
from automated_phishing_detection.operational_schedule import cell_for_ordinal


def child_case(tmp_path, accepted, source, monkeypatch, ordinal=121):
    attempt = receipt.reserve_attempt(
        tmp_path / "attempt", identity={"kind": "invented_cell"}
    )
    descriptor = cells.build_cell_descriptor(accepted, cell_for_ordinal(ordinal))
    binding = cells.bind_cell_descriptor(
        descriptor.descriptor_bytes, cell_reservation_sha256=attempt.reservation_sha256
    )
    contents = {
        "accepted-inputs.json": accepted.metadata_bytes,
        "descriptor.json": descriptor.descriptor_bytes,
        "binding.json": binding,
        "manifest": descriptor.manifest_bytes,
    }
    with retain_inputs(tmp_path, contents) as paths:
        pass
    environment(monkeypatch, attempt)
    return SimpleNamespace(
        binding=source.binding,
        profile=SimpleNamespace(profile_sha256=source.profile),
        attempt=attempt,
        contents=contents,
        paths=paths,
        digest=sha256(binding).hexdigest(),
        requests=descriptor.requests,
    )


def environment(monkeypatch, attempt):
    import os

    for name in tuple(os.environ):
        if name.startswith("APD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("APD_BASE_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("APD_ATTEMPT_DIRECTORY", str(attempt.directory))
    monkeypatch.setenv("APD_RESERVATION_SHA256", attempt.reservation_sha256)


def keywords(case):
    return {
        "accepted_inputs_directory": case.paths[0],
        "cell_input_directory": case.paths[1],
        "expected_binding_sha256": case.digest,
    }
