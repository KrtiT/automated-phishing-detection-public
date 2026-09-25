import json

import pytest
from operational_cell_io_integration_fixtures import (
    candidates,
    case,
    disk_case,
    manifests,
)

from automated_phishing_detection._operational_attempt_io import held_attempt_writer
from automated_phishing_detection._operational_cell_protocol import (
    PRIVATE_NAMES,
    SNAPSHOT_NAMES,
)
from automated_phishing_detection.operational_cell_acceptance import (
    VerifiedOperationalCell,
)
from automated_phishing_detection.operational_cell_completion import (
    hold_operational_cell,
)

__all__ = ["candidates", "case", "manifests"]


@pytest.mark.parametrize("workload", ["http", "shift"])
def test_real_scientific_verifiers_and_shared_writer_publish_exact_snapshot(
    tmp_path, case, workload
):
    fixture = disk_case(tmp_path, case, workload)
    with hold_operational_cell(
        fixture.attempt, fixture.public, expected_identity=fixture.identity
    ) as completer:
        with held_attempt_writer(fixture.attempt, names=PRIVATE_NAMES) as writer:
            for name in PRIVATE_NAMES:
                writer.retain(name, fixture.payloads[name])
        snapshot = completer.complete(**fixture.arguments)
        assert type(snapshot) is VerifiedOperationalCell
        assert len(snapshot.payloads) == len(SNAPSHOT_NAMES) == 36
        assert completer.candidate is snapshot
    public = json.loads(fixture.public.read_bytes())
    assert public["status"] == "operational_evidence_published"
    assert set(public["private_sha256"]) == set(PRIVATE_NAMES)
    assert snapshot.summary == public["summary"]
    for name in PRIVATE_NAMES:
        assert snapshot.payload(f"attempt/{name}") == fixture.payloads[name]
        assert snapshot.payload(f"attempt/evidence/{name}") == fixture.payloads[name]
