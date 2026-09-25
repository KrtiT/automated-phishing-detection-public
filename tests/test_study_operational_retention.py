"""Real A/B saved verification; constructed wrapper is not actual child proof."""

import gc
import weakref
from dataclasses import replace

import pytest
from operational_cell_io_integration_fixtures import (
    candidates,
    case,
    disk_case,
    manifests,
)
from study_operational_fixtures import compact
from test_study_operational_records import api

from automated_phishing_detection._operational_attempt_io import held_attempt_writer
from automated_phishing_detection._operational_cell_protocol import PRIVATE_NAMES
from automated_phishing_detection.operational_cell_completion import (
    hold_operational_cell,
)
from automated_phishing_detection.operational_cell_runner import (
    ObservedOperationalCell,
    OperationalCellFailure,
)

__all__ = ["candidates", "case", "manifests"]


def observed_case(tmp_path, source, workload):
    fixture = disk_case(tmp_path, source, workload)
    with hold_operational_cell(
        fixture.attempt, fixture.public, expected_identity=fixture.identity
    ) as completer:
        with held_attempt_writer(fixture.attempt, names=PRIVATE_NAMES) as writer:
            for name in PRIVATE_NAMES:
                writer.retain(name, fixture.payloads[name])
        snapshot = completer.complete(**fixture.arguments)
    return ObservedOperationalCell(fixture.arguments["observation"], snapshot)


@pytest.mark.parametrize("workload", ["http", "shift"])
def test_pack_original_bytes_without_retaining_full_snapshot(tmp_path, case, workload):
    observed = observed_case(tmp_path, case, workload)
    source, observation = observed.snapshot.accepted, observed.observation
    snapshot_reference = weakref.ref(observed.snapshot)
    result = api().retain_accepted_cell(observed, accepted=source)
    assert result.observation is observation
    assert result.run_bytes is observed.snapshot.payload("attempt/run.json")
    assert result.descriptor_bytes is observed.snapshot.inputs.descriptor_bytes
    assert len(result.snapshot_sha256) == 36
    del observed
    gc.collect()
    assert snapshot_reference() is None


def test_returned_fallback_is_accepted_but_not_fabricated_compact(tmp_path, case):
    observed = observed_case(tmp_path, case, "http")
    prefix = tuple(compact(api(), ordinal) for ordinal in range(1, 21))
    slots = api().freeze_cell_accounting(prefix, returned=observed)
    assert slots[20].status == "accepted" and slots[20].returned is observed
    assert slots[20].accepted is None
    row = api().cell_accounting_projection(slots)[20]
    assert row["retention"] == "unpacked" and row["snapshot_sha256"] is None
    assert row["observation_sha256"] is not None and row["reservation_sha256"] is None
    stop = api().StoppedOperationalCell(observed.snapshot.inputs.cell, "completion")
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting(prefix, stopped=stop, returned=observed)


def test_actual_failed_candidate_stays_stopped_with_original_progress(tmp_path, case):
    observed = observed_case(tmp_path, case, "http")
    cell = observed.snapshot.inputs.cell
    failure = OperationalCellFailure(
        cell, None, observed.observation, None, observed.snapshot, "finalization", True
    )
    stop = api().StoppedOperationalCell(cell, failure.stage, failure, b"partial")
    slots = api().freeze_cell_accounting(
        tuple(compact(api(), ordinal) for ordinal in range(1, 21)), stopped=stop
    )
    row = api().cell_accounting_projection(slots)[20]
    assert row["status"] == "stopped" and row["publishing"] is True
    assert row["snapshot_sha256"] is None and slots[20].stopped.failure is failure


def test_source_or_observation_substitution_is_not_packed(tmp_path, case):
    observed = observed_case(tmp_path, case, "http")
    source = observed.snapshot.accepted
    with pytest.raises(api().StudyOperationalRecordError):
        api().retain_accepted_cell(observed, accepted=replace(source))
    different = replace(observed.observation, record=b"different")
    with pytest.raises(api().StudyOperationalRecordError):
        api().retain_accepted_cell(
            replace(observed, observation=different), accepted=source
        )
