"""Only invented source manifests and root joins feed stopped-cell tests."""

import copy
import json
from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

import pytest
from external_producer_fixtures import prepared_external
from external_replay_codec_fixtures import encode
from stopped_study_authorization_fixtures import (
    change_frame,
    make_stopped,
    refresh_accounting,
)
from stopped_study_cell_process_fixtures import finalization, process_records

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_schedule import cell_for_ordinal
from automated_phishing_detection.replay_manifest_codec import encode_replay_manifest


def _inputs(root, cell, manifests):
    source = json.loads(root.contents["source-results.json"])
    manifest = (
        encode(prepared_external(1001).retained)
        if cell.workload == "shift_period"
        else encode_replay_manifest(manifests[cell.prevalence_basis_points])
    )
    descriptor = canonical_bytes(
        dict(
            schema_version=1,
            kind="operational-cell-descriptor-v1",
            root_reservation_sha256=root.snapshot.reservation_sha256,
            cell=asdict(cell),
            manifest_sha256=sha256(manifest).hexdigest(),
            accepted_inputs_sha256=source["accepted_inputs_sha256"],
        )
    )
    return {"descriptor.json": descriptor, "manifest": manifest}, source[
        "accepted_inputs"
    ]


def _reservation(root, cell, inputs, metadata):
    identity = dict(
        kind="operational_cell",
        protocol="operational-cell-v1",
        **metadata["execution"],
        operational_profile_sha256=metadata["operational_profile_sha256"],
        root_reservation_sha256=metadata["root_reservation_sha256"],
        descriptor_sha256=sha256(inputs["descriptor.json"]).hexdigest(),
    )
    directory = f"{root.authorization.paths.attempt.parent}/cells-dir/cell-{cell.ordinal:03d}-attempt"
    return receipt._json_bytes(
        dict(
            schema_version=1, status="reserved", directory=directory, identity=identity
        ),
        "fixture",
    )


def make_cell(prepared, manifests, *, ordinal=73, readiness=False):
    root = make_stopped(prepared, manifests, prefix=ordinal - 1, stopped_admissions=1)
    cell = cell_for_ordinal(ordinal)
    inputs, metadata = _inputs(root, cell, manifests)
    reservation = _reservation(root, cell, inputs, metadata)
    reservation_pin = sha256(reservation).hexdigest()
    inputs["binding.json"] = canonical_bytes(
        dict(
            schema_version=1,
            kind="operational-cell-binding-v1",
            descriptor_sha256=sha256(inputs["descriptor.json"]).hexdigest(),
            cell_reservation_sha256=reservation_pin,
        )
    )
    binding_pin = sha256(inputs["binding.json"]).hexdigest()
    payloads = (
        process_records(
            cell, reservation_pin, binding_pin, metadata, pid=888, readiness=readiness
        )
        | finalization(reservation_pin)
        | {"reservation.json": reservation}
    )
    root.accounting["authorization_ledger"]["admissions"][-1]["launched_pid"] = 888
    change_frame(root, -1, cell_binding_sha256=binding_pin)
    case = SimpleNamespace(root=root, inputs=inputs, values=payloads, ordinal=ordinal)
    refresh_cell(case)
    return case


def refresh_cell(case, *, progress=True):
    if progress:
        case.root.scientific["cells"][case.ordinal - 1].update(
            stage="observation",
            publishing=False,
            observation_sha256=None,
            reservation_sha256=sha256(case.values["reservation.json"]).hexdigest(),
            progress_sha256=sha256(case.values["process-pair.json"]).hexdigest(),
        )
        refresh_accounting(case.root)
    case.attempt_payloads = tuple(
        (f"attempt/{name}", content) for name, content in case.values.items()
    )
    case.input_payloads = tuple(case.inputs.items())
    case.attempt_pins = {
        name: sha256(content).hexdigest() for name, content in case.attempt_payloads
    }
    case.input_pins = {
        name: sha256(content).hexdigest() for name, content in case.input_payloads
    }


def verify_cell(case, **overrides):
    from automated_phishing_detection.stopped_study_cell import (
        verify_stopped_study_cell_history,
    )

    arguments = dict(
        expected_profile_sha256=case.root.authorization.profile_sha256,
        expected_envelope_sha256=case.root.authorization.envelope_sha256,
        expected_root_snapshot_sha256=case.root.pins,
        expected_attempt_snapshot_sha256=case.attempt_pins,
        expected_input_snapshot_sha256=case.input_pins,
    )
    return verify_stopped_study_cell_history(
        case.root.snapshot,
        case.attempt_payloads,
        case.input_payloads,
        **(arguments | overrides),
    )


@pytest.fixture(scope="module")
def invented_cell_history(prepared, manifests):
    return make_cell(prepared, manifests, ordinal=2)


@pytest.fixture
def cell_history(invented_cell_history):
    return copy.deepcopy(invented_cell_history)


def change_record(case, name, **changes):
    value = json.loads(case.values[name]) | changes
    content = receipt._json_bytes(value, "fixture")
    if name not in ("reservation.json", "finalize.claim", "outcome.json"):
        content += b"\n"
    case.values[name] = content
    refresh_cell(case)
