"""Rehash invented joined evidence to exercise semantic validation boundaries."""

import json
from hashlib import sha256

from stopped_study_authorization_fixtures import change_frame
from stopped_study_cell_fixtures import change_record, refresh_cell

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def rebind_inputs(case):
    descriptor_pin = sha256(case.inputs["descriptor.json"]).hexdigest()
    reservation = json.loads(case.values["reservation.json"])
    reservation["identity"]["descriptor_sha256"] = descriptor_pin
    case.values["reservation.json"] = receipt._json_bytes(reservation, "fixture")
    reservation_pin = sha256(case.values["reservation.json"]).hexdigest()
    bound = json.loads(case.inputs["binding.json"])
    bound.update(
        descriptor_sha256=descriptor_pin, cell_reservation_sha256=reservation_pin
    )
    case.inputs["binding.json"] = canonical_bytes(bound)
    binding_pin = sha256(case.inputs["binding.json"]).hexdigest()
    change_frame(case.root, -1, cell_binding_sha256=binding_pin)
    for name in (
        "process-pair-intent.json",
        "process-pair.json",
        "finalize.claim",
        "outcome.json",
    ):
        change_record(case, name, reservation_sha256=reservation_pin)
    change_record(case, "service-role.json", binding_sha256=binding_pin)
    refresh_cell(case)


def change_input(case, name, **changes):
    value = json.loads(case.inputs[name]) | changes
    case.inputs[name] = canonical_bytes(value)
    rebind_inputs(case)


def change_manifest(case, value):
    case.inputs["manifest"] = receipt._json_bytes(value, "fixture")
    change_input(
        case,
        "descriptor.json",
        manifest_sha256=sha256(case.inputs["manifest"]).hexdigest(),
    )
