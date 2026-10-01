"""Rejoin invented outer hashes to reach historical semantic checks."""

import json
from dataclasses import replace
from types import SimpleNamespace

from study_history_cell_fixtures import digest, hashes

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.evaluation_manifest import _manifest_bytes


def reordered_manifest(manifest):
    rows = manifest.records[::-1]
    content = _manifest_bytes(manifest.prevalence_basis_points, rows)
    return replace(manifest, records=rows, sha256=digest(content))


def _publication(values, execution):
    public = json.loads(values["public-summary.json"])
    public["execution"] = execution
    values["public-summary.json"] = receipt._json_bytes(public, "fixture")
    for name in ("attempt/finalize.claim", "attempt/outcome.json"):
        value = json.loads(values[name])
        value["reservation_sha256"] = execution["reservation_sha256"]
        if name == "attempt/outcome.json":
            value["public_summary_sha256"] = digest(values["public-summary.json"])
        values[name] = receipt._json_bytes(value, "fixture")


def _reservation(values, metadata, descriptor):
    execution = json.loads(values["public-summary.json"])["execution"]
    execution.update(metadata["execution"])
    for field in ("root_reservation_sha256", "operational_profile_sha256"):
        execution[field] = metadata[field]
    execution["descriptor_sha256"] = digest(descriptor)
    reservation = json.loads(values["attempt/reservation.json"])
    reservation["identity"] = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    values["attempt/reservation.json"] = receipt._json_bytes(reservation, "fixture")
    execution["reservation_sha256"] = digest(values["attempt/reservation.json"])
    _publication(values, execution)
    return execution["reservation_sha256"]


def rebind_context(history, *, metadata=None, descriptor=None):
    arguments, values = dict(history.arguments), dict(history.values)
    metadata = metadata or json.loads(arguments["accepted_metadata_bytes"])
    accepted = canonical_bytes(metadata)
    described = descriptor or json.loads(arguments["descriptor_bytes"])
    described["accepted_inputs_sha256"] = digest(accepted)
    described["root_reservation_sha256"] = metadata["root_reservation_sha256"]
    descriptor_bytes = canonical_bytes(described)
    reservation = _reservation(values, metadata, descriptor_bytes)
    binding = json.loads(arguments["binding_bytes"])
    binding.update(
        descriptor_sha256=digest(descriptor_bytes), cell_reservation_sha256=reservation
    )
    arguments.update(
        accepted_metadata_bytes=accepted,
        descriptor_bytes=descriptor_bytes,
        binding_bytes=canonical_bytes(binding),
        expected_descriptor_sha256=digest(descriptor_bytes),
        expected_binding_sha256=digest(canonical_bytes(binding)),
        expected_cell_reservation_sha256=reservation,
        expected_snapshot_sha256=hashes(values),
    )
    return SimpleNamespace(values=values, arguments=arguments, working=history.working)


def rebind_source_payloads(history, first, second):
    metadata = json.loads(history.arguments["accepted_metadata_bytes"])
    metadata["internal"]["snapshot_sha256"] = hashes(first)
    handoff = canonical_bytes(metadata["internal"])
    execution = metadata["external"]["execution"]
    execution["internal_handoff_sha256"] = digest(handoff)
    for directory in ("checkpoints", "evidence"):
        second[f"attempt/{directory}/internal-source-handoff.json"] = handoff
    public = json.loads(second["public-summary.json"])
    public["execution"] = execution
    second["public-summary.json"] = canonical_bytes(public)
    metadata["external"]["snapshot_sha256"] = hashes(second)
    changed = rebind_context(history, metadata=metadata)
    for role, values in (("internal", first), ("external", second)):
        name = f"{role}_snapshot"
        changed.arguments[name] = replace(
            changed.arguments[name], payloads=tuple(values.items())
        )
    return changed
