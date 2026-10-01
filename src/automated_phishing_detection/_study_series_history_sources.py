"""Join retained source bytes to original receipts and unchanged science."""

import json

from ._checkpoint_codec import canonical_bytes
from ._external_preparation_outputs import _PUBLIC_INPUTS
from ._study_history_snapshot_sources import external_snapshot, internal_snapshot
from ._study_series_history_context import payloads, pins
from .study_history_external import verify_historical_external_science
from .study_history_internal import verify_historical_internal_science


def _internal(values, expected, origin):
    metadata = origin.metadata["internal"]
    internal_snapshot(tuple(values.items()), expected, metadata, origin.profile)
    source_pins = {
        name.removeprefix("source/"): pin
        for name, pin in expected.items()
        if name.startswith("source/")
    }
    return verify_historical_internal_science(
        values,
        expected_snapshot_sha256=expected,
        expected_execution=metadata["execution"],
        expected_source_sha256=source_pins,
        expected_attempt_directory=origin.profile["paths"]["internal-attempt"],
    )


def _external(values, expected, internal, origin):
    metadata = origin.metadata["external"]
    external_snapshot(
        tuple(values.items()),
        expected,
        metadata,
        origin.metadata["internal"],
        internal,
        origin.profile,
    )
    source_pins = {
        name: expected["attempt/evidence/" + retained]
        for name, retained in _PUBLIC_INPUTS
    }
    profile = json.loads(values["public-summary.json"])["source_profile"]
    return verify_historical_external_science(
        values,
        expected_snapshot_sha256=expected,
        expected_execution=metadata["execution"],
        expected_source_sha256=source_pins,
        expected_profile_bytes=canonical_bytes(profile),
        expected_handoff_bytes=canonical_bytes(origin.metadata["internal"]),
        expected_overlap_bytes=internal["attempt/checkpoints/source-overlap.json"],
        expected_attempt_directory=origin.profile["paths"]["external-attempt"],
    )


def restore_sources(reader, value, origin):
    members = value["accepted_sources"]
    internal = payloads(reader, members["internal"])
    restored = _internal(internal, pins(members["internal"]), origin)
    external = payloads(reader, members["external"])
    return restored, _external(external, pins(members["external"]), internal, origin)
