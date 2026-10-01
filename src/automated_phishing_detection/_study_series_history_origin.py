"""Reconstruct original authority and sampled interruption evidence from pins."""

import json
from dataclasses import dataclass, field

from . import _study_execution_schema as schema
from ._adopted_study_records import decoded
from ._checkpoint_codec import canonical_bytes
from ._study_root_records import StudyRootSnapshot
from ._study_series_history_context import payloads, pins, reference
from .stopped_study_authorization import verify_stopped_study_authorization
from .stopped_study_cell import verify_stopped_study_cell_history
from .stopped_study_timeline import verify_stopped_study_timeline


@dataclass(frozen=True)
class HistoricalOrigin:
    snapshot: StudyRootSnapshot = field(repr=False)
    authority: object = field(repr=False)
    arguments: dict = field(repr=False)
    metadata_bytes: bytes = field(repr=False)
    metadata: dict = field(repr=False)
    profile: dict = field(repr=False)
    accounting: dict = field(repr=False)
    scientific: dict = field(repr=False)


def restore_origin(reader, value, profile, index):
    selected = value["selected_root"]
    snapshot = StudyRootSnapshot(
        selected["reservation_sha256"],
        tuple(payloads(reader, selected["payloads"]).items()),
    )
    arguments = dict(
        expected_profile_sha256=profile["origin"]["profile_sha256"],
        expected_envelope_sha256=profile["origin"]["envelope_sha256"],
        expected_snapshot_sha256=pins(selected["payloads"]),
    )
    authority = verify_stopped_study_authorization(snapshot, **arguments)
    schema.require(
        authority.profile_bytes == canonical_bytes(profile["origin"]["profile"])
    )
    schema.require(authority.accepted_ordinals == index.accepted_ordinals)
    schema.require(authority.stopped_ordinal == index.stopped_ordinal)
    return _origin(snapshot, authority, arguments)


def _origin(snapshot, authority, arguments):
    original = schema.parse(authority.profile_bytes, authority.profile_sha256)
    accounting = json.loads(authority.accounting_bytes)
    scientific = json.loads(decoded(accounting["scientific_accounting_bytes"]))
    metadata = json.loads(authority.source_results_bytes)["accepted_inputs"]
    return HistoricalOrigin(
        snapshot,
        authority,
        arguments,
        canonical_bytes(metadata),
        metadata,
        original,
        accounting,
        scientific,
    )


def verify_physical(reader, value, origin):
    observations = payloads(reader, value["physical_observations"])
    for member in value["supervisor_files"].values():
        reference(reader, member)
    verify_stopped_study_timeline(
        origin.snapshot,
        tuple(observations.items()),
        **origin.arguments,
        expected_observation_sha256=pins(value["physical_observations"]),
        expected_supervisor_sha256=pins(value["supervisor_files"]),
    )
    selected = value["stopped_cell"]
    arguments = dict(origin.arguments)
    arguments["expected_root_snapshot_sha256"] = arguments.pop(
        "expected_snapshot_sha256"
    )
    stopped = verify_stopped_study_cell_history(
        origin.snapshot,
        tuple(payloads(reader, selected["payloads"]).items()),
        tuple(payloads(reader, selected["input_payloads"]).items()),
        **arguments,
        expected_attempt_snapshot_sha256=pins(selected["payloads"]),
        expected_input_snapshot_sha256=pins(selected["input_payloads"]),
    )
    schema.require(stopped.cell_ordinal == selected["ordinal"])
    schema.require(stopped.cell_reservation_sha256 == selected["reservation_sha256"])
