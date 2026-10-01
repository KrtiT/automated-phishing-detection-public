"""Repin invented histories to exercise structural checks beyond byte hashes."""

import json

from stopped_study_authorization_fixtures import refresh_accounting
from study_history_snapshot_cell_fixtures import cells
from study_history_snapshot_fixtures import refresh_sources
from study_history_snapshot_source_fixtures import (
    digest,
    encoded,
    external_source,
    finish,
    hashes,
    reserve,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes


def repin_source(case, role):
    metadata = case.source["accepted_inputs"]
    metadata[role]["snapshot_sha256"] = hashes(getattr(case, role))
    if role == "internal":
        case.external = external_source(
            metadata["external"],
            metadata["internal"],
            case.internal,
            case.profile["paths"]["external-attempt"],
            case.source_profile,
        )
    case.source["accepted_inputs_sha256"] = digest(canonical_bytes(metadata))
    refresh_sources(case)
    cells(case)
    refresh_accounting(case)


def replace_source_record(case, role, name, mutate):
    payloads = getattr(case, role)
    value = json.loads(payloads[name])
    mutate(value)
    payloads[name] = encoded(value)
    if name == "public-summary.json":
        outcome = json.loads(payloads["attempt/outcome.json"])
        outcome["public_summary_sha256"] = digest(payloads[name])
        payloads["attempt/outcome.json"] = encoded(outcome)
    repin_source(case, role)


def replace_cell(case, name, content):
    ordinal, reservation, original = case.cells[0]
    payloads = dict(original) | {name: content}
    case.cells = ((ordinal, reservation, tuple(payloads.items())), *case.cells[1:])
    case.scientific["cells"][0]["snapshot_sha256"] = hashes(payloads)
    refresh_accounting(case)


def move_source(case, role):
    values = getattr(case, role)
    execution = case.source["accepted_inputs"][role]["execution"]
    reserve(values, execution, "/invented/unapproved-attempt")
    public = json.loads(values["public-summary.json"])
    public["execution"] = execution
    private = {
        name.removeprefix("attempt/evidence/"): content
        for name, content in values.items()
        if name.startswith("attempt/evidence/")
    }
    finish(values, public, private)
    repin_source(case, role)


def republish_cell(case, name, value):
    ordinal, reservation, original = case.cells[0]
    values = dict(original)
    for prefix in ("attempt", "attempt/evidence"):
        values[f"{prefix}/{name}"] = canonical_bytes(value)
    public = json.loads(values["public-summary.json"])
    private = {
        name.removeprefix("attempt/evidence/"): content
        for name, content in values.items()
        if name.startswith("attempt/evidence/")
    }
    public["private_sha256"] = hashes(private)
    finish(values, public, private)
    case.cells = ((ordinal, reservation, tuple(values.items())), *case.cells[1:])
    case.scientific["cells"][0]["snapshot_sha256"] = hashes(values)
    refresh_accounting(case)
