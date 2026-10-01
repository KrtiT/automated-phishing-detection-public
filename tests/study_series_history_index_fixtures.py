"""Invented locator metadata only; no catalog, approval or research bytes."""

from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

from study_series_adoption_fixtures import digest
from study_series_history_index_profile_fixtures import series_profile

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_completion_records import _LOGICAL_NAMES
from automated_phishing_detection._internal_handoff_validation import SNAPSHOT_NAMES
from automated_phishing_detection._operational_cell_protocol import (
    SNAPSHOT_NAMES as CELLS,
)
from automated_phishing_detection._operational_input_files import CELL_NAMES
from automated_phishing_detection._stopped_study_cell_records import ATTEMPT_NAMES
from automated_phishing_detection._stopped_study_root import MEMBERS as ROOT
from automated_phishing_detection._stopped_study_timeline_records import (
    MEMBERS,
    SUPERVISOR_NAMES,
)


def api():
    name = "automated_phishing_detection.study_series_history_index"
    assert find_spec(name), "missing pure series history index validator"
    return import_module(name)


def reference(path, pin=None):
    return {"path": path, "sha256": pin or digest(path.encode())}


def attempts(value):
    selected = value["origin"]
    return [
        {
            "ordinal": ordinal,
            "profile": reference(
                f"/invented/history/attempt-{ordinal}/profile.json",
                selected["profile_sha256"] if ordinal == 4 else None,
            ),
            "envelope": reference(
                f"/invented/history/attempt-{ordinal}/envelope.json",
                selected["envelope_sha256"] if ordinal == 4 else None,
            ),
            "root_reservation_sha256": selected["root_reservation_sha256"]
            if ordinal == 4
            else digest(f"root-{ordinal}".encode()),
            "inventory": reference(
                f"/invented/history/attempt-{ordinal}/inventory.json"
            ),
            "interruption_review": reference(
                f"/invented/history/attempt-{ordinal}/review.json"
            ),
            "disposition": "selected_entire_eligible_prefix"
            if ordinal == 4
            else "preserve_excluded",
        }
        for ordinal in range(1, 5)
    ]


def sources(value, kind, names):
    prior = value["origin"]["profile"]
    locations = prior["paths"]
    payloads = {}
    for name in names:
        if name == "public-summary.json":
            path = locations[f"{kind}-public-summary"]
        elif name.startswith("source/"):
            path = locations["repo-root"] + "/" + name.removeprefix("source/")
        else:
            path = locations[f"{kind}-attempt"] + "/" + name.removeprefix("attempt/")
        pin = prior["source_artifact_scope"].get(name.removeprefix("source/"))
        if name.endswith("/bindings.json"):
            pin = value["scientific_pins"][f"{kind}_bindings_sha256"]
        payloads[name] = reference(path, pin)
    return payloads


def cell(value, ordinal, stopped=False):
    base = value["origin"]["profile"]["paths"]["cells-dir"] + f"/cell-{ordinal:03d}"
    names = ATTEMPT_NAMES if stopped else CELLS
    payloads = {
        name: reference(
            base + "-summary.json"
            if name == "public-summary.json"
            else base + "-attempt/" + name.removeprefix("attempt/")
        )
        for name in names
    }
    result = {
        "ordinal": ordinal,
        "payloads": payloads,
        "reservation_sha256": payloads["attempt/reservation.json"]["sha256"],
    }
    if stopped:
        result.update(
            inventory_kind="service_only_cancelled_v1",
            input_payloads={
                name: reference(base + "-inputs/" + name) for name in CELL_NAMES
            },
        )
    return result


def origin_records(value):
    prior = value["origin"]["profile"]
    ancestry = prior["continuation"]
    root = {
        name: reference(prior["paths"]["attempt"] + "/" + name.removeprefix("attempt/"))
        for name in ROOT
    }
    root["attempt/reservation.json"]["sha256"] = value["origin"][
        "root_reservation_sha256"
    ]
    root["attempt/study-accounting.json"]["sha256"] = value["segment"][
        "predecessor_accounting_sha256"
    ]
    return {
        "original_hold": {
            "profile": reference(
                "/invented/hold/profile.json", ancestry["prior_profile_sha256"]
            ),
            "envelope": reference(
                "/invented/hold/envelope.json", ancestry["prior_envelope_sha256"]
            ),
            "inventory": reference("/invented/hold/inventory.json"),
            "reservation_sha256": ancestry["prior_root_reservation_sha256"],
        },
        "selected_root": {
            "reservation_sha256": value["origin"]["root_reservation_sha256"],
            "payloads": root,
        },
    }


def make_case(prefix=2):
    value = series_profile(prefix)
    historical_attempts = attempts(value)
    index = {
        "schema_version": 1,
        "index_id": "study-series-history-index-v1",
        "selected_attempt_ordinal": 4,
        "attempts": historical_attempts,
        **origin_records(value),
        "accepted_sources": {
            "internal": sources(value, "internal", SNAPSHOT_NAMES),
            "external": sources(value, "external", _LOGICAL_NAMES),
        },
        "accepted_cells": [cell(value, ordinal) for ordinal in range(1, prefix + 1)],
        "stopped_cell": cell(value, prefix + 1, True),
        "physical_observations": {
            name: reference(f"/invented/physical/{name}") for name in MEMBERS
        },
        "supervisor_files": {
            name: reference(f"/invented/supervisors/{name}")
            for name in SUPERVISOR_NAMES
        },
        "interruption_review": deepcopy(historical_attempts[-1]["interruption_review"]),
    }
    case = SimpleNamespace(index=index, profile=value)
    return refresh(case)


def refresh(case):
    case.profile["history"]["index_sha256"] = digest(case.index)
    return case


def validate(case, **overrides):
    values = dict(
        index_bytes=canonical_bytes(case.index),
        profile_bytes=canonical_bytes(case.profile),
        expected_index_sha256=digest(case.index),
        expected_profile_sha256=digest(case.profile),
    )
    return api().validate_series_history_index(**(values | overrides))


def selected(value, path):
    for name in path.split("."):
        value = value[int(name)] if type(value) is list else value[name]
    return value
