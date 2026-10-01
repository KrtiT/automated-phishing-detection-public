"""Invented full-size publications; fixture observations grant no authority."""

import importlib
import importlib.util
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from operational_cell_acceptance_fixtures import http_case, verify
from operational_cell_shift_fixtures import shift_case
from operational_input_fixtures import candidates, case, manifests
from test_operational_cell_publication import published

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_cell_acceptance as original

__all__ = ["candidates", "case", "manifests"]


def api():
    name = "automated_phishing_detection.study_history_cell"
    assert importlib.util.find_spec(name), "missing historical cell science verifier"
    return importlib.import_module(name)


def digest(content):
    return sha256(content).hexdigest()


def make_history(source, workload, *, prepared=None):
    fixture = shift_case if workload == "shift_period" else http_case
    working = verify(original, prepared or fixture(original, source))
    public = original.build_cell_public(
        working, reservation_sha256=working.reservation_sha256
    )
    values = published(working, receipt._json_bytes(public, "fixture"))
    inputs = working.inputs
    arguments = {
        "expected_snapshot_sha256": hashes(values),
        "accepted_metadata_bytes": inputs.accepted_bytes,
        "internal_snapshot": working.accepted.internal.snapshot,
        "external_snapshot": working.accepted.external.snapshot,
        "descriptor_bytes": inputs.descriptor_bytes,
        "binding_bytes": inputs.binding_bytes,
        "expected_descriptor_sha256": digest(inputs.descriptor_bytes),
        "expected_binding_sha256": inputs.binding_sha256,
        "expected_cell_reservation_sha256": working.reservation_sha256,
        "expected_attempt_directory": "/invented/cell",
    }
    return SimpleNamespace(values=values, arguments=arguments, working=working)


def hashes(values):
    return {name: digest(content) for name, content in values.items()}


def restore(history, *, payloads=None, **changes):
    return api().verify_historical_cell_science(
        tuple(history.values.items()) if payloads is None else payloads,
        **(history.arguments | changes),
    )


def republish(history, replacements):
    values = dict(history.values)
    values.update({f"attempt/{name}": value for name, value in replacements.items()})
    values.update(
        {f"attempt/evidence/{name}": value for name, value in replacements.items()}
    )
    public = json.loads(values["public-summary.json"])
    public["private_sha256"].update(hashes(replacements))
    values["public-summary.json"] = receipt._json_bytes(public, "fixture")
    outcome = json.loads(values["attempt/outcome.json"])
    outcome["private_sha256"] = public["private_sha256"]
    outcome["public_summary_sha256"] = digest(values["public-summary.json"])
    values["attempt/outcome.json"] = receipt._json_bytes(outcome, "fixture")
    return tuple(values.items()), hashes(values)


@pytest.fixture(scope="module", params=("fixed_cascade", "shift_period"))
def history(case, request):
    return make_history(case, request.param)
