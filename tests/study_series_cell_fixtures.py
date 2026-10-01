"""Full synthetic publications using the distinct current-series input kind."""

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from operational_cell_acceptance_fixtures import _reservation, http_run
from operational_cell_process_fixtures import process_records
from operational_cell_shift_fixtures import scored_source
from shift_run_codec_fixtures import checkpoints as shift_checkpoints
from study_reduction_fixtures import _shift
from study_series_input_fixtures import (
    api as inputs_api,
)
from study_series_input_fixtures import (
    candidates,
    descriptor,
    digest,
    manifests,
    metadata,
    series_case,
)
from test_operational_cell_publication import published
from test_study_series_cell import api

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import (
    PRIVATE_NAMES,
    PROTOCOL,
)
from automated_phishing_detection.http_replay import summarize_run
from automated_phishing_detection.http_run_checkpoints import _checkpoint
from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor
from automated_phishing_detection.shift_replay import summarize_shift_run
from automated_phishing_detection.shift_run_codec import encode_shift_run

__all__ = ["candidates", "manifests", "series_case", "fresh_cell"]


def _context(case, ordinal):
    selected = SimpleNamespace(**vars(case))
    selected.profile = copy.deepcopy(case.profile)
    selected.profile["segment"]["start_ordinal"] = min(ordinal, 73)
    selected.external = scored_source(case.original).external.snapshot
    content = metadata(selected)
    described = descriptor(selected, ordinal, content=content)
    value = json.loads(content)
    identity = dict(
        kind="operational_cell",
        protocol=PROTOCOL,
        **value["execution"],
        operational_profile_sha256=value["operational_profile_sha256"],
        root_reservation_sha256=value["root_reservation_sha256"],
        descriptor_sha256=digest(described.descriptor_bytes),
    )
    attempt, reservation = _reservation(identity)
    binding = bind_cell_descriptor(
        described.descriptor_bytes, cell_reservation_sha256=attempt.reservation_sha256
    )
    return selected, content, described, attempt, reservation, binding, identity


def _arguments(context):
    selected, content, described, attempt, unused, binding, unused_identity = context
    return dict(
        metadata_bytes=content,
        profile_bytes=canonical_bytes(selected.profile),
        internal_snapshot=selected.internal,
        external_snapshot=selected.external,
        descriptor_bytes=described.descriptor_bytes,
        binding_bytes=binding,
        manifest_bytes=described.manifest_bytes,
        expected_metadata_sha256=digest(content),
        expected_profile_sha256=digest(selected.profile),
        expected_descriptor_sha256=digest(described.descriptor_bytes),
        expected_binding_sha256=digest(binding),
        expected_cell_reservation_sha256=attempt.reservation_sha256,
        expected_attempt_directory=str(attempt.directory),
    )


def _inputs(context):
    selected, content, described, attempt, unused, binding, unused_identity = context
    return inputs_api().restore_series_cell_inputs(
        content,
        canonical_bytes(selected.profile),
        selected.internal,
        selected.external,
        described.descriptor_bytes,
        binding,
        described.manifest_bytes,
        expected_metadata_sha256=digest(content),
        expected_profile_sha256=digest(selected.profile),
        expected_binding_sha256=digest(binding),
        expected_cell_reservation_sha256=attempt.reservation_sha256,
    )


def _run(inputs):
    if inputs.cell.workload == "shift_period":
        run = _shift(inputs.cell, inputs)
        warmup, measured = shift_checkpoints(run)
        return (
            run,
            encode_shift_run(run),
            summarize_shift_run(run),
            {
                "warmup.json": warmup,
                "measured.json": measured,
            },
        )
    run = http_run(inputs)
    return (
        run,
        encode_http_run(run),
        summarize_run(run),
        {
            "warmup.json": _checkpoint(run, measured=False),
            "measured.json": _checkpoint(run, measured=True),
        },
    )


def _publication(values, inputs, attempt, identity, summary):
    private = {name: values[name] for name in PRIVATE_NAMES}
    working = SimpleNamespace(
        payloads=tuple(values.items()),
        private_outputs=private,
        reservation_sha256=attempt.reservation_sha256,
    )
    public = dict(
        schema_version=1,
        protocol=PROTOCOL,
        status="operational_evidence_published",
        execution=identity | {"reservation_sha256": attempt.reservation_sha256},
        cell=asdict(inputs.cell),
        summary=summary,
        private_sha256={name: digest(body) for name, body in private.items()},
    )
    return published(working, receipt._json_bytes(public, "invented"))


def make_cell(case, ordinal):
    context = _context(case, ordinal)
    selected, content, described, attempt, reservation, binding, identity = context
    carrier = _inputs(context)
    inputs = carrier.computational
    run, encoded, summary, checkpoints = _run(inputs)
    commands = (("/invented/python", "service"), ("/invented/python", "client"))
    deadlines = dict(startup=300, shutdown=180, terminate=10, kill=10)
    values, unused = process_records(
        inputs, attempt.reservation_sha256, commands, deadlines
    )
    values.update(checkpoints, **{"reservation.json": reservation, "run.json": encoded})
    payloads = _publication(values, inputs, attempt, identity, summary)
    arguments = _arguments(context) | {
        "expected_snapshot_sha256": {
            name: digest(body) for name, body in payloads.items()
        }
    }
    return SimpleNamespace(
        values=payloads, arguments=arguments, run=run, inputs=carrier, summary=summary
    )


@pytest.fixture(scope="module", params=(21, 111, 121))
def fresh_cell(series_case, request):
    return make_cell(series_case, request.param)


def verify(case, *, payloads=None, **changes):
    return api().verify_series_cell_science(
        tuple(case.values.items()) if payloads is None else payloads,
        **(case.arguments | changes),
    )
