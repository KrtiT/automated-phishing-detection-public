"""Invented parent-bound bytes, not a live admission or real study input."""

import json
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from study_series_input_fixtures import (
    candidates,
    descriptor,
    digest,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_series_admission import SeriesAdmissionFrame
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor

__all__ = ["candidates", "manifests", "series_case", "child_case"]


def api():
    name = "automated_phishing_detection.study_series_child_inputs"
    assert find_spec(name), "missing byte-only series child restoration"
    return import_module(name)


def frame(case, ordinal):
    value = json.loads(case.metadata)
    return SeriesAdmissionFrame(
        role="service",
        profile_sha256=digest(case.profile),
        envelope_sha256="4" * 64,
        parent_pid=123,
        command_sha256="5" * 64,
        series_reservation_sha256=value["series_reservation_sha256"],
        segment_reservation_sha256=value["root_reservation_sha256"],
        origin_reservation_sha256=value["origin"]["root_reservation_sha256"],
        history_index_sha256=value["history_index_sha256"],
        intent_sha256="6" * 64,
        predecessor_sha256="7" * 64,
        accepted_inputs_sha256=digest(case.metadata),
        cell_binding_sha256=digest(case.binding),
        segment_ordinal=2,
        cell_ordinal=ordinal,
    )


@pytest.fixture(scope="module", params=(73, 111, 121))
def child_case(series_case, request):
    payloads = descriptor(series_case, request.param)
    case = SimpleNamespace(
        metadata=metadata(series_case),
        profile=canonical_bytes(series_case.profile),
        descriptor=payloads.descriptor_bytes,
        manifest=payloads.manifest_bytes,
        binding=bind_cell_descriptor(
            payloads.descriptor_bytes, cell_reservation_sha256="3" * 64
        ),
        expected=restore(series_case, request.param),
    )
    case.frame = frame(case, request.param)
    return case


def restored(case, **changes):
    arguments = dict(
        metadata_bytes=case.metadata,
        profile_bytes=case.profile,
        descriptor_bytes=case.descriptor,
        binding_bytes=case.binding,
        manifest_bytes=case.manifest,
        frame=case.frame,
        expected_cell_reservation_sha256="3" * 64,
    )
    return api().restore_series_child_inputs(**(arguments | changes))


def rebound(
    case, *, metadata_value=None, profile_value=None, described=None, body=None
):
    selected = SimpleNamespace(**vars(case))
    profile_value = json.loads(case.profile) if profile_value is None else profile_value
    metadata_value = (
        json.loads(case.metadata) if metadata_value is None else metadata_value
    )
    selected.profile = canonical_bytes(profile_value)
    metadata_value["profile_sha256"] = digest(selected.profile)
    selected.metadata = canonical_bytes(metadata_value)
    described = json.loads(case.descriptor) if described is None else described
    described["accepted_inputs_sha256"] = digest(selected.metadata)
    selected.manifest = case.manifest if body is None else body
    described["manifest_sha256"] = digest(selected.manifest)
    selected.descriptor = canonical_bytes(described)
    selected.binding = bind_cell_descriptor(
        selected.descriptor, cell_reservation_sha256="3" * 64
    )
    selected.frame = frame(selected, described["cell"]["ordinal"])
    return selected


def relink_origin(origin):
    from automated_phishing_detection._internal_handoff_validation import SOURCE_LINKS
    from automated_phishing_detection._operational_input_schema import EXECUTION_FIELDS

    first, second = origin["internal"], origin["external"]
    for field in EXECUTION_FIELDS:
        first["execution"][field] = origin["execution"][field]
        second["execution"][field] = origin["execution"][field]
    for field, name in SOURCE_LINKS.items():
        first["snapshot_sha256"][name] = first["execution"][field]
    for field in (
        "study_preparation_reservation_sha256",
        "study_preparation_complete_sha256",
    ):
        second["execution"][field] = first["execution"][field]
    expected = {
        "internal_handoff_sha256": digest(first),
        "internal_overlap_sha256": first["snapshot_sha256"][
            "attempt/checkpoints/source-overlap.json"
        ],
        "internal_reservation_sha256": first["execution"]["reservation_sha256"],
    }
    second["execution"].update(expected)
    for directory in ("checkpoints", "evidence"):
        for name, field in (
            ("internal-source-handoff.json", "internal_handoff_sha256"),
            ("internal-source-overlap.json", "internal_overlap_sha256"),
        ):
            second["snapshot_sha256"][f"attempt/{directory}/{name}"] = expected[field]
