"""Invented prefix declarations and receipts, never historical provenance."""

import copy
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import digest, make_case, refresh
from study_series_input_fixtures import candidates, manifests, metadata, series_case

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_series_admission import SeriesAdmissionFrame
from automated_phishing_detection.study_series_execution import SeriesPublicBinding

__all__ = ["candidates", "manifests", "series_case", "prefix_case"]


def api():
    name = "automated_phishing_detection.study_series_prefix"
    assert find_spec(name), "missing pure series root prefix"
    return import_module(name)


def binding(case):
    header = make_case()
    header.envelope["profile"] = copy.deepcopy(case.profile)
    refresh(header)
    base = replace(
        case.original.binding,
        root=Path(case.profile["paths"]["repo_root"]),
        revision=case.profile["execution"]["revision"],
        source_hashes=tuple(sorted(case.profile["source_artifact_scope"].items())),
    )
    return SeriesPublicBinding(
        base,
        object(),
        object(),
        object(),
        canonical_bytes(header.policy),
        canonical_bytes(case.profile),
        canonical_bytes(header.envelope),
        Path("/invented/envelope.json"),
    )


def reserve(identity, directory):
    content = receipt._json_bytes(
        dict(
            schema_version=1, status="reserved", directory=directory, identity=identity
        ),
        "invented",
    )
    return receipt.Attempt(Path(directory), digest(content)), content


def frame(case, **changes):
    profile = case.source.profile
    return SeriesAdmissionFrame(
        **(
            dict(
                role="service",
                profile_sha256=case.binding.profile_sha256,
                envelope_sha256=case.binding.envelope_sha256,
                parent_pid=123,
                command_sha256="a" * 64,
                series_reservation_sha256=case.series.reservation_sha256,
                segment_reservation_sha256=case.segment.reservation_sha256,
                origin_reservation_sha256=profile["origin"]["root_reservation_sha256"],
                history_index_sha256=profile["history"]["index_sha256"],
                intent_sha256=digest(case.intent),
                predecessor_sha256=digest(case.imported),
                accepted_inputs_sha256=digest(case.metadata),
                cell_binding_sha256="b" * 64,
                segment_ordinal=2,
                cell_ordinal=profile["segment"]["start_ordinal"],
            )
            | changes
        )
    )


def imported(case, **changes):
    profile = case.source.profile
    expected = dict(
        origin_metadata_bytes=case.source.origin_bytes,
        imported_prefix_length=profile["segment"]["start_ordinal"] - 1,
        origin_accounting_sha256=profile["segment"]["predecessor_accounting_sha256"],
    )
    return api().history_import_bytes(
        case.binding, case.series, case.segment, case.metadata, **(expected | changes)
    )


def make_prefix_case(source, *, paths=None):
    source = SimpleNamespace(**vars(source))
    source.profile = copy.deepcopy(source.profile)
    source.profile["paths"].update({} if paths is None else paths)
    case = SimpleNamespace(source=source, binding=binding(source))
    paths = source.profile["paths"]
    case.series, series_bytes = reserve(
        api().series_identity(case.binding), paths["series_attempt"]
    )
    case.segment, segment_bytes = reserve(
        api().segment_identity(case.binding, case.series), paths["segment_attempt"]
    )
    case.metadata = metadata(
        source,
        series_reservation_sha256=case.series.reservation_sha256,
        segment_reservation_sha256=case.segment.reservation_sha256,
    )
    return complete_case(case, series_bytes, segment_bytes)


def complete_case(case, series_bytes, segment_bytes):
    case.imported = imported(case)
    case.intent = api().segment_intent_bytes(
        case.binding, case.series, case.segment, case.imported, case.metadata
    )
    case.payloads = (
        ("series/reservation.json", series_bytes),
        ("segment/reservation.json", segment_bytes),
        ("segment/segment-intent.json", case.intent),
        ("segment/history-import.json", case.imported),
    )
    case.frame = frame(case)
    return case


@pytest.fixture(scope="module")
def prefix_case(series_case):
    return make_prefix_case(series_case)
