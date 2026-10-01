"""Actual temporary receipts and complete invented pure prefix validation."""

import json
from dataclasses import replace

import pytest
from study_series_child_prefix_fixtures import api
from study_series_prefix_fixtures import (
    candidates,
    make_prefix_case,
    manifests,
    series_case,
)

from automated_phishing_detection import execution_receipt as receipt

__all__ = ["candidates", "manifests", "series_case"]


def actual(tmp_path, source):
    selected = make_prefix_case(
        source,
        paths={
            "series_attempt": str(tmp_path.resolve() / "series"),
            "segment_attempt": str(tmp_path.resolve() / "segment"),
        },
    )
    payloads = dict(selected.payloads)
    for kind, attempt in (("series", selected.series), ("segment", selected.segment)):
        value = json.loads(payloads[f"{kind}/reservation.json"])
        assert (
            receipt.reserve_attempt(attempt.directory, identity=value["identity"])
            == attempt
        )
    with receipt._attempt_directory(selected.segment) as directory:
        for name in ("segment-intent.json", "history-import.json"):
            receipt._install_record(directory, name, payloads[f"segment/{name}"])
    return selected


def test_complete_four_file_prefix_with_real_validator(tmp_path, series_case):
    selected = actual(tmp_path, series_case)
    with api().hold_series_child_prefix(selected.binding, selected.frame) as payloads:
        assert dict(payloads) == dict(selected.payloads)
        (selected.segment.directory / "new-accounting.json").write_bytes(b"invented")
        (selected.series.directory / "later-publication").mkdir(mode=0o700)


@pytest.mark.parametrize(
    "name",
    (
        "profile_sha256",
        "envelope_sha256",
        "history_index_sha256",
        "origin_reservation_sha256",
    ),
)
def test_mismatched_live_profile_frame_never_opens_root_files(
    tmp_path, series_case, monkeypatch, name
):
    selected = actual(tmp_path, series_case)

    def forbidden(*args, **kwargs):
        pytest.fail("wrong prefix context reached files")

    monkeypatch.setattr(api().files, "enter_directory", forbidden)
    frame = replace(selected.frame, **{name: "0" * 64})
    with pytest.raises(api().SeriesChildPrefixError):
        with api().hold_series_child_prefix(selected.binding, frame):
            pytest.fail("wrong context yielded")
