"""Rehashed context and cell substitutions reject before any numerical work."""

import json
from dataclasses import replace

import pytest
from study_series_reduction_fixtures import digest, matrix, reduce, replace_inputs
from test_study_series_reduction import api

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["matrix"]


@pytest.fixture
def no_runs(monkeypatch):
    from automated_phishing_detection import study_history_cell, study_series_cell

    def forbidden(*arguments, **keywords):
        pytest.fail("invalid input reached numerical run decoding")

    monkeypatch.setattr(study_history_cell, "decode_run", forbidden)
    monkeypatch.setattr(study_series_cell, "decode_run", forbidden)


@pytest.mark.parametrize(
    "change", ["list", "short", "reorder", "duplicate", "cutover", "types"]
)
def test_exact_closed_ordered_prefix_and_suffix(matrix, no_runs, change):
    old, new = matrix.historical, matrix.fresh
    if change == "list":
        old = list(old)
    elif change == "short":
        new = new[:-1]
    elif change == "reorder":
        old = old[1:2] + old[:1] + old[2:]
    elif change == "duplicate":
        new = new[:-1] + new[-2:-1]
    elif change == "cutover":
        old, new = old[:-1], old[-1:] + new
    else:
        old = old[:-1] + new[:1]
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, historical=old, fresh=new)


@pytest.mark.parametrize("name", ["profile", "selected_metadata", "current_context"])
@pytest.mark.parametrize("changed", ["pin", "bytes", "rehashed"])
def test_independent_context_and_profile_authentication(matrix, no_runs, name, changed):
    key, pin = f"{name}_bytes", f"expected_{name}_sha256"
    changes = {pin: "0" * 64}
    if changed != "pin":
        content = matrix.arguments[key] + b"\n"
        changes = {key: content}
        if changed == "rehashed":
            changes[pin] = digest(content)
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, **changes)


@pytest.mark.parametrize("position", [0, 71, 72, 124])
@pytest.mark.parametrize(
    "field",
    [
        "accepted_bytes",
        "descriptor_bytes",
        "binding_bytes",
        "manifest_bytes",
        "requests",
    ],
)
def test_each_cell_keeps_its_exact_source_and_context(matrix, no_runs, position, field):
    values = list(matrix.historical + matrix.fresh)
    original = values[position]
    inputs = original.inputs if position < 72 else original.inputs.computational
    member = getattr(inputs, field)
    changed = tuple(reversed(member)) if field == "requests" else member + b"\n"
    values[position] = replace_inputs(original, **{field: changed})
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, historical=tuple(values[:72]), fresh=tuple(values[72:]))


def test_duplicate_reservation_rejects_even_with_matching_rebound_binding(
    matrix, no_runs
):
    first, last = matrix.historical[0], matrix.fresh[-1]
    bound = json.loads(last.inputs.computational.binding_bytes)
    bound["cell_reservation_sha256"] = first.reservation_sha256
    last = replace_inputs(last, binding_bytes=canonical_bytes(bound))
    last = replace(last, reservation_sha256=first.reservation_sha256)
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, fresh=matrix.fresh[:-1] + (last,))


@pytest.mark.parametrize("kind", ["internal_snapshot", "external_snapshot"])
def test_selected_pair_rejects_foreign_source_payloads(matrix, no_runs, kind):
    snapshot = matrix.arguments[kind]
    payloads = dict(snapshot.payloads)
    payloads["public-summary.json"] += b"\n"
    changed = replace(snapshot, payloads=tuple(sorted(payloads.items())))
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, **{kind: changed})


def test_fresh_cell_origin_metadata_cannot_be_relabelled(matrix, no_runs):
    record = matrix.fresh[0]
    changed = replace(
        record,
        inputs=replace(
            record.inputs,
            origin_metadata_bytes=matrix.arguments["current_context_bytes"],
        ),
    )
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, fresh=(changed,) + matrix.fresh[1:])


@pytest.mark.parametrize(
    "changed", ["duplicate", "missing", "extra", "mutable", "buffer"]
)
def test_payload_inventory_is_immutable_and_exact(matrix, no_runs, changed):
    record = matrix.historical[0]
    payloads = record.payloads
    if changed == "duplicate":
        payloads += payloads[:1]
    elif changed == "missing":
        payloads = payloads[:-1]
    elif changed == "extra":
        payloads += (("extra", b"invented"),)
    elif changed == "mutable":
        payloads = list(payloads)
    else:
        payloads = ((payloads[0][0], bytearray(payloads[0][1])),) + payloads[1:]
    with pytest.raises(api().SeriesReductionError):
        reduce(
            matrix,
            historical=(replace(record, payloads=payloads),) + matrix.historical[1:],
        )
