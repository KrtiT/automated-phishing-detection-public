"""Every original byte and independent inventory remains authenticated."""

import pytest
from study_history_cell_fixtures import (
    api,
    candidates,
    case,
    hashes,
    history,
    manifests,
    restore,
)

from automated_phishing_detection._operational_cell_protocol import SNAPSHOT_NAMES

__all__ = ["candidates", "case", "history", "manifests"]


@pytest.mark.parametrize("name", SNAPSHOT_NAMES)
def test_every_original_member_is_pinned(history, name):
    values = history.values | {name: history.values[name] + b"\n"}
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=tuple(values.items()))


@pytest.mark.parametrize(
    "kind", ("missing", "extra", "duplicate", "list", "member", "bytes")
)
def test_payload_inventory_is_closed(history, kind):
    pairs = tuple(history.values.items())
    variants = {
        "missing": pairs[:-1],
        "extra": (*pairs, ("extra", b"x")),
        "duplicate": (*pairs, pairs[0]),
        "list": list(pairs),
        "member": (list(pairs[0]), *pairs[1:]),
        "bytes": ((pairs[0][0], bytearray(pairs[0][1])), *pairs[1:]),
    }
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=variants[kind])


@pytest.mark.parametrize("kind", ("missing", "extra", "bad", "list"))
def test_independent_inventory_is_exact(history, kind):
    pins = dict(history.arguments["expected_snapshot_sha256"])
    name = next(iter(pins))
    variants = {
        "missing": {key: value for key, value in pins.items() if key != name},
        "extra": pins | {"extra": "a" * 64},
        "bad": pins | {name: "g" * 64},
        "list": list(pins.items()),
    }
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, expected_snapshot_sha256=variants[kind])


@pytest.mark.parametrize(
    "name",
    (
        "expected_descriptor_sha256",
        "expected_binding_sha256",
        "expected_cell_reservation_sha256",
    ),
)
def test_each_independent_context_pin_is_required(history, name):
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, **{name: "0" * 64})


@pytest.mark.parametrize(
    "name", ("descriptor_bytes", "binding_bytes", "accepted_metadata_bytes")
)
def test_original_context_bytes_cannot_be_substituted(history, name):
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, **{name: history.arguments[name] + b"\n"})


@pytest.mark.parametrize(
    "name", ("run.json", "warmup.json", "measured.json", "service-role.json")
)
def test_rehashed_working_and_published_copy_must_match(history, name):
    values = history.values | {
        f"attempt/{name}": history.values[f"attempt/{name}"] + b"\n"
    }
    with pytest.raises(api().HistoricalCellScienceError):
        restore(
            history,
            payloads=tuple(values.items()),
            expected_snapshot_sha256=hashes(values),
        )
