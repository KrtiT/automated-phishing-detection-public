"""Every retained buffer is authenticated before its own parser runs."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from operational_input_fixtures import build, candidates, case, manifests
from test_operational_cell_inputs import module
from test_operational_inputs import module as inputs_module

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["candidates", "manifests", "case"]


@pytest.fixture
def buffers(case):
    api = module()
    accepted = build(inputs_module(), case)
    selected = api.build_cell_descriptor(accepted, cell_for_ordinal(121))
    binding = api.bind_cell_descriptor(
        selected.descriptor_bytes, cell_reservation_sha256="8" * 64
    )
    return [
        accepted.metadata_bytes,
        selected.descriptor_bytes,
        binding,
        selected.manifest_bytes,
    ]


def restore(buffers, **changes):
    options = {
        "expected_binding_sha256": sha256(buffers[2]).hexdigest(),
        "expected_cell_reservation_sha256": "8" * 64,
    }
    return module().restore_cell_inputs(*buffers, **(options | changes))


@pytest.mark.parametrize("index", range(4))
def test_wrong_buffer_hash_precedes_that_buffer_parse(buffers, monkeypatch, index):
    expected = sha256(buffers[2]).hexdigest()
    invalid = b"not-json-and-not-bound"
    buffers[index] = invalid
    original = json.loads

    def parse(content, *args, **kwargs):
        assert content != invalid, "unbound buffer parsed"
        return original(content, *args, **kwargs)

    monkeypatch.setattr(json, "loads", parse)
    with pytest.raises(ValueError):
        restore(buffers, expected_binding_sha256=expected)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    "change", ["extra", "missing", "version_bool", "kind", "noncanonical"]
)
def test_rehashed_closed_envelopes_reject_mutations(buffers, index, change):
    value = json.loads(buffers[index])
    if change == "extra":
        value["extra"] = 1
    elif change == "missing":
        del value["kind"]
    elif change == "version_bool":
        value["schema_version"] = True
    elif change == "kind":
        value["kind"] = "wrong"
    buffers[index] = canonical_bytes(value) + (
        b" " if change == "noncanonical" else b""
    )
    relink(buffers, index)
    with pytest.raises(ValueError):
        restore(buffers)


def relink(buffers, index):
    if index == 0:
        value = json.loads(buffers[1])
        value["accepted_inputs_sha256"] = sha256(buffers[0]).hexdigest()
        buffers[1] = canonical_bytes(value)
    if index < 2:
        value = json.loads(buffers[2])
        value["descriptor_sha256"] = sha256(buffers[1]).hexdigest()
        buffers[2] = canonical_bytes(value)


def test_independent_reservation_is_never_inferred_from_record(buffers):
    with pytest.raises(ValueError):
        restore(buffers, expected_cell_reservation_sha256="7" * 64)


def test_shift_manifest_must_match_accepted_external_payload(buffers, case):
    row = case.external.snapshot.rows[0].record
    from external_replay_codec_fixtures import encode

    rows = [scored.record for scored in case.external.snapshot.rows]
    rows[0] = replace(row, raw_url=row.raw_url.replace("https://", "HTTPS://"))
    buffers[3] = encode(rows)
    value = json.loads(buffers[1])
    value["manifest_sha256"] = sha256(buffers[3]).hexdigest()
    buffers[1] = canonical_bytes(value)
    relink(buffers, 1)
    with pytest.raises(ValueError):
        restore(buffers)


def test_internal_manifest_source_must_match_accepted_source(case):
    api = module()
    accepted = build(inputs_module(), case)
    selected = api.build_cell_descriptor(accepted, cell_for_ordinal(1))
    value = json.loads(selected.manifest_bytes)
    for row in value["records"]:
        row["record_id"] = row["record_id"].replace("a" * 64, "b" * 64)
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    described = json.loads(selected.descriptor_bytes)
    described["manifest_sha256"] = sha256(content).hexdigest()
    descriptor = canonical_bytes(described)
    bound = api.bind_cell_descriptor(descriptor, cell_reservation_sha256="8" * 64)
    with pytest.raises(ValueError):
        restore([accepted.metadata_bytes, descriptor, bound, content])
