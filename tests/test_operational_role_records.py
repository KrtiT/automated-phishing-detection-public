"""Closed role projections remain consistency records, not owned process proof."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from operational_runtime_fixtures import inputs, primary, records

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._process_support import command_hash

__all__ = ["records"]


def context(api):
    return api.OperationalRoleContext(
        123, ("/python", "/invented/service.py"), "http://127.0.0.1:1234"
    )


@pytest.mark.parametrize("ordinal", [1, 91, 121])
@pytest.mark.parametrize("role", ["service", "client"])
def test_exact_closed_role_roundtrip(records, ordinal, role):
    cell, actual = inputs(ordinal), context(records)
    content = (
        records.build_service_role(cell, actual, primary=primary())
        if role == "service"
        else records.build_client_role(cell, actual)
    )
    expected = {
        "schema_version": 1,
        "protocol": "operational-role-v1",
        "role": role,
        "binding_sha256": cell.binding_sha256,
        "pid": 123,
        "command_sha256": command_hash(actual.command),
        "base_url": actual.base_url,
        "workload": cell.cell.workload,
    }
    if role == "service":
        expected.update(primary())
    assert content == canonical_bytes(expected)
    assert (
        records.verify_role_record(content, inputs=cell, context=actual, role=role)
        is None
    )


def test_context_is_frozen_and_does_not_render_command(records):
    actual = context(records)
    with pytest.raises(FrozenInstanceError):
        actual.pid = 456
    assert "/invented" not in repr(actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", True),
        ("pid", 0),
        ("pid", 1.0),
        ("command", []),
        ("command", ()),
        ("command", ("",)),
        ("command", ("nul\0",)),
        ("command", (1,)),
        ("base_url", "http://localhost:1234"),
        ("base_url", "http://127.0.0.2:1234"),
        ("base_url", "http://127.0.0.1:0"),
        ("base_url", "http://127.0.0.1:65536"),
        ("base_url", "http://127.0.0.1:1234/"),
        ("base_url", "http://127.0.0.1:01234"),
        ("base_url", "http://127.0.0.1:1234?x"),
    ],
)
def test_context_rejects_nonexact_pid_command_or_endpoint(records, field, value):
    with pytest.raises(records.OperationalRoleError):
        replace(context(records), **{field: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("schema_version", True),
        ("protocol", "other"),
        ("role", "client"),
        ("binding_sha256", "f" * 64),
        ("pid", 456),
        ("command_sha256", "f" * 64),
        ("base_url", "http://127.0.0.1:5678"),
        ("workload", "transformer_only"),
    ],
)
def test_saved_role_joins_every_independent_context_field(records, name, value):
    cell, actual = inputs(), context(records)
    saved = json.loads(records.build_service_role(cell, actual, primary=primary()))
    saved[name] = value
    with pytest.raises(records.OperationalRoleError):
        records.verify_role_record(
            canonical_bytes(saved), inputs=cell, context=actual, role="service"
        )


@pytest.mark.parametrize(
    "change", ["extra", "missing", "noncanonical", "duplicate", "type"]
)
def test_saved_role_rejects_schema_and_canonical_mutations(records, change):
    cell, actual = inputs(), context(records)
    content = records.build_client_role(cell, actual)
    value = json.loads(content)
    if change == "extra":
        content = canonical_bytes(value | {"artifact_hashes": {}})
    elif change == "missing":
        del value["pid"]
        content = canonical_bytes(value)
    elif change == "noncanonical":
        content += b" "
    elif change == "duplicate":
        content = content.replace(b'"pid":123,', b'"pid":123,"pid":123,')
    else:
        content = content.decode()
    with pytest.raises(records.OperationalRoleError):
        records.verify_role_record(content, inputs=cell, context=actual, role="client")
