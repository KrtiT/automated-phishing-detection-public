"""Reject malformed joins and retain first interruptions across owner cleanup."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from operational_owner_fixtures import owner_case
from operational_runtime_fixtures import records, runtime
from test_operational_role_records import context
from test_operational_runtime import app_for, reject_startup

from automated_phishing_detection import bound_runtime

__all__ = ["records", "runtime"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "list", "pair"])
def test_loaded_hash_projection_never_discards_invalid_entries(
    runtime, records, monkeypatch, change
):
    import asyncio

    case = owner_case(monkeypatch)
    pairs = case.models.artifact_hashes
    changes = {
        "duplicate": pairs + (pairs[0],),
        "missing": pairs[1:],
        "extra": pairs + (("extra", "f" * 64),),
        "list": list(pairs),
        "pair": (list(pairs[0]),) + pairs[1:],
    }
    case.models.artifact_hashes = changes[change]
    app = app_for(runtime, records, case)
    asyncio.run(reject_startup(app))
    assert case.retained == [] and case.scorer.urls == []
    assert not case.active and not app.state.owner.is_alive


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "f" * 40),
        ("contract_sha256", "f" * 64),
        ("runtime_json", '{"changed":true}'),
        ("source_hashes", (("data/sources.json", "f" * 64),)),
    ],
)
def test_execution_mismatch_rejects_without_paths_or_owner(
    runtime, records, monkeypatch, field, value
):
    case = owner_case(monkeypatch)
    case.binding = replace(case.binding, **{field: value})
    with pytest.raises(runtime.OperationalRuntimeError):
        app_for(runtime, records, case)
    assert case.events == [] and case.scorer is None


@pytest.mark.parametrize("later", [OSError("cleanup"), KeyboardInterrupt("second")])
def test_first_writer_interruption_survives_final_binding_failure(
    runtime, records, monkeypatch, later
):
    case = owner_case(monkeypatch)
    first = KeyboardInterrupt("first")
    checks = []

    def checked(binding):
        checks.append(binding)
        if len(checks) == 3:
            raise later

    def retain(name, content):
        raise first

    monkeypatch.setattr(bound_runtime, "recheck_binding", checked)
    with pytest.raises(KeyboardInterrupt) as caught:
        with runtime._owner(
            case.binding, case.paths, case.inputs, context(records), retain
        ):
            pytest.fail("interrupted role writer yielded a scorer")
    assert caught.value is first and len(checks) == 3
    assert not case.active and case.scorer.urls == []


def test_failed_session_entry_is_not_retried(runtime, records, monkeypatch):
    case = owner_case(monkeypatch)
    error, entries = OSError("entry"), []

    @contextmanager
    def failed(binding, paths):
        entries.append((binding, paths))
        raise error
        yield

    monkeypatch.setattr(runtime, "open_bound_session", failed)
    with pytest.raises(OSError) as caught:
        with runtime._owner(
            case.binding, case.paths, case.inputs, context(records), lambda *args: None
        ):
            pytest.fail("failed entry yielded a scorer")
    assert caught.value is error and len(entries) == 1
    assert case.events == [] and case.retained == []
