"""Real root files around invented preparation and mocked source orchestration."""

import asyncio
import importlib
import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace

import study_run_context_fixtures as contexts
from test_study_run_paths import relocate

from automated_phishing_detection._study_preparation_records import (
    PreparedStudySnapshot,
)


def api():
    name = "automated_phishing_detection.study_runner"
    assert importlib.util.find_spec(name), "missing whole-study coordinator"
    return importlib.import_module(name)


def setup(tmp_path, prepared, monkeypatch, preparation=None):
    module = api()
    paths = relocate(contexts.paths(contexts.api()), tmp_path)
    paths.cells_directory.mkdir(parents=True, mode=0o700)
    paths.cells_directory.parent.chmod(0o700)
    retained = prepared.preparation if preparation is None else preparation
    case = SimpleNamespace(
        module=module,
        body=module.body,
        binding=prepared.binding,
        profile=prepared.profile,
        paths=paths,
        retained=retained,
        events=[],
        deadlines=contexts.deadlines(),
        fresh=PreparedStudySnapshot(retained.reservation_sha256, retained.payloads),
    )
    install(case, monkeypatch)
    return case


def forbidden(*arguments, **keywords):
    raise AssertionError("whole-study hold invoked scoring or reduction")


def install(case, monkeypatch):
    def prepare(binding, paths):
        assert binding is case.binding and paths is case.paths.preparation
        case.events.append("prepare")
        return case.fresh

    @contextmanager
    def hold(binding, paths, reservation, completion):
        assert binding is case.binding and paths is case.paths.external
        assert reservation == case.retained.reservation_sha256
        assert completion == case.retained.completion_sha256
        case.events.append("hold_preparation")
        try:
            yield case.retained
        finally:
            case.events.append("release_preparation")

    monkeypatch.setattr(
        case.module, "recheck_binding", lambda binding: case.events.append("binding")
    )
    monkeypatch.setattr(case.body, "_run_bound_preparation", prepare)
    monkeypatch.setattr(case.body, "held_preparation", hold)
    for name in (
        "_run_observed_prepared_sources",
        "build_accepted_inputs",
        "_run_bound_cell",
        "reduce_accepted_study",
    ):
        monkeypatch.setattr(case.body, name, forbidden)


def execute(case):
    return asyncio.run(
        case.module._run_bound_study(
            case.binding,
            case.profile,
            paths=case.paths,
            deadlines=case.deadlines,
        )
    )
