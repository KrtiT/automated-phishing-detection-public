"""Real retained inputs with mocked orchestration, never owned-exit proof."""

import asyncio
import importlib
import importlib.util
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from operational_input_fixtures import build, source_case
from operational_profile_fixtures import REQUIRED

from automated_phishing_detection import operational_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_process_records import ProcessObservation
from automated_phishing_detection._operational_profile import (
    CandidateOperationalProfile,
    _projection,
)
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.operational_input_transport import (
    retain_operational_root_inputs,
)
from automated_phishing_detection.operational_schedule import cell_for_ordinal


def api():
    name = "automated_phishing_detection.operational_cell_runner"
    assert importlib.util.find_spec(name), "missing private operational cell runner"
    return importlib.import_module(name)


def accepted_source(manifests, root=None):
    source = source_case(manifests)
    hashes = {name: sha256(name.encode()).hexdigest() for name in REQUIRED}
    hashes.update(dict(source.binding.source_hashes))
    source.binding = replace(
        source.binding, source_hashes=tuple(sorted(hashes.items()))
    )
    if root is not None:
        from operational_cell_shift_fixtures import scored_source

        source = scored_source(source)
        source.binding = replace(source.binding, root=root)
    profile = CandidateOperationalProfile(canonical_bytes(_projection(source.binding)))
    source.profile = profile.profile_sha256
    accepted = build(operational_inputs, source)
    return source, profile, accepted


def setup(tmp_path, manifests, monkeypatch, ordinal=121, root=None):
    module = api()
    source, profile, accepted = accepted_source(manifests, root)
    paths = module.OperationalCellPaths(
        *(tmp_path / name for name in ("accepted", "inputs", "attempt", "public.json"))
    )
    with retain_operational_root_inputs(
        paths.accepted_inputs_directory, accepted_inputs=accepted.metadata_bytes
    ):
        pass
    return runner_case(
        module, source, profile, accepted, paths, tmp_path, ordinal, monkeypatch
    )


def runner_case(
    module, source, profile, accepted, paths, tmp_path, ordinal, monkeypatch
):
    case = SimpleNamespace(
        module=module,
        binding=source.binding,
        profile=profile,
        accepted=accepted,
        cell=cell_for_ordinal(ordinal),
        paths=paths,
        events=[],
        source=source,
        deadlines=dict(startup=10.0, shutdown=10.0, terminate=2.0, kill=2.0),
        artifacts=ArtifactPaths(
            *(tmp_path / name for name in ("length", "logistic", "transformer", "gmm"))
        ),
    )
    monkeypatch.setattr(
        module, "recheck_binding", lambda value: case.events.append("recheck")
    )
    return case


def execute(case, **changes):
    options = dict(paths=case.paths, artifacts=case.artifacts, deadlines=case.deadlines)
    return asyncio.run(
        case.module._run_bound_cell(
            case.binding, case.profile, case.accepted, case.cell, **(options | changes)
        )
    )


def orchestration(case, monkeypatch):
    case.observation = ProcessObservation(b"invented observed bytes")
    case.snapshot = object()
    case.completer = SimpleNamespace(publishing=False, working=None, candidate=None)

    def complete(**options):
        case.events.append("complete")
        assert options["accepted"] is case.accepted
        assert options["observation"] is case.observation
        assert options["expected_deadlines"] == case.deadlines
        case.inputs = options["inputs"]
        case.completer.publishing = True
        case.completer.working = object()
        case.completer.candidate = case.snapshot
        return case.snapshot

    case.completer.complete = complete
    monkeypatch.setattr(case.module, "hold_operational_cell", holder(case))
    monkeypatch.setattr(case.module, "_observe_pair_with_writer", observer(case))


def holder(case):
    @contextmanager
    def hold(attempt, public, *, expected_identity):
        case.events.append("holder_enter")
        case.attempt, case.identity = attempt, expected_identity
        try:
            yield case.completer
        finally:
            case.events.append("holder_exit")

    return hold


def observer(case):
    async def observe(attempt, *, service_command, client_command, deadlines, writer):
        case.events.append("observe")
        assert attempt is case.attempt
        case.commands = service_command, client_command
        writer(attempt, "process-pair-intent.json", b"invented parent intent")
        return case.observation

    return observe
