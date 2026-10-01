"""Real private holders around wholly invented series computational inputs."""

import json
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

from study_series_child_inputs_fixtures import rebound

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_input_transport as storage
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor


def api():
    name = "automated_phishing_detection._study_series_child_operational"
    assert find_spec(name), "missing series operational child integration"
    return import_module(name)


def setup(child_case, series_case, tmp_path, monkeypatch):
    module, events = api(), []
    ordinal = child_case.frame.cell_ordinal
    profile = json.loads(child_case.profile)
    profile["paths"].update(
        historical_inputs_dir=str(tmp_path / "historical-inputs"),
        cells_dir=str(tmp_path / "cells"),
    )
    Path(profile["paths"]["cells_dir"]).mkdir(mode=0o700)
    attempt = receipt.reserve_attempt(
        tmp_path / "cells" / f"cell-{ordinal:03d}-attempt",
        identity={"invented": "cell"},
    )
    selected = rebound(child_case, profile_value=profile)
    selected.binding = bind_cell_descriptor(
        selected.descriptor, cell_reservation_sha256=attempt.reservation_sha256
    )
    selected.frame = replace(
        selected.frame, role="client", cell_binding_sha256=digest(selected.binding)
    )
    auth = authorization(series_case.original.binding, selected.profile, profile)
    case = SimpleNamespace(**locals())
    install(case, monkeypatch)
    write_inputs(case)
    return case


def authorization(original, profile_bytes, profile):
    base = replace(
        original,
        root=Path(profile["paths"]["repo_root"]),
        revision=profile["execution"]["revision"],
        source_hashes=tuple(sorted(profile["source_artifact_scope"].items())),
    )
    return SimpleNamespace(
        base=base,
        profile_bytes=profile_bytes,
        operational=SimpleNamespace(
            profile_sha256=profile["components"]["current_operational"]
        ),
    )


def digest(content):
    return sha256(content).hexdigest()


def install(case, monkeypatch):
    origin = canonical_bytes(json.loads(case.selected.metadata)["origin"])
    case.held = SimpleNamespace(
        authorization=case.auth,
        admission=SimpleNamespace(frame=case.selected.frame),
        prefix_payloads=(
            (
                "segment/history-import.json",
                canonical_bytes(
                    {
                        "origin_metadata_sha256": digest(origin),
                    }
                ),
            ),
        ),
        environment={
            "APD_STUDY_SERIES_ADMISSION_FD": "7",
            "APD_ATTEMPT_DIRECTORY": str(case.attempt.directory),
            "APD_RESERVATION_SHA256": case.attempt.reservation_sha256,
            "APD_BASE_URL": "http://127.0.0.1:12345",
        },
    )
    case.arguments = SimpleNamespace(
        role="client",
        cell_ordinal=case.ordinal,
        expected_binding_sha256=digest(case.selected.binding),
    )
    monkeypatch.setattr(
        case.module, "recheck_held_child", lambda held: case.events.append("recheck")
    )


def write_inputs(case):
    root = Path(case.profile["paths"]["historical_inputs_dir"])
    cell = Path(case.profile["paths"]["cells_dir"]) / f"cell-{case.ordinal:03d}-inputs"
    with storage.retain_operational_root_inputs(
        root, accepted_inputs=case.selected.metadata
    ):
        with storage.retain_operational_cell_inputs(
            cell,
            descriptor=case.selected.descriptor,
            binding=case.selected.binding,
            manifest=case.selected.manifest,
        ):
            pass


def hold(case):
    return case.module.held_operational(case.held, case.arguments)
