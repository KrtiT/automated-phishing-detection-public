"""Closed locator references and original path projections without IO."""

from pathlib import Path
from types import SimpleNamespace

from . import _study_execution_schema as schema
from . import _study_series_adoption_schema as series
from ._study_run_paths import cell_paths

FIELDS = {
    "index": "schema_version index_id selected_attempt_ordinal attempts original_hold selected_root accepted_sources accepted_cells stopped_cell physical_observations supervisor_files interruption_review",
    "attempt": "ordinal profile envelope root_reservation_sha256 inventory interruption_review disposition",
    "hold": "profile envelope inventory reservation_sha256",
    "root": "reservation_sha256 payloads",
    "sources": "internal external",
    "cell": "ordinal reservation_sha256 payloads",
    "stopped": "ordinal reservation_sha256 inventory_kind payloads input_payloads",
}


def closed(value, kind):
    schema.closed(value, FIELDS[kind].split())


def reference(value, path=None, pin=None):
    schema.closed(value, ("path", "sha256"))
    schema.lexical_path(value["path"])
    schema.digest(value["sha256"])
    schema.require(path is None or value["path"] == path)
    schema.require(pin is None or value["sha256"] == pin)


def members(value, names, locations=None):
    schema.closed(value, names)
    for name, member in value.items():
        reference(member, None if locations is None else locations[name])


def locations(names, attempt, summary=None, repository=None):
    values = {}
    for name in names:
        if name == "public-summary.json":
            values[name] = summary
        elif name.startswith("source/"):
            values[name] = str(Path(repository) / name.removeprefix("source/"))
        else:
            values[name] = str(Path(attempt) / name.removeprefix("attempt/"))
    return values


def cell_locations(original, ordinal):
    return cell_paths(
        SimpleNamespace(
            accepted_inputs_directory=Path(original["accepted-inputs-dir"]),
            cells_directory=Path(original["cells-dir"]),
        ),
        ordinal,
    )


def _references(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for member in value.values():
                yield from _references(member)
    elif type(value) is list:
        for member in value:
            yield from _references(member)


def _history_pins(history):
    pins = {}
    for name in ("index", "eligible_prefix_review", "exposure_record"):
        path, digest = history[f"{name}_path"], history[f"{name}_sha256"]
        schema.require(pins.get(path, digest) == digest)
        pins[path] = digest
    return pins


def flatten(value, profile):
    pins = _history_pins(profile["history"])
    outputs = tuple(
        schema.lexical_path(path)
        for name, path in profile["paths"].items()
        if name != "repo_root"
    )
    retained = {}
    for member in _references(value):
        path, digest = member["path"], member["sha256"]
        schema.require(pins.get(path, digest) == digest)
        for output in outputs:
            series.disjoint(schema.lexical_path(path), output)
        pins[path] = retained[path] = digest
    return tuple(sorted(retained.items()))
