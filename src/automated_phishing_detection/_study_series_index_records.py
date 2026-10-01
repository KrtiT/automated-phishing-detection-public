"""Complete declared rosters and identity joins, not retained-file completeness."""

from . import _study_execution_schema as schema
from . import _study_series_adoption_schema as series
from . import _study_series_index_refs as refs
from ._external_completion_records import _LOGICAL_NAMES as EXTERNAL
from ._internal_handoff_validation import SNAPSHOT_NAMES as INTERNAL
from ._operational_cell_protocol import SNAPSHOT_NAMES as CELLS
from ._operational_input_files import CELL_NAMES
from ._stopped_study_cell_records import ATTEMPT_NAMES
from ._stopped_study_root import MEMBERS as ROOT
from ._stopped_study_timeline_records import MEMBERS, SUPERVISOR_NAMES


def _attempts(value, origin):
    schema.require(type(value) is list and len(value) == 4)
    for ordinal, attempt in enumerate(value, 1):
        refs.closed(attempt, "attempt")
        series.ordinal(attempt["ordinal"], ordinal, ordinal)
        schema.digest(attempt["root_reservation_sha256"])
        for name in ("profile", "envelope", "inventory", "interruption_review"):
            refs.reference(attempt[name])
        disposition = (
            "selected_entire_eligible_prefix" if ordinal == 4 else "preserve_excluded"
        )
        schema.require(attempt["disposition"] == disposition)
    for name in ("profile", "envelope"):
        refs.reference(value[-1][name], pin=origin[f"{name}_sha256"])
    schema.require(
        value[-1]["root_reservation_sha256"] == origin["root_reservation_sha256"]
    )
    _distinct_attempts(value)


def _distinct_attempts(value):
    schema.require(len({entry["root_reservation_sha256"] for entry in value}) == 4)
    for name in ("profile", "envelope", "inventory", "interruption_review"):
        for field in ("path", "sha256"):
            schema.require(len({entry[name][field] for entry in value}) == 4)


def _hold(value, prior):
    refs.closed(value, "hold")
    ancestry = prior["continuation"]
    for name in ("profile", "envelope"):
        refs.reference(value[name], pin=ancestry[f"prior_{name}_sha256"])
    refs.reference(value["inventory"])
    schema.require(
        value["reservation_sha256"] == ancestry["prior_root_reservation_sha256"]
    )


def _root(value, profile, paths):
    refs.closed(value, "root")
    schema.require(
        value["reservation_sha256"] == profile["origin"]["root_reservation_sha256"]
    )
    refs.members(value["payloads"], ROOT, refs.locations(ROOT, paths["attempt"]))
    refs.reference(
        value["payloads"]["attempt/reservation.json"], pin=value["reservation_sha256"]
    )
    refs.reference(
        value["payloads"]["attempt/study-accounting.json"],
        pin=profile["segment"]["predecessor_accounting_sha256"],
    )


def _source_pins(value, profile):
    source_scope = profile["origin"]["profile"]["source_artifact_scope"]
    for name in ("data/sources.json", "reports/phiusiil-preparation-summary.json"):
        refs.reference(value["internal"][f"source/{name}"], pin=source_scope[name])
    refs.reference(
        value["internal"]["source/data/sources.json"],
        pin=profile["scientific_pins"]["source_spec_sha256"],
    )
    for kind, directory in (
        ("internal", "scientific-checkpoints"),
        ("external", "checkpoints"),
    ):
        for copy in (directory, "evidence"):
            refs.reference(
                value[kind][f"attempt/{copy}/bindings.json"],
                pin=profile["scientific_pins"][f"{kind}_bindings_sha256"],
            )


def _sources(value, profile, paths):
    refs.closed(value, "sources")
    for kind, names in (("internal", INTERNAL), ("external", EXTERNAL)):
        locations = refs.locations(
            names,
            paths[f"{kind}-attempt"],
            paths[f"{kind}-public-summary"],
            paths["repo-root"],
        )
        refs.members(value[kind], names, locations)
    _source_pins(value, profile)


def _cell(value, paths, ordinal, stopped=False):
    refs.closed(value, "stopped" if stopped else "cell")
    series.ordinal(value["ordinal"], ordinal, ordinal)
    schema.digest(value["reservation_sha256"])
    selected = refs.cell_locations(paths, ordinal)
    names = ATTEMPT_NAMES if stopped else CELLS
    refs.members(
        value["payloads"],
        names,
        refs.locations(names, selected.attempt, str(selected.public_summary)),
    )
    refs.reference(
        value["payloads"]["attempt/reservation.json"], pin=value["reservation_sha256"]
    )
    if stopped:
        schema.require(value["inventory_kind"] == "service_only_cancelled_v1")
        locations = {
            name: str(selected.cell_input_directory / name) for name in CELL_NAMES
        }
        refs.members(value["input_payloads"], CELL_NAMES, locations)


def _cells(value, stopped, profile, paths):
    start = profile["segment"]["start_ordinal"]
    schema.require(type(value) is list and len(value) == start - 1)
    for ordinal, cell in enumerate(value, 1):
        _cell(cell, paths, ordinal)
    _cell(stopped, paths, start, True)


def index(value, profile):
    refs.closed(value, "index")
    series.version(value["schema_version"])
    schema.require(value["index_id"] == "study-series-history-index-v1")
    series.ordinal(value["selected_attempt_ordinal"], 4, 4)
    prior = profile["origin"]["profile"]
    paths = prior["paths"]
    _attempts(value["attempts"], profile["origin"])
    _hold(value["original_hold"], prior)
    _root(value["selected_root"], profile, paths)
    _sources(value["accepted_sources"], profile, paths)
    _cells(value["accepted_cells"], value["stopped_cell"], profile, paths)
    refs.members(value["physical_observations"], MEMBERS)
    refs.members(value["supervisor_files"], SUPERVISOR_NAMES)
    refs.reference(value["interruption_review"])
    schema.require(
        value["interruption_review"] == value["attempts"][-1]["interruption_review"]
    )
