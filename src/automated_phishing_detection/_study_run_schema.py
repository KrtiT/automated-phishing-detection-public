"""Closed administrative root schemas, not scientific or process authority."""

import math

from . import _operational_input_schema as operational
from ._checkpoint_codec import canonical_bytes

PROTOCOL = "study-root-v1"
EXECUTION = operational.EXECUTION_FIELDS
STAGES = (
    "validation",
    "output_preflight",
    "reservation",
    "root_retention",
    "preparation",
    "preparation_retention",
    "prediction_barrier",
    "sources",
    "accepted_inputs",
    "cell_execution",
    "cell_compaction",
    "reduction",
    "root_publication",
    "root_finalization",
)
_INTERNAL = {"retained_rows", "positive_rows", "negative_rows", "positive_domains"}
_EXTERNAL = {
    "input_test_rows",
    "quarantined_test_rows",
    "retained_test_rows",
    "gold_rows",
    "gold_positive_domains",
    "certified_rows",
    "tranco_rows",
    "secondary_rows",
    "complete_windows",
}
_REQUIREMENTS = (
    "internal_negative_rows",
    "internal_positive_domains",
    "certified_rows",
    "gold_positive_domains",
    "tranco_rows",
    "external_complete_windows",
    "http_100_negative_rows",
    "http_100_positive_rows",
    "http_10_negative_rows",
    "http_10_positive_rows",
    "http_500_negative_rows",
    "http_500_positive_rows",
    "shift_warmup_rows",
)


class StudyRunRecordError(ValueError):
    """A root projection is malformed or inconsistent, never a capacity hold."""


def require(condition):
    if not condition:
        raise StudyRunRecordError("invalid_study_run_record")


def keys(value, names):
    require(type(value) is dict and set(value) == set(names))


def load(content):
    return operational.loads(content)


def same(first, second):
    require(canonical_bytes(first) == canonical_bytes(second))


def execution(value, *, reserved=True):
    names = {*EXECUTION, "kind", "protocol", "operational_profile_sha256"}
    keys(value, names | ({"reservation_sha256"} if reserved else set()))
    require(value["kind"] == "whole_study" and value["protocol"] == PROTOCOL)
    operational.shared_execution({name: value[name] for name in EXECUTION})
    operational.digest(value["operational_profile_sha256"])
    if reserved:
        operational.digest(value["reservation_sha256"])


def deadlines(value):
    keys(value, {"startup", "shutdown", "terminate", "kill"})
    require(
        all(
            type(number) in (int, float) and math.isfinite(number) and number > 0
            for number in value.values()
        )
    )


def envelope(status, context):
    execution(context)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": status,
        "execution": context,
    }


def record(content, context, extra):
    value = load(content)
    keys(value, {"schema_version", "protocol", "status", "execution", *extra})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["protocol"] == PROTOCOL and type(value["status"]) is str)
    same(value["execution"], context)
    return value


def _counts(value, names):
    keys(value, names)
    require(all(type(number) is int and number >= 0 for number in value.values()))


def feasibility(value):
    keys(value, {"schema_version", "protocol", "scope", "counts", "shortages"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["protocol"] == "study-preparation-feasibility-v1")
    require(value["scope"] == "necessary_population_capacity_only")
    keys(value["counts"], {"internal", "external"})
    _counts(value["counts"]["internal"], _INTERNAL)
    external = value["counts"]["external"]
    keys(external, _EXTERNAL | {"secondary_positive_strata"})
    _counts({name: external[name] for name in _EXTERNAL}, _EXTERNAL)
    strata = external["secondary_positive_strata"]
    keys(strata, {"ncsc_silver", "chongluadao_openphish_bronze"})
    for counts in strata.values():
        _counts(counts, {"rows", "domains"})
    _shortages(value["shortages"])


def _shortages(values):
    require(type(values) is list)
    previous = -1
    for value in values:
        keys(value, {"requirement", "category", "required", "available"})
        name = value["requirement"]
        require(type(name) is str and name in _REQUIREMENTS)
        position = _REQUIREMENTS.index(name)
        require(position > previous)
        previous = position
        require(
            value["category"]
            == (
                "descriptive"
                if position == 12
                else "sensitivity"
                if 8 <= position <= 11
                else "primary"
            )
        )
        require(type(value["required"]) is int and type(value["available"]) is int)
        require(0 <= value["available"] < value["required"])


def accepted_metadata(value, context):
    operational.validate_metadata(value)
    same(value["execution"], {name: context[name] for name in EXECUTION})
    require(value["root_reservation_sha256"] == context["reservation_sha256"])
    require(
        value["operational_profile_sha256"] == context["operational_profile_sha256"]
    )


def accounting(value):
    from ._study_run_accounting_projection import validate

    require(type(value["stage"]) is str and value["stage"] in STAGES)
    require(value["status"] in ("whole_study_hold", "matrix_accepted", "failed"))
    sources = value["internal_status"], value["external_status"]
    require(
        all(
            type(status) is str and status in ("unattempted", "accepted", "stopped")
            for status in sources
        )
    )
    require(sources[1] == "unattempted" or sources[0] == "accepted")
    cells = value["cells"]
    validate(cells)
    statuses = tuple(cell["status"] for cell in cells)
    if value["status"] == "whole_study_hold":
        require(
            sources == ("unattempted", "unattempted")
            and set(statuses) == {"unattempted"}
        )
    if value["status"] == "matrix_accepted":
        require(all(cell["retention"] == "compact" for cell in cells))
        require(sources == ("accepted", "accepted") and set(statuses) == {"accepted"})
    if any(status != "unattempted" for status in statuses):
        require(sources == ("accepted", "accepted"))
