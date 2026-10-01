"""Authenticate original and current computational identities without authority."""

from . import _operational_input_schema as schema
from . import _study_history_cell_inputs as history
from . import _study_series_adoption_profile as profiles
from . import _study_series_policy as policy
from ._checkpoint_codec import canonical_bytes
from ._study_history_snapshot_records import digest

FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "root_reservation_sha256",
        "series_reservation_sha256",
        "profile_sha256",
        "history_index_sha256",
        "execution",
        "operational_profile_sha256",
        "primary",
        "origin",
    )
)
KIND = "study-series-operational-inputs-v1"


def _profile(content, expected):
    value = schema.authenticated(content, expected)
    profiles.profile(value, policy.policy_projection())
    return value


def _execution(profile):
    return {
        "revision": profile["execution"]["revision"],
        "execution_contract_sha256": profile["execution"]["contract_sha256"],
        "runtime_sha256": profile["execution"]["runtime_sha256"],
        "source_spec_sha256": profile["scientific_pins"]["source_spec_sha256"],
    }


def _preparation(origin, profile):
    for kind in ("internal", "external"):
        execution = origin[kind]["execution"]
        schema.require(execution["source_interface"] == "retained_study_preparation_v1")
        for name in ("reservation", "complete"):
            schema.require(
                execution[f"study_preparation_{name}_sha256"]
                == profile["origin"][f"preparation_{name}_sha256"]
            )


def _source_pins(origin, profile):
    selected = profile["scientific_pins"]
    schema.require(
        digest(canonical_bytes(origin["primary"]))
        == selected["primary_metadata_sha256"]
    )
    scope = profile["origin"]["profile"]["source_artifact_scope"]
    for name, field in (
        ("data/sources.json", "source_spec_sha256"),
        ("reports/phiusiil-preparation-summary.json", "preparation_summary_sha256"),
    ):
        schema.require(origin["internal"]["execution"][field] == scope[name])
    for kind in ("internal", "external"):
        schema.require(
            origin[kind]["snapshot_sha256"]["attempt/evidence/bindings.json"]
            == selected[f"{kind}_bindings_sha256"]
        )


def _origin(value, profile, internal, external):
    schema.validate_metadata(value)
    schema.require(
        value["root_reservation_sha256"] == profile["origin"]["root_reservation_sha256"]
    )
    schema.same(
        value["execution"],
        _execution(profile) | {"revision": profile["origin"]["revision"]},
    )
    schema.require(
        value["operational_profile_sha256"]
        == profile["components"]["original_operational"]
    )
    schema.require(
        value["external"]["execution"]["source_profile_sha256"]
        == profile["components"]["original_external"]
    )
    _preparation(value, profile)
    _source_pins(value, profile)
    history.sources(internal, external, value)


def _reservations(series, segment, origin):
    for value in (series, segment, origin):
        schema.digest(value)
    schema.require(len({series, segment, origin}) == 3)


def build(
    origin_bytes,
    profile_bytes,
    internal,
    external,
    *,
    expected_origin_sha256,
    expected_profile_sha256,
    series_reservation_sha256,
    segment_reservation_sha256,
):
    origin = schema.authenticated(origin_bytes, expected_origin_sha256)
    profile = _profile(profile_bytes, expected_profile_sha256)
    _origin(origin, profile, internal, external)
    _reservations(
        series_reservation_sha256,
        segment_reservation_sha256,
        origin["root_reservation_sha256"],
    )
    return _encode(
        origin,
        profile,
        expected_profile_sha256,
        series_reservation_sha256,
        segment_reservation_sha256,
    )


def _encode(origin, profile, profile_pin, series, segment):
    return canonical_bytes(
        {
            "schema_version": 1,
            "kind": KIND,
            "root_reservation_sha256": segment,
            "series_reservation_sha256": series,
            "profile_sha256": profile_pin,
            "history_index_sha256": profile["history"]["index_sha256"],
            "execution": _execution(profile),
            "operational_profile_sha256": profile["components"]["current_operational"],
            "primary": origin["primary"],
            "origin": origin,
        }
    )


def authenticate(
    metadata_bytes,
    profile_bytes,
    internal,
    external,
    *,
    expected_metadata_sha256,
    expected_profile_sha256,
):
    value = schema.authenticated(metadata_bytes, expected_metadata_sha256)
    profile = _profile(profile_bytes, expected_profile_sha256)
    schema.keys(value, FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["kind"] == KIND)
    _origin(value["origin"], profile, internal, external)
    _reservations(
        value["series_reservation_sha256"],
        value["root_reservation_sha256"],
        value["origin"]["root_reservation_sha256"],
    )
    _current(value, profile, expected_profile_sha256)
    return value, profile


def _current(value, profile, profile_pin):
    schema.require(value["profile_sha256"] == profile_pin)
    schema.require(value["history_index_sha256"] == profile["history"]["index_sha256"])
    schema.require(
        value["operational_profile_sha256"]
        == profile["components"]["current_operational"]
    )
    schema.same(value["execution"], _execution(profile))
    schema.same(value["primary"], value["origin"]["primary"])
