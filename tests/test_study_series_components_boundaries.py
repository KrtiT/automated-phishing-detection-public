"""Independent pins and current complete execution identity reject mutations."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_components_fixtures import api, arguments, case
from study_urlnorm_fixtures import digest


def reject(values):
    module = api()
    with pytest.raises(
        module.SeriesComponentError, match="^invalid_series_components$"
    ):
        module.verify_series_component_transition(**values)


@pytest.mark.parametrize("name", ["base", "external", "operational"])
@pytest.mark.parametrize("kind", ["none", "duck", "subclass"])
def test_exact_existing_input_types(name, kind):
    values = arguments(case())
    original = values[name]
    replacement = None
    if kind == "duck":
        replacement = SimpleNamespace(**vars(original))
    elif kind == "subclass":
        replacement = type("Derived", (type(original),), {})(**vars(original))
    values[name] = replacement
    reject(values)


@pytest.mark.parametrize("name", ["profile_bytes", "audit_summary_bytes"])
@pytest.mark.parametrize(
    "content", [None, b"private malformed", b"{}\n", bytearray(b"{}")]
)
def test_independently_pinned_immutable_original_bytes(name, content):
    values = arguments(case())
    values[name] = content
    reject(values)


@pytest.mark.parametrize("pin", [None, True, "A" * 64, "1" * 64, "1" * 63])
def test_independent_profile_pin(pin):
    values = arguments(case())
    values["expected_profile_sha256"] = pin
    reject(values)


@pytest.mark.parametrize(
    "name,value",
    [
        ("root", Path("/invented/elsewhere")),
        ("root", "/invented/repository"),
        ("revision", "c" * 40),
        ("contract_sha256", "c" * 64),
        ("runtime_json", '{"invented_runtime":false}'),
        ("runtime_json", '{ "invented_runtime": true }'),
        ("runtime_json", '{"invented_runtime":true,"invented_runtime":true}'),
        ("runtime_json", '{"nonfinite":NaN}'),
        ("runtime_json", "[]"),
        ("runtime_json", b"{}"),
    ],
)
def test_complete_actual_base_identity(name, value):
    selected = case()
    selected.base = replace(selected.base, **{name: value})
    reject(arguments(selected))


@pytest.mark.parametrize(
    "kind", ["list", "reverse", "duplicate", "missing", "extra", "changed", "pair_list"]
)
def test_complete_sorted_unique_source_tuple(kind):
    selected = case()
    pins = selected.base.source_hashes
    replacements = {
        "list": list(pins),
        "reverse": tuple(reversed(pins)),
        "duplicate": (*pins, pins[0]),
        "missing": pins[1:],
        "extra": tuple(sorted((*pins, ("invented-extra", digest(b"extra"))))),
        "changed": ((pins[0][0], digest(b"changed")), *pins[1:]),
        "pair_list": (list(pins[0]), *pins[1:]),
    }
    selected.base = replace(selected.base, source_hashes=replacements[kind])
    reject(arguments(selected))


@pytest.mark.parametrize("kind", ["external", "operational"])
@pytest.mark.parametrize(
    "content", [None, b"{}\n", bytearray(b"{}\n"), b"[]\n", b"private"]
)
def test_component_bytes_are_immutable_canonical_and_independently_pinned(
    kind, content
):
    values = arguments(case())
    values[kind] = replace(values[kind], canonical_bytes=content)
    reject(values)


@pytest.mark.parametrize(
    "field",
    ["schema_version", "extra", "execution", "scientific_pins", "transition", "origin"],
)
def test_rehashed_profile_still_passes_complete_closed_schema(field):
    selected = case()
    selected.profile[field] = True
    reject(arguments(selected))


def test_rehashed_profile_source_spec_cannot_diverge_from_base():
    selected = case()
    selected.profile["scientific_pins"]["source_spec_sha256"] = digest(b"other-source")
    reject(arguments(selected))
