from importlib import import_module

import pytest

from automated_phishing_detection import url_features


def extension():
    from importlib.util import find_spec

    name = "automated_phishing_detection.transport_neutral_features"
    assert find_spec(name) is not None, "transport-neutral representation is missing"
    return import_module(name)


@pytest.mark.parametrize(
    "suffix",
    [
        "safe.example",
        "SAFE.example/CaseSensitive?Key=Value#Fragment",
        "user:password@account.example:8443/login?return=%2Fhome",
        "xn--bcher-kva.example/path",
        "bücher.example/é?q=✓",
        "a.b.example/%2f?empty=#",
        "safe.example:443/path",
        "safe.example:80/path",
    ],
)
def test_all_features_are_exactly_scheme_invariant(suffix):
    module = extension()
    vectors = [
        module.extract_transport_neutral_features(f"{scheme}://{suffix}")
        for scheme in ("http", "https", "HTTP", "HTTPS")
    ]
    assert all(vector == vectors[0] for vector in vectors)
    assert len(vectors[0]) == len(module.FEATURE_NAMES) == 24
    assert module.REPRESENTATION_ID == "transport-neutral-structural-v1"


def test_model_input_changes_only_scheme_and_does_not_rewrite_source():
    module = extension()
    original = "HTTPS://User:Pass@MiXeD.example:443/Case%2f?Q=Value#Frag"
    transformed = module.transport_neutral_model_input(original)
    assert transformed == "http://User:Pass@MiXeD.example:443/Case%2f?Q=Value#Frag"
    assert original == "HTTPS://User:Pass@MiXeD.example:443/Case%2f?Q=Value#Frag"
    expected = tuple(
        value
        for name, value in zip(
            url_features.FEATURE_NAMES,
            url_features.extract_url_features(transformed),
            strict=True,
        )
        if name != "is_https"
    )
    assert module.extract_transport_neutral_features(original) == expected
    assert module.FEATURE_NAMES != url_features.FEATURE_NAMES


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        True,
        b"https://safe.example",
        "",
        "safe.example",
        "ftp://safe.example",
        "https://safe.example/with space",
        "https://safe.example/%QZ",
        "https://127.0.0.1",
        "https://safe.example:invalid",
    ],
)
def test_invalid_original_cannot_be_repaired_into_an_eligible_model_input(value):
    module = extension()
    with pytest.raises(url_features.FeatureExtractionError):
        module.transport_neutral_model_input(value)
    with pytest.raises(url_features.FeatureExtractionError):
        module.extract_transport_neutral_features(value)


def test_port_path_query_and_fragment_information_is_not_silently_removed():
    module = extension()
    base = module.extract_transport_neutral_features("https://safe.example")
    for suffix in (":8443", "/login", "?a=1", "#fragment"):
        assert (
            module.extract_transport_neutral_features("https://safe.example" + suffix)
            != base
        )
