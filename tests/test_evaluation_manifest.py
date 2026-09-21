import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256

import numpy as np
import pytest

from automated_phishing_detection.evaluation_manifest import (
    InsufficientStratum,
    ManifestRecord,
    ReplayManifest,
    build_manifest,
)

SOURCE_HASH = "a" * 64
PREVALENCES = (10, 100, 500)
# Generated independently from fixed ordinal pools, before the selector existed.
GOLDENS = {
    10: (
        (2844, 3113, 10526, 6224, 5632, 6369, 6376, 8581, 4024, 7475, 8844, 4616),
        "8ef50aa4c486c53be039fd4bb26ea2abd82ce14d46d091d79cda1ad97110aa48",
        "bd3eeb72eac50f68f60ad1d47d1f480a38582946787ddccbf14a26be548683f3",
    ),
    100: (
        (806, 3266, 10496, 521, 2554, 9983, 4985, 7303, 4939, 4017, 4743, 4300),
        "5334138db5f033e29705dfa67e535969be5be64aeebb5173585c259664e83b4b",
        "1e1c673786b2d1d6fc405258fcdf013aaab03d2c00741367d68cb441e619ab89",
    ),
    500: (
        (4448, 11315, 2240, 10572, 4408, 5993, 2074, 3700, 3345, 628, 6198, 4330),
        "3d48a243856652f406dda5fdba72e376b0b369e896d595fa2f6174fc7feb32b1",
        "d68a239cd602956d372554c0637caad50868c51df500d35ec635f8d996755c0d",
    ),
}


def _record(ordinal, label=0):
    canonical = f"https://r{ordinal:05d}.example/caf\u00e9?row={ordinal}#fragment"
    return ManifestRecord(
        record_id=f"phiusiil-row-v1:{SOURCE_HASH}:{ordinal:016x}",
        raw_url=f"HTTPS://R{ordinal:05d}.Example:00443/caf\u00e9?row={ordinal}#fragment",
        canonical_url_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
        registrable_domain=f"r{ordinal:05d}.example",
        is_phishing=label,
        split="group_test",
    )


@pytest.fixture(scope="module")
def candidates():
    return tuple(_record(ordinal, int(ordinal > 11000)) for ordinal in range(1, 11701))


@pytest.fixture(scope="module")
def manifests(candidates):
    return {
        bp: build_manifest(candidates, prevalence_basis_points=bp) for bp in PREVALENCES
    }


def _reference_ordinals(bp):
    selected = []
    for label, base, size, required in (
        (0, 1, 11000, 10000 - bp),
        (1, 11001, 700, bp),
    ):
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([20260816, 1, bp, label]))
        )
        offsets = generator.permutation(np.arange(size, dtype=np.int64))
        selected.extend(int(offset) + base for offset in offsets[:required])
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20260816, 1, bp, 2]))
    )
    return tuple(
        selected[int(offset)]
        for offset in generator.permutation(np.arange(10000, dtype=np.int64))
    )


@pytest.mark.parametrize("bp", PREVALENCES)
def test_exact_measured_count_and_class_counts(manifests, bp):
    manifest = manifests[bp]
    assert type(manifest) is ReplayManifest
    assert manifest.prevalence_basis_points == bp
    assert type(manifest.records) is tuple
    assert len(manifest.records) == 10000
    assert sum(record.is_phishing for record in manifest.records) == bp


@pytest.mark.parametrize("bp", PREVALENCES)
def test_reference_order_and_canonical_json_hash_goldens(manifests, bp):
    manifest = manifests[bp]
    ordinals = tuple(
        int(record.record_id.rsplit(":", 1)[1], 16) for record in manifest.records
    )
    first_ordinals, order_hash, manifest_hash = GOLDENS[bp]
    assert ordinals == _reference_ordinals(bp)
    assert ordinals[:12] == first_ordinals
    assert (
        sha256(",".join(map(str, ordinals)).encode("ascii")).hexdigest() == order_hash
    )
    assert manifest.sha256 == manifest_hash
    payload = {
        "schema_version": 1,
        "algorithm_id": "replay-manifest-v1",
        "prevalence_basis_points": bp,
        "records": [asdict(record) for record in manifest.records],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(serialized).hexdigest() == manifest.sha256


@pytest.mark.parametrize("bp", PREVALENCES)
def test_unique_membership_and_warmup_prefix_are_reused(manifests, candidates, bp):
    manifest = manifests[bp]
    assert len({record.record_id for record in manifest.records}) == 10000
    assert len({record.canonical_url_sha256 for record in manifest.records}) == 10000
    assert set(manifest.records) <= set(candidates)
    assert manifest.warmup_records == manifest.records[:1000]
    assert len(manifest.warmup_records) == 1000
    assert all(
        warmup is measured
        for warmup, measured in zip(manifest.warmup_records, manifest.records)
    )
    # Warmup is a separate replay, not removed from the 10,000 measured rows.
    assert len(manifest.records) == 10000


def test_input_order_and_global_rng_state_do_not_change_results(candidates, manifests):
    shuffled = list(candidates)
    random.Random(123).shuffle(shuffled)
    original_order = tuple(shuffled)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(123456)
        np.random.seed(98765)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        assert build_manifest(shuffled, prevalence_basis_points=100) == manifests[100]
        assert random.getstate() == expected_python_state
        actual_numpy_state = np.random.get_state()
        assert actual_numpy_state[0] == expected_numpy_state[0]
        assert np.array_equal(actual_numpy_state[1], expected_numpy_state[1])
        assert actual_numpy_state[2:] == expected_numpy_state[2:]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert tuple(shuffled) == original_order


def test_concurrent_construction_order_is_irrelevant(candidates, manifests):
    order = (500, 10, 100, 10, 500, 100)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda bp: build_manifest(candidates, prevalence_basis_points=bp), order
            )
        )
    assert results == [manifests[bp] for bp in order]


def test_cross_prevalence_overlap_is_allowed(manifests):
    membership = [set(manifests[bp].records) for bp in PREVALENCES]
    assert membership[0] & membership[1] & membership[2]


def test_records_and_manifests_are_frozen(manifests):
    with pytest.raises(FrozenInstanceError):
        manifests[10].records[0].raw_url = "https://changed.example/"
    with pytest.raises(FrozenInstanceError):
        manifests[10].sha256 = "b" * 64


@pytest.mark.parametrize("bp", PREVALENCES)
def test_exact_capacity_is_accepted(candidates, bp):
    exact = candidates[: 10000 - bp] + candidates[11000 : 11000 + bp]
    manifest = build_manifest(exact, prevalence_basis_points=bp)
    assert set(manifest.records) == set(exact)


@pytest.mark.parametrize("bp", PREVALENCES)
@pytest.mark.parametrize("label", (0, 1))
def test_insufficient_stratum_reports_exact_capacity(candidates, bp, label):
    required = bp if label else 10000 - bp
    negatives = candidates[: 10000 - bp - (label == 0)]
    positives = candidates[11000 : 11000 + bp - (label == 1)]
    with pytest.raises(InsufficientStratum) as raised:
        build_manifest(negatives + positives, prevalence_basis_points=bp)
    error = raised.value
    assert isinstance(error, ValueError)
    assert error.prevalence_basis_points == bp
    assert error.label == label
    assert error.required == required
    assert error.available == required - 1
    assert "stratum" in str(error).lower()


def test_empty_input_reports_insufficient_negative_capacity():
    with pytest.raises(InsufficientStratum) as raised:
        build_manifest((), prevalence_basis_points=10)
    assert raised.value.label == 0
    assert raised.value.available == 0


@pytest.mark.parametrize(
    "bp", (0, 1, 50, 1000, 10000, True, False, 10.0, "10", None, np.int64(10))
)
def test_only_exact_integer_prevalences_are_accepted(bp):
    with pytest.raises(ValueError, match="prevalence_basis_points"):
        build_manifest((), prevalence_basis_points=bp)


@pytest.mark.parametrize("value", (None, "", b"", {}, set(), iter(()), range(0)))
def test_only_materialized_lists_and_tuples_are_accepted(value):
    with pytest.raises(TypeError, match="list or tuple"):
        build_manifest(value, prevalence_basis_points=10)


@pytest.mark.parametrize("value", (None, {}, (), "record", 1))
def test_only_manifest_records_are_accepted(value):
    with pytest.raises(TypeError, match="ManifestRecord"):
        build_manifest([value], prevalence_basis_points=10)


@pytest.mark.parametrize(
    "label", (True, False, 0.0, 1.0, "0", "1", -1, 2, None, np.int64(1))
)
def test_labels_are_exact_binary_integers(label):
    with pytest.raises(ValueError, match="is_phishing"):
        build_manifest(
            [replace(_record(1), is_phishing=label)], prevalence_basis_points=10
        )


@pytest.mark.parametrize(
    "record_id",
    (
        None,
        "",
        f"phiusiil-row-v2:{SOURCE_HASH}:0000000000000001",
        f"phiusiil-row-v1:{'A' * 64}:0000000000000001",
        f"phiusiil-row-v1:{SOURCE_HASH}:000000000000000A",
        f"phiusiil-row-v1:{SOURCE_HASH}:0000000000000000",
        f"phiusiil-row-v1:{SOURCE_HASH}:1",
        f"phiusiil-row-v1:{SOURCE_HASH}:10000000000000000",
        f"phiusiil-row-v1:{SOURCE_HASH}:0000000000000001\n",
    ),
)
def test_record_ids_match_the_one_based_preparation_contract(record_id):
    with pytest.raises(ValueError, match="record_id"):
        build_manifest(
            [replace(_record(1), record_id=record_id)], prevalence_basis_points=10
        )


@pytest.mark.parametrize(
    "split", (None, 1, "train", "validation", "GROUP_TEST", "group_test ")
)
def test_only_group_test_records_are_allowed(split):
    with pytest.raises(ValueError, match="split"):
        build_manifest([replace(_record(1), split=split)], prevalence_basis_points=10)


@pytest.mark.parametrize("digest", (None, "", "a" * 63, "g" * 64, "A" * 64, "b" * 64))
def test_canonical_hash_is_lowercase_sha256_and_matches_the_url(digest):
    with pytest.raises(ValueError, match="canonical_url_sha256"):
        build_manifest(
            [replace(_record(1), canonical_url_sha256=digest)],
            prevalence_basis_points=10,
        )


@pytest.mark.parametrize(
    "url",
    (
        None,
        "",
        "example.com/path",
        "ftp://example.com/",
        "https://example.com/%xy",
        " https://example.com/",
        "https://example.com/\n",
        "https://127.0.0.1/",
    ),
)
def test_raw_url_must_satisfy_existing_canonical_url_validation(url):
    with pytest.raises(ValueError):
        build_manifest([replace(_record(1), raw_url=url)], prevalence_basis_points=10)


@pytest.mark.parametrize(
    "domain",
    (
        None,
        "",
        "R00001.example",
        "r00001.example.",
        "r00001.example/path",
        "r00001.example:443",
        "example",
        "127.0.0.1",
        "a..example",
        "-a.example",
        "a_.example",
        "\u00e9.example",
        "unrelated.example",
        "notr00001.example",
        "xn--.example",
        "a" * 64 + ".example",
    ),
)
def test_domain_is_canonical_ascii_and_related_to_the_hostname(domain):
    with pytest.raises(ValueError, match="registrable_domain"):
        build_manifest(
            [replace(_record(1), registrable_domain=domain)], prevalence_basis_points=10
        )


def test_domain_can_be_a_parent_of_the_url_hostname(candidates):
    original = candidates[0]
    raw = "https://sub.r00001.example/path"
    updated = replace(
        original,
        raw_url=raw,
        canonical_url_sha256=sha256(raw.encode("utf-8")).hexdigest(),
    )
    manifest = build_manifest(
        [updated, *candidates[1:9990], *candidates[11000:11010]],
        prevalence_basis_points=10,
    )
    assert updated in manifest.records


def test_duplicate_record_ids_are_rejected():
    duplicate = replace(_record(2), record_id=_record(1).record_id)
    with pytest.raises(ValueError, match="duplicate record_id"):
        build_manifest([_record(1), duplicate], prevalence_basis_points=10)


def test_duplicate_canonical_keys_are_rejected():
    duplicate = replace(_record(1), record_id=_record(2).record_id)
    with pytest.raises(ValueError, match="duplicate canonical_url_sha256"):
        build_manifest([_record(1), duplicate], prevalence_basis_points=10)


def test_mixed_source_hashes_are_rejected():
    second = replace(
        _record(2), record_id=f"phiusiil-row-v1:{'b' * 64}:0000000000000002"
    )
    with pytest.raises(ValueError, match="source"):
        build_manifest([_record(1), second], prevalence_basis_points=10)


def test_invalid_surplus_candidate_is_not_silently_omitted(candidates):
    invalid = replace(_record(11701), is_phishing=True)
    with pytest.raises(ValueError, match="is_phishing"):
        build_manifest([*candidates, invalid], prevalence_basis_points=10)


def test_all_rows_are_validated_before_capacity_checks():
    invalid = replace(_record(2), split="train")
    with pytest.raises(ValueError, match="split") as raised:
        build_manifest([_record(1), invalid], prevalence_basis_points=10)
    assert not isinstance(raised.value, InsufficientStratum)


def test_hash_binds_raw_spelling_without_changing_membership(candidates, manifests):
    selected = manifests[10].records[0]
    updated = replace(
        selected, raw_url=selected.raw_url.replace("HTTPS://", "https://")
    )
    changed = [updated if record == selected else record for record in candidates]
    manifest = build_manifest(changed, prevalence_basis_points=10)
    assert [record.record_id for record in manifest.records] == [
        record.record_id for record in manifests[10].records
    ]
    assert manifest.sha256 != manifests[10].sha256
    assert manifest.records[0].raw_url == updated.raw_url


@pytest.mark.parametrize("bp", PREVALENCES)
def test_exactly_one_explicit_int64_permutation_per_purpose(
    candidates, monkeypatch, bp
):
    original_seed_sequence = np.random.SeedSequence
    original_generator = np.random.Generator
    entropies = []
    inputs = []

    def seed_sequence(entropy):
        entropies.append(entropy)
        return original_seed_sequence(entropy)

    class TracedGenerator:
        def __init__(self, bit_generator):
            assert type(bit_generator) is np.random.PCG64
            self.generator = original_generator(bit_generator)

        def permutation(self, values):
            inputs.append(values.copy())
            return self.generator.permutation(values)

    monkeypatch.setattr(np.random, "SeedSequence", seed_sequence)
    monkeypatch.setattr(np.random, "Generator", TracedGenerator)
    build_manifest(candidates, prevalence_basis_points=bp)
    assert entropies == [[20260816, 1, bp, purpose] for purpose in (0, 1, 2)]
    assert len(inputs) == 3
    for values, count in zip(inputs, (11000, 700, 10000)):
        assert values.dtype == np.dtype(np.int64)
        assert np.array_equal(values, np.arange(count, dtype=np.int64))


def test_validation_failure_never_initializes_a_generator(candidates, monkeypatch):
    def forbidden_seed(*args, **kwargs):
        pytest.fail("selection started before all candidates were validated")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden_seed)
    invalid = replace(_record(11701), split="train")
    with pytest.raises(ValueError, match="split"):
        build_manifest([*candidates, invalid], prevalence_basis_points=10)


@pytest.mark.parametrize("label", (0, 1))
def test_insufficient_capacity_never_initializes_a_generator(
    candidates, monkeypatch, label
):
    def forbidden_seed(*args, **kwargs):
        pytest.fail("selection started before both capacity checks completed")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden_seed)
    negatives = candidates[: 9990 - (label == 0)]
    positives = candidates[11000 : 11010 - (label == 1)]
    with pytest.raises(InsufficientStratum) as raised:
        build_manifest(negatives + positives, prevalence_basis_points=10)
    assert raised.value.label == label


def test_maximum_preparation_ordinal_is_valid():
    record = replace(
        _record(1), record_id=f"phiusiil-row-v1:{SOURCE_HASH}:ffffffffffffffff"
    )
    with pytest.raises(InsufficientStratum):
        build_manifest([record], prevalence_basis_points=10)


def test_raw_url_must_encode_as_utf8():
    with pytest.raises(ValueError, match="raw_url"):
        build_manifest(
            [replace(_record(1), raw_url="https://r00001.example/\ud800")],
            prevalence_basis_points=10,
        )
