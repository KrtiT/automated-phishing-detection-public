"""Necessary population capacity does not establish research estimability."""

import json

from study_feasibility_fixtures import (
    api,
    assess,
    external_rows,
    internal_rows,
    shortages,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes


def test_projection_is_closed_canonical_and_has_no_authority_or_decision():
    content = api().assess_preparation_feasibility(internal_rows(), external_rows())
    result = json.loads(content)
    assert canonical_bytes(result) == content
    assert content.isascii() and content.endswith(b"\n")
    assert set(result) == {"schema_version", "protocol", "scope", "counts", "shortages"}
    assert result["schema_version"] == 1
    assert result["protocol"] == "study-preparation-feasibility-v1"
    assert result["scope"] == "necessary_population_capacity_only"
    assert set(result["counts"]) == {"internal", "external"}
    assert all(
        set(item) == {"requirement", "category", "required", "available"}
        for item in result["shortages"]
    )


def test_counts_are_derived_from_the_prepared_rows():
    result = assess()
    assert result["counts"]["internal"] == {
        "retained_rows": 3,
        "positive_rows": 2,
        "negative_rows": 1,
        "positive_domains": 2,
    }
    assert result["counts"]["external"] == {
        "input_test_rows": 6,
        "quarantined_test_rows": 0,
        "retained_test_rows": 6,
        "gold_rows": 2,
        "gold_positive_domains": 2,
        "certified_rows": 1,
        "tranco_rows": 1,
        "secondary_rows": 2,
        "complete_windows": 0,
        "secondary_positive_strata": {
            "ncsc_silver": {"rows": 1, "domains": 1},
            "chongluadao_openphish_bronze": {"rows": 1, "domains": 1},
        },
    }


def test_valid_empty_populations_produce_shortages_not_synthetic_results():
    result = assess((), external_rows(()))
    assert result["counts"]["internal"] == {
        "retained_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "positive_domains": 0,
    }
    assert len(result["shortages"]) == 13
    assert all(item["available"] == 0 for item in result["shortages"])
    assert {item["category"] for item in result["shortages"]} == {
        "primary",
        "sensitivity",
        "descriptive",
    }


def test_quarantine_zero_is_not_a_claim_of_publisher_emptiness():
    result = assess((), external_rows((), quarantine_count=3))
    external = result["counts"]["external"]
    assert external["input_test_rows"] == external["quarantined_test_rows"] == 3
    assert external["retained_test_rows"] == 0
    assert len(result["shortages"]) == 13


def test_controls_and_secondary_positives_do_not_fill_primary_roles():
    result = assess(external=external_rows(("silver", "bronze", "tranco")))
    external = result["counts"]["external"]
    assert external["gold_rows"] == external["certified_rows"] == 0
    assert external["secondary_rows"] == 2 and external["tranco_rows"] == 1
    assert "gold_positive_domains" in shortages(result)
    assert "certified_rows" in shortages(result)
    assert "tranco_rows" not in shortages(result)


def test_absent_secondary_strata_are_explicit_without_new_hold_thresholds():
    result = assess(external=external_rows(("gold", "gold", "certified", "tranco")))
    assert result["counts"]["external"]["secondary_positive_strata"] == {
        "ncsc_silver": {"rows": 0, "domains": 0},
        "chongluadao_openphish_bronze": {"rows": 0, "domains": 0},
    }
    assert not any("secondary" in name for name in shortages(result))


def test_full_necessary_capacity_does_not_claim_guaranteed_estimability():
    roles = ("gold", "gold", "certified", "tranco") + ("silver",) * 996
    result = assess(internal_rows(9990, 500, 2), external_rows(roles))
    assert result["shortages"] == []
    assert (
        "status" not in result and "hypotheses" not in result and "ready" not in result
    )


def test_aggregate_projection_contains_no_urls_ids_domains_or_artifact_claims():
    content = api().assess_preparation_feasibility(internal_rows(), external_rows())
    for private in (
        b"https://",
        b"phiusiil-row",
        b"external-",
        b".test",
        b"artifact",
        b"sha256",
    ):
        assert private not in content


def test_internal_input_order_does_not_change_population_capacity():
    assert assess(tuple(reversed(internal_rows()))) == assess()


def test_shortage_identity_and_order_are_fixed_not_result_selected():
    assert list(shortages(assess((), external_rows(())))) == [
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
    ]


def test_no_new_power_or_precision_requirement_is_added_to_rate_capacity():
    result = assess()
    assert not set(shortages(result)) & {
        "internal_negative_rows",
        "certified_rows",
        "tranco_rows",
        "internal_positive_domains",
        "gold_positive_domains",
    }
