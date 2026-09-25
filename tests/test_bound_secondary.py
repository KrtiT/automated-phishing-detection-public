"""Accepted secondary artifacts bind without research-source access or fitting."""

import importlib
import inspect
import json
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from automated_phishing_detection.transformer_inference import LoadedTransformerCascade

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def secondary():
    return importlib.import_module("automated_phishing_detection.bound_secondary")


TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
SEEDS = (42, 43, 44, 45, 46)


def _canonical(value):
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _digest(value):
    return sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _threshold(value):
    return {"status": "selected", "threshold": value}


@pytest.fixture
def binding_fixture(tmp_path, secondary, monkeypatch):
    root = tmp_path / "checkout"
    reports = root / "reports"
    reports.mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()

    tabular_bytes = {
        name: f"invented-{name}\n".encode("ascii") for name in TABULAR_NAMES
    }
    seed_bytes = {seed: f"invented-seed-{seed}\n".encode("ascii") for seed in SEEDS[1:]}
    tabular_thresholds = {
        name: 0.11 + index / 100 for index, name in enumerate(TABULAR_NAMES)
    }
    seed_thresholds = {seed: 0.41 + index / 100 for index, seed in enumerate(SEEDS)}
    half_widths = {seed: index / 100 for index, seed in enumerate(SEEDS)}

    retained_members = [
        {
            "member": "drift",
            "public_summary_sha256": "0" * 64,
            "summary": {
                "status": "development_member_completed",
                "member": "drift",
                "private_sha256": {
                    "training-reference.json": "1" * 64,
                    "validation-audit.json": "2" * 64,
                },
                "result": {"analysis_role": "secondary_descriptive_only"},
            },
        }
    ]
    retained_members[0]["public_summary_sha256"] = _digest(
        retained_members[0]["summary"]
    )
    for name in TABULAR_NAMES[:-1]:
        kind = "permutation" if name.startswith("permutation_") else "formatting"
        seed = int(name.rsplit("_", 1)[1]) if kind == "permutation" else 42
        summary = {
            "status": "development_member_completed",
            "member": name,
            "private_sha256": {"model.json": sha256(tabular_bytes[name]).hexdigest()},
            "result": {
                "analysis_role": "descriptive_secondary_not_primary",
                "model_kind": kind,
                "seed": seed,
                "validation_threshold": _threshold(tabular_thresholds[name]),
            },
        }
        retained_members.append(
            {
                "member": name,
                "public_summary_sha256": _digest(summary),
                "summary": summary,
            }
        )
    rf_summary = {
        "schema_version": 1,
        "stage": "random_forest",
        "status": "development_correction_stage_completed",
        "private_sha256": {
            "model.json": sha256(tabular_bytes["random_forest"]).hexdigest()
        },
        "result": {
            "analysis_role": "descriptive_secondary_not_primary",
            "model_kind": "random_forest",
            "seed": 42,
            "validation_threshold": _threshold(tabular_thresholds["random_forest"]),
        },
    }
    tabular_completion = {
        "status": "completed_secondary_development_correction",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "new_fits": 1,
        "retries": 0,
        "worker_exit_codes": {"retained_audit": 0, "random_forest": 0},
        "retained_audit": {
            "stage": "retained_audit",
            "status": "development_correction_stage_completed",
            "result": {
                "status": "retained_development_members_audited",
                "analysis_stage": "development_validation_only",
                "protected_evaluation_authorized": False,
                "fits": 0,
                "members": retained_members,
            },
        },
        "corrected_random_forest": rf_summary,
    }
    tabular_report = {
        "status": "accepted_development_evidence",
        "execution_observation": {
            "parent_exit_code": 0,
            "worker_exit_codes": {"retained_audit": 0, "random_forest": 0},
        },
        "completion_summary_sha256": _digest(tabular_completion),
        "completion": tabular_completion,
        "receipt_sha256": {"random_forest.json": _digest(rf_summary)},
    }

    seed_members = []
    primary_weight_bytes = b"invented-primary-seed-42\n"
    for seed in SEEDS:
        result = {
            "seed": seed,
            "new_fit": seed != 42,
            "primary_artifacts_changed": False,
            "weights_sha256": sha256(
                primary_weight_bytes if seed == 42 else seed_bytes[seed]
            ).hexdigest(),
            "calibration": {
                "stage1": {"threshold": 0.5},
                "transformer_threshold": _threshold(seed_thresholds[seed]),
                "cascade_band": {
                    "status": "selected",
                    "accepted_cascade": True,
                    "half_width": half_widths[seed],
                },
            },
        }
        stage = "seed_42_calibration" if seed == 42 else f"seed_{seed}"
        summary = {
            "stage": stage,
            "status": "completed_secondary_seed_probe_stage",
            "result": result,
        }
        seed_members.append(
            {"stage": stage, "summary_sha256": _digest(summary), "summary": summary}
        )
    seed_completion = {
        "status": "completed_secondary_seed_probe_correction",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "new_fits": 0,
        "seed_stage_executions": 0,
        "retries": 0,
        "worker_exit_codes": {"retained_seed_audit": 0, "probes": 0},
        "retained_seed_audit": {
            "status": "completed_secondary_seed_probe_correction_stage",
            "result": {
                "status": "retained_seed_stages_audited",
                "analysis_stage": "development_validation_only",
                "protected_evaluation_authorized": False,
                "primary_artifacts_changed": False,
                "fits": 0,
                "seed_selection_performed": False,
                "members": seed_members,
            },
        },
    }
    seed_report = {
        "status": "accepted_development_evidence",
        "execution_observation": {
            "parent_exit_code": 0,
            "worker_exit_codes": {"retained_seed_audit": 0, "probes": 0},
        },
        "completion_summary_sha256": _digest(seed_completion),
        "completion": seed_completion,
    }
    report_bytes = {
        "tabular": _canonical(tabular_report),
        "seeds": _canonical(seed_report),
    }
    for role, content in report_bytes.items():
        relative = secondary.PUBLIC_REPORTS[role][0]
        (root / relative).write_bytes(content)
    monkeypatch.setattr(
        secondary,
        "PUBLIC_REPORTS",
        {
            role: (secondary.PUBLIC_REPORTS[role][0], sha256(content).hexdigest())
            for role, content in report_bytes.items()
        },
    )

    path_values = {}
    for name, content in tabular_bytes.items():
        path = private / f"{name}.json"
        path.write_bytes(content)
        path_values[name] = path
    for seed, content in seed_bytes.items():
        path = private / f"seed-{seed}.npz"
        path.write_bytes(content)
        path_values[f"seed_{seed}_weights"] = path
    paths = secondary.SecondaryArtifactPaths(
        *(path_values[field.name] for field in fields(secondary.SecondaryArtifactPaths))
    )
    vocabulary_bytes = b'{"invented":"vocabulary"}'
    primary = LoadedTransformerCascade(
        stage1_model=SimpleNamespace(),
        vocabulary=SimpleNamespace(to_json=lambda: vocabulary_bytes.decode("utf-8")),
        stage1_threshold=0.5,
        transformer_threshold=0.4100001,
        half_width=0.125,
        device=torch.device("mps"),
        public_summary_sha256="3" * 64,
        artifact_hashes=tuple(
            sorted(
                {
                    "transformer-weights.npz": sha256(primary_weight_bytes).hexdigest(),
                    "vocabulary.json": sha256(vocabulary_bytes).hexdigest(),
                }.items()
            )
        ),
        _model=object(),
    )
    loaded = []

    def load_model(content):
        loaded.append(content)
        return secondary.SecondaryModel(content)

    monkeypatch.setattr(secondary, "load_secondary_model_bytes", load_model)
    return SimpleNamespace(
        root=root,
        paths=paths,
        primary=primary,
        tabular_bytes=tabular_bytes,
        seed_bytes=seed_bytes,
        tabular_thresholds=tabular_thresholds,
        seed_thresholds=seed_thresholds,
        half_widths=half_widths,
        loaded=loaded,
    )


def test_public_api_pins_accepted_reports_and_explicit_private_paths(secondary):
    assert secondary.PUBLIC_REPORTS == {
        "tabular": (
            "reports/secondary-development-correction-v2-summary.json",
            "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d",
        ),
        "seeds": (
            "reports/secondary-seed-probe-correction-v1-summary.json",
            "d63a85792088871cfb667e7e5cbe86c6148dc3a6c7430ca2e4db29d788ab8e23",
        ),
    }
    assert [field.name for field in fields(secondary.SecondaryArtifactPaths)] == [
        "formatting",
        "permutation_42",
        "permutation_43",
        "permutation_44",
        "permutation_45",
        "permutation_46",
        "random_forest",
        "seed_43_weights",
        "seed_44_weights",
        "seed_45_weights",
        "seed_46_weights",
    ]
    assert list(inspect.signature(secondary.load_bound_secondary).parameters) == [
        "root",
        "paths",
        "primary",
    ]
    assert list(inspect.signature(secondary.score_bound_secondary).parameters) == [
        "bound",
        "raw_urls",
        "stage1_probabilities",
        "seed_42_probabilities",
    ]


def test_binding_authenticates_reports_then_exact_artifacts(
    secondary, binding_fixture, monkeypatch
):
    fixture = binding_fixture
    original = secondary.fixed_cascade._read_regular_file
    reads = []

    def read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(secondary.fixed_cascade, "_read_regular_file", read)
    bound = secondary.load_bound_secondary(fixture.root, fixture.paths, fixture.primary)
    assert reads[:2] == [
        fixture.root / secondary.PUBLIC_REPORTS["tabular"][0],
        fixture.root / secondary.PUBLIC_REPORTS["seeds"][0],
    ]
    assert reads[2:] == [
        *(getattr(fixture.paths, name) for name in TABULAR_NAMES),
        *(getattr(fixture.paths, f"seed_{seed}_weights") for seed in SEEDS[1:]),
    ]
    assert fixture.loaded == [fixture.tabular_bytes[name] for name in TABULAR_NAMES]
    assert [member.name for member in bound.tabular] == list(TABULAR_NAMES)
    assert [member.threshold for member in bound.tabular] == [
        fixture.tabular_thresholds[name] for name in TABULAR_NAMES
    ]
    assert [member.seed for member in bound.seeds] == list(SEEDS)
    assert [member.transformer_threshold for member in bound.seeds] == [
        fixture.seed_thresholds[seed] for seed in SEEDS
    ]
    assert [member.half_width for member in bound.seeds] == [
        fixture.half_widths[seed] for seed in SEEDS
    ]
    assert bound.seeds[0].reuses_primary is True
    assert bound.seeds[0]._weights_bytes is None
    assert [member._weights_bytes for member in bound.seeds[1:]] == [
        fixture.seed_bytes[seed] for seed in SEEDS[1:]
    ]
    assert bound.stage1_threshold == fixture.primary.stage1_threshold
    assert bound.vocabulary_bytes == fixture.primary.vocabulary.to_json().encode()
    assert bound.device_type == "mps"


def test_current_accepted_reports_expose_the_frozen_inventory_and_points(secondary):
    reports = secondary._read_reports(ROOT)
    primary_summary = json.loads(
        (ROOT / "reports/rq1-transformer-cascade-v2-summary.json").read_bytes()
    )
    baseline_summary = json.loads(
        (ROOT / "reports/rq1-baseline-v2-summary.json").read_bytes()
    )
    primary = LoadedTransformerCascade(
        stage1_model=SimpleNamespace(),
        vocabulary=SimpleNamespace(to_json=lambda: "unused"),
        stage1_threshold=baseline_summary["models"]["Logistic-L1"][
            "validation_threshold"
        ]["threshold"],
        transformer_threshold=0.0,
        half_width=0.0,
        device=torch.device("mps"),
        public_summary_sha256="0" * 64,
        artifact_hashes=tuple(sorted(primary_summary["artifact_hashes"].items())),
        _model=object(),
    )
    tabular = secondary._tabular_points(reports["tabular"])
    seeds = secondary._seed_points(reports["seeds"], primary)
    assert [point[0] for point in tabular] == list(TABULAR_NAMES)
    assert [point[2] for point in tabular] == [
        0.999925571711023,
        0.5042071644591366,
        0.5031540848581733,
        0.5058518481954868,
        0.5087053417015189,
        0.5014918824187501,
        0.2,
    ]
    assert [point[0] for point in seeds] == list(SEEDS)
    assert [point[2] for point in seeds] == [
        0.03397693857550621,
        0.17120759189128876,
        0.03427749499678612,
        0.05094735324382782,
        0.02175699733197689,
    ]
    assert [point[3] for point in seeds] == [0.0] * 5
    assert seeds[0][1] == primary_summary["artifact_hashes"]["transformer-weights.npz"]


def _scoring_bound(secondary):
    tabular = tuple(
        secondary.BoundTabular(
            name,
            secondary.SecondaryModel(name.encode("ascii")),
            0.5,
            sha256(name.encode("ascii")).hexdigest(),
        )
        for name in TABULAR_NAMES
    )
    seeds = tuple(
        secondary.BoundSeed(
            seed,
            {42: 0.4, 43: 0.6, 44: 0.5, 45: 0.3, 46: 0.8}[seed],
            {42: 0.0, 43: 0.25, 44: 0.1, 45: 0.2, 46: 0.3}[seed],
            sha256(f"weights-{seed}".encode()).hexdigest(),
            seed == 42,
            None if seed == 42 else f"weights-{seed}".encode(),
        )
        for seed in SEEDS
    )
    return secondary.BoundSecondary(
        tabular,
        seeds,
        0.5,
        b'{"invented":"vocabulary"}',
        "mps",
        (("tabular", "1" * 64), ("seeds", "2" * 64)),
    )


def test_scoring_preserves_families_reuses_seed_42_and_loads_other_seeds_in_order(
    secondary, monkeypatch
):
    bound = _scoring_bound(secondary)
    urls = ("https://first.example/path", "https://second.example/path")
    events = []
    tabular_values = {
        name: (0.25 + index / 100, 0.75 - index / 100)
        for index, name in enumerate(TABULAR_NAMES)
    }

    def tabular_score(model, supplied):
        name = model.artifact_bytes.decode("ascii")
        events.append(("tabular", name, supplied))
        return tabular_values[name]

    def load_seed(weights, vocabulary, *, seed, device):
        events.append(("load", seed))
        assert weights == f"weights-{seed}".encode()
        assert vocabulary == bound.vocabulary_bytes
        assert device == torch.device("mps")
        return SimpleNamespace(
            seed=seed,
            weights_sha256=sha256(weights).hexdigest(),
            vocabulary_sha256=sha256(vocabulary).hexdigest(),
            device=device,
        )

    seed_values = {
        43: (0.7, 0.2),
        44: (0.4, 0.9),
        45: (0.1, 0.35),
        46: (0.85, 0.1),
    }

    def score_seed(loaded, supplied):
        events.append(("score", loaded.seed, supplied))
        return seed_values[loaded.seed]

    monkeypatch.setattr(
        secondary.SecondaryModel,
        "score_urls_singleton_ordered",
        tabular_score,
    )
    monkeypatch.setattr(
        secondary.secondary_transformer,
        "load_secondary_transformer_bytes",
        load_seed,
    )
    monkeypatch.setattr(
        secondary.secondary_transformer,
        "score_secondary_transformer_urls",
        score_seed,
    )
    result = secondary.score_bound_secondary(
        bound,
        urls,
        (0.5, 0.7),
        (0.41, 0.1),
    )
    assert len(result.rows) == 2
    assert [[score.name for score in row.tabular] for row in result.rows] == [
        list(TABULAR_NAMES),
        list(TABULAR_NAMES),
    ]
    assert [score.decision for score in result.rows[0].tabular] == [0] * 7
    assert [score.decision for score in result.rows[1].tabular] == [1] * 7
    assert [[score.seed for score in row.seeds] for row in result.rows] == [
        list(SEEDS),
        list(SEEDS),
    ]
    seed_42 = result.rows[0].seeds[0]
    assert seed_42.transformer_probability == 0.41
    assert seed_42.transformer_decision == 1
    assert seed_42.band_selected is True
    assert seed_42.cascade_probability == 0.41
    assert seed_42.cascade_decision == 1
    assert result.rows[1].seeds[0].band_selected is False
    assert result.rows[1].seeds[0].cascade_probability == 0.7
    assert result.counts == secondary.SecondaryInferenceCounts(
        tuple((name, 2) for name in TABULAR_NAMES),
        ((42, 0), (43, 2), (44, 2), (45, 2), (46, 2)),
        2,
    )
    assert [event[:2] for event in events if event[0] in {"load", "score"}] == [
        ("load", 43),
        ("score", 43),
        ("load", 44),
        ("score", 44),
        ("load", 45),
        ("score", 45),
        ("load", 46),
        ("score", 46),
    ]
    assert not any(event[:2] == ("load", 42) for event in events)


@pytest.mark.parametrize(
    "stage1,seed_42",
    [
        ((0.5,), (0.4, 0.6)),
        ((0.5, float("nan")), (0.4, 0.6)),
        ((0.5, 0.6), (0.4, True)),
    ],
)
def test_invalid_primary_score_vectors_fail_before_secondary_scoring(
    secondary, monkeypatch, stage1, seed_42
):
    bound = _scoring_bound(secondary)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid aligned scores reached secondary model execution")

    monkeypatch.setattr(
        secondary.SecondaryModel,
        "score_urls_singleton_ordered",
        forbidden,
    )
    monkeypatch.setattr(
        secondary.secondary_transformer,
        "load_secondary_transformer_bytes",
        forbidden,
    )
    with pytest.raises(secondary.BoundSecondaryError):
        secondary.score_bound_secondary(
            bound,
            ("https://first.example", "https://second.example"),
            stage1,
            seed_42,
        )


def test_public_report_hash_failure_precedes_every_private_read(
    secondary, binding_fixture, monkeypatch
):
    fixture = binding_fixture
    tabular_path = fixture.root / secondary.PUBLIC_REPORTS["tabular"][0]
    tabular_path.write_bytes(b"{}")
    original = secondary.fixed_cascade._read_regular_file
    reads = []

    def read(path):
        path = Path(path)
        assert path.is_relative_to(fixture.root), "private read preceded report auth"
        reads.append(path)
        return original(path)

    monkeypatch.setattr(secondary.fixed_cascade, "_read_regular_file", read)
    with pytest.raises(secondary.BoundSecondaryError, match="report SHA-256"):
        secondary.load_bound_secondary(fixture.root, fixture.paths, fixture.primary)
    assert reads == [tabular_path]
    assert fixture.loaded == []


@pytest.mark.parametrize("mutation", ["seed_status", "tabular_order"])
def test_authenticated_report_semantics_fail_before_private_reads(
    secondary, binding_fixture, monkeypatch, mutation
):
    fixture = binding_fixture
    role = "seeds" if mutation == "seed_status" else "tabular"
    path = fixture.root / secondary.PUBLIC_REPORTS[role][0]
    report = json.loads(path.read_bytes())
    if mutation == "seed_status":
        report["status"] = "preliminary"
    else:
        report["completion"]["retained_audit"]["result"]["members"].reverse()
        report["completion_summary_sha256"] = _digest(report["completion"])
    content = _canonical(report)
    path.write_bytes(content)
    monkeypatch.setitem(
        secondary.PUBLIC_REPORTS,
        role,
        (secondary.PUBLIC_REPORTS[role][0], sha256(content).hexdigest()),
    )
    original = secondary.fixed_cascade._read_regular_file
    reads = []

    def read(candidate):
        candidate = Path(candidate)
        assert candidate.is_relative_to(fixture.root), (
            "private read preceded report validation"
        )
        reads.append(candidate)
        return original(candidate)

    monkeypatch.setattr(secondary.fixed_cascade, "_read_regular_file", read)
    with pytest.raises(secondary.BoundSecondaryError):
        secondary.load_bound_secondary(fixture.root, fixture.paths, fixture.primary)
    assert reads == [
        fixture.root / secondary.PUBLIC_REPORTS["tabular"][0],
        fixture.root / secondary.PUBLIC_REPORTS["seeds"][0],
    ]
    assert fixture.loaded == []


def test_private_artifact_hash_checked_before_decoder(
    secondary, binding_fixture, monkeypatch
):
    fixture = binding_fixture
    fixture.paths.formatting.write_bytes(b"mutated invented model")
    with pytest.raises(secondary.BoundSecondaryError, match="formatting.*SHA-256"):
        secondary.load_bound_secondary(fixture.root, fixture.paths, fixture.primary)
    assert fixture.loaded == []


def test_seed_42_is_cross_bound_to_primary_weights_before_private_artifacts(
    secondary, binding_fixture, monkeypatch
):
    fixture = binding_fixture
    hashes = dict(fixture.primary.artifact_hashes)
    hashes["transformer-weights.npz"] = "0" * 64
    primary = replace(fixture.primary, artifact_hashes=tuple(sorted(hashes.items())))
    original = secondary.fixed_cascade._read_regular_file
    reads = []

    def read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(secondary.fixed_cascade, "_read_regular_file", read)
    with pytest.raises(secondary.BoundSecondaryError, match="seed 42 weights"):
        secondary.load_bound_secondary(fixture.root, fixture.paths, primary)
    assert reads == [
        fixture.root / secondary.PUBLIC_REPORTS["tabular"][0],
        fixture.root / secondary.PUBLIC_REPORTS["seeds"][0],
    ]
    assert fixture.loaded == []
