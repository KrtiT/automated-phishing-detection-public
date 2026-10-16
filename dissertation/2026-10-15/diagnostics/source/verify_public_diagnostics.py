"""Recalculate published aggregate statistics; no private inputs or new fitting.

The separate retained-data verifier reconstructs predictions and checks metrics.
This public check verifies completeness, identities and aggregation only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(root):
    load = lambda n: json.loads((root / n).read_text())
    protocol, summary = load("PROTOCOL.json"), load("diagnostic-summary.json")
    expected = set(range(1000, 1099))
    paths = list((root / "additional-fits").glob("*-result.json"))
    fits = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["seed"])
    assert len(fits) == 99 and {r["seed"] for r in fits} == expected
    assert {p.name for p in paths} == {f"{s}-result.json" for s in expected}
    for r in fits:
        assert r["status"] == "completed" and not r["warnings"]
        assert 0 < r["iterations"] < protocol["classifier"]["max_iter"]
        assert len(r["coefficients"]) == 25
        assert r["parameters"] == protocol["classifier"]
        assert r["protocol_sha256"] == sha(root / "PROTOCOL.json")
        assert r["analysis_source_sha256"] == sha(root / "source/diagnose_controls.py")
    with (root / "permutation-summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["seed"]) for r in rows] == list(range(1000, 1099))
    for r, row in zip(fits, rows):
        for key in ("auc", "ap", "without_https_auc", "randomized_validation_auc", "domain_equal_weighted_auc"):
            assert float(row[key]) == r[key]
    for key in ("auc", "ap", "without_https_auc", "randomized_validation_auc", "domain_equal_weighted_auc"):
        a = np.array([r[key] for r in fits])
        calculated = {"n": len(a), "mean": a.mean(), "sd": a.std(ddof=1), "min": a.min(), "max": a.max(),
                      "q025": np.quantile(a, .025), "median": np.median(a), "q975": np.quantile(a, .975)}
        assert all(abs(value - summary[key][name]) < 1e-12 for name, value in calculated.items()), key
    r = spearmanr([r["coefficients"][20] for r in fits], [r["auc"] for r in fits]).statistic
    assert abs(r - summary["auc_https_coefficient_spearman"]) < 1e-12
    assert sum(r["auc"] >= summary["original_max_auc"] for r in fits) == summary["at_or_above_original_max_auc"]
    for key, field in (("auc", "mean_absolute_auc_departure"), ("without_https_auc", "mean_absolute_auc_departure_without_https")):
        assert abs(np.mean([abs(r[key] - .5) for r in fits]) - summary[field]) < 1e-12
    strict = [load(f"convergence/{s}-result.json") for s in range(42, 47)]
    for r in strict:
        assert not r["warnings"] and r["iterations"] < 20000
        assert r["protocol_sha256"] == sha(root / "CONVERGENCE_PROTOCOL.json")
        assert r["source_sha256"] == sha(root / "source/check_convergence.py")
        assert abs(r["auc_after"] - r["auc_before"] - r["auc_difference"]) < 1e-15
    assert max(abs(r["auc_difference"]) for r in strict) == summary["convergence"]["max_absolute_auc_change"]
    assert max(r["kkt"]["max_residual_mean_loss"] for r in strict) == summary["convergence"]["max_mean_loss_kkt_residual"]
    return {"status": "verified", "additional_fits": 99, "strict_continuations": 5,
            "scope": "Completeness, source/protocol hashes and aggregate arithmetic; no independent validation of private rows."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    print(json.dumps(check(parser.parse_args().diagnostics.resolve()), indent=2))
