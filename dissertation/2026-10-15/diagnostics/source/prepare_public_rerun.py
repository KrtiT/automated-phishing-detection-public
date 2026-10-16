"""Prepare the 99-permutation experiment from the original licensed partitions.

This adapter does not authenticate the five earlier fits. It reconstructs and
checks the exact numerical inputs used by the frozen permutation producer.
It never visits URL destinations or reads external/test partitions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))
    from automated_phishing_detection.gmm_monitor import _require_runtime
    from automated_phishing_detection.url_features import FEATURE_NAMES, extract_url_features

    _require_runtime()
    original = args.diagnostics.resolve()
    protocol = json.loads((original / "PROTOCOL.json").read_text())
    expected = json.loads((original / "preparation-receipt.json").read_text())
    assert not args.output.exists(), "Use a new output directory; never overwrite a run"
    partitions = []
    for name in ("train", "validation"):
        raw = (args.data / f"{name}.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == protocol[name + "_sha256"], name
        rows = [json.loads(line) for line in raw.splitlines()]
        assert all(r["split"] == name for r in rows)
        partitions.append(rows)
    train, validation = partitions
    assert (len(train), len(validation)) == (166248, 32695)
    for field in ("record_id", "registrable_domain", "canonical_url_sha256"):
        assert not ({r[field] for r in train} & {r[field] for r in validation})
    assert list(FEATURE_NAMES) == json.loads((original / "feature-order.json").read_text())
    y, v = [np.array([r["is_phishing"] for r in rows], dtype=np.int64) for rows in partitions]
    x, z = [np.array([extract_url_features(r["raw_url"]) for r in rows], dtype=np.float64) for rows in partitions]
    scaler = StandardScaler().fit(x)
    arrays = {"x": scaler.transform(x), "z": scaler.transform(z), "y": y, "v": v}
    keys = {"x": "training_features", "z": "validation_features", "y": "training_labels", "v": "validation_labels"}
    for name, a in arrays.items():
        dtype = "<i8" if name in ("y", "v") else "<f8"
        assert hashlib.sha256(np.asarray(a, dtype=dtype).tobytes()).hexdigest() == expected[keys[name] + "_sha256"], name
    counts = Counter(r["registrable_domain"] for r in validation)
    arrays["domain_weights"] = np.array([1 / counts[r["registrable_domain"]] for r in validation])
    output = args.output.resolve()
    (output / "private-cache").mkdir(parents=True)
    for name, a in arrays.items():
        np.save(output / "private-cache" / f"{name}.npy", a)
    for name in ("PROTOCOL.json", "feature-order.json"):
        shutil.copy2(original / name, output / name)
    shutil.copy2(original / "source/diagnose_controls.py", output / "diagnose_controls.py")
    receipt = {"status": "exact_inputs_reconstructed", "checks": keys,
               "expected_input_receipt_sha256": hashlib.sha256((original / "preparation-receipt.json").read_bytes()).hexdigest(),
               "authenticates_original_five_fits": False}
    (output / "adapter-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
