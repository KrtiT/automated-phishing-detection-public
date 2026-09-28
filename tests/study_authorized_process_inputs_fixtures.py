"""Invented original sources with genuine whole-study necessary capacity."""

import json
from dataclasses import replace
from hashlib import sha256

from phishvn_source_fixtures import bundle, members, record

from automated_phishing_detection import phiusiil, protocol_preflight


def internal_inputs(binding, paths):
    source = json.loads((binding.root / "data/sources.json").read_bytes())
    suffix = protocol_preflight.parse_suffix_rules(paths.suffix_rules.read_text())
    original = phiusiil.resolve_rows(
        phiusiil._parse_csv_rows(paths.source_csv.read_bytes()),
        csv_sha256=source["phiusiil"]["csv_sha256"],
        suffix_rules=suffix,
    )
    assigned = phiusiil.assign_splits(original.retained)
    content = _capacity_csv(assigned)
    paths.source_csv.write_bytes(content)
    source["phiusiil"]["csv_sha256"] = sha256(content).hexdigest()
    source_bytes = json.dumps(source).encode()
    resolution = phiusiil.resolve_rows(
        phiusiil._parse_csv_rows(content),
        csv_sha256=source["phiusiil"]["csv_sha256"],
        suffix_rules=suffix,
    )
    assigned = phiusiil.assign_splits(resolution.retained)
    return _public_records(binding, source, source_bytes, resolution, assigned)


def _capacity_csv(assigned):
    domains = sorted(
        {row.registrable_domain for row in assigned if row.split == "group_test"}
    )
    retained = [row for row in assigned if row.split != "group_test"]
    rows = [(row.raw_url, 1 - row.is_phishing) for row in retained]
    rows += [
        (
            f"https://host{index}.{domains[index % len(domains)]}/capacity",
            int(index < 9990),
        )
        for index in range(10490)
    ]
    return ("URL,label\n" + "".join(f"{url},{label}\n" for url, label in rows)).encode()


def _public_records(binding, source, source_bytes, resolution, assigned):
    outputs = phiusiil._private_output_contents(assigned, resolution)
    hashes = {name: sha256(content).hexdigest() for name, content in outputs.items()}
    sums = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()
    hashes["SHA256SUMS"] = sha256(sums).hexdigest()
    report = phiusiil._build_summary(
        assigned, resolution, source, sha256(source_bytes).hexdigest(), hashes
    )
    contents = {
        "data/sources.json": source_bytes,
        "reports/phiusiil-preparation-summary.json": json.dumps(report).encode(),
    }
    for name, content in contents.items():
        (binding.root / name).write_bytes(content)
    return replace(
        binding,
        source_hashes=tuple(
            (name, sha256(content).hexdigest()) for name, content in contents.items()
        ),
    )


def external_inputs(path):
    rows = [
        record(f"gold{index}", url=f"https://gold{index}.com/a") for index in range(998)
    ]
    rows += [
        record(
            "certified",
            url="https://certified.com/a",
            source="tinnhiem_web",
            label="benign",
            tier="gold",
        ),
        record(
            "tranco",
            url="https://tranco.com/a",
            source="tranco",
            label="benign",
            tier="silver",
        ),
    ]
    archive = bundle(members(rows))
    path.write_bytes(archive.content)
    return archive.pins
