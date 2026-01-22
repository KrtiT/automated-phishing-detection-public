#!/usr/bin/env python3
"""Replays the held-out test set against the FastAPI service to capture latency."""

import argparse
import asyncio
import json
import time
from itertools import cycle
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from server import app


BASE_DIR = Path(__file__).resolve().parents[1]
TEST_FILE = BASE_DIR / "data" / "processed" / "v2025.08.10" / "test_data.csv"
DEFAULT_OUT = BASE_DIR / "data" / "processed" / "v2025.08.10" / "eval" / "burst_latency.json"


async def send_request(client, payload):
    start = time.perf_counter()
    response = await client.post("/predict", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000
    return latency_ms, response.json()


def expand_payloads(payloads, total):
    if total is None or total <= len(payloads):
        return payloads[: total or len(payloads)]
    expanded = []
    feeder = cycle(payloads)
    while len(expanded) < total:
        expanded.append(next(feeder))
    return expanded


async def main(args):
    test_path = Path(args.test_file) if args.test_file else TEST_FILE
    df = pd.read_csv(test_path)
    feature_cols = [
        "url_length",
        "domain_length",
        "path_length",
        "num_dots",
        "num_hyphens",
        "num_underscores",
        "num_slashes",
        "has_ip",
        "is_https",
        "has_port",
        "num_params",
        "is_ai_endpoint",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
    ]
    payloads = []
    timestamps = pd.to_datetime(df.get("timestamp")) if "timestamp" in df.columns else None
    for idx, row in df.iterrows():
        row_features = row[feature_cols].to_numpy(dtype=float)
        finite_features = np.nan_to_num(row_features, nan=0.0, posinf=0.0, neginf=0.0)
        features = finite_features.tolist()
        record_id = None
        if timestamps is not None and not pd.isna(timestamps.iloc[idx]):
            record_id = f"{row['source']}_{timestamps.iloc[idx].strftime('%Y%m%dT%H%M%S')}"
        context = {
            "record_id": record_id,
            "owasp_category": None if pd.isna(row.get("owasp_category")) else row.get("owasp_category"),
            "atlas_tactic": None if pd.isna(row.get("atlas_tactic")) else row.get("atlas_tactic"),
        }
        payloads.append({"features": features, "context": context})

    payloads = expand_payloads(payloads, args.total)

    if args.mode == "asgi":
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=10.0)
    else:
        transport = None
        if args.uds:
            transport = httpx.AsyncHTTPTransport(uds=str(args.uds))
        client = httpx.AsyncClient(transport=transport, base_url=args.base_url, timeout=10.0)

    latencies = []
    async with client as session:
        for i in range(0, len(payloads), args.concurrency):
            batch = payloads[i : i + args.concurrency]
            tasks = [asyncio.create_task(send_request(session, req)) for req in batch]
            results = await asyncio.gather(*tasks)
            latencies.extend([res[0] for res in results])

    stats = {
        "mode": args.mode,
        "count": len(latencies),
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(np.max(latencies)),
    }
    out_file = args.output or DEFAULT_OUT
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(stats, indent=2))
    print("Saved", out_file, stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay burst traffic against the FastAPI service")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--mode", choices=["asgi", "http"], default="asgi")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--uds", type=Path, help="Unix domain socket path for uvicorn")
    parser.add_argument("--total", type=int, help="Total requests to replay (defaults to held-out set size)")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--test-file", type=Path, help="Optional test CSV to replay")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
