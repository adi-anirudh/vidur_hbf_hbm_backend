#!/usr/bin/env python3
"""Validate, de-duplicate, and atomically merge the final model shards."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from run_sweep_b200 import (
    BASELINES,
    BATCHES,
    CONTEXTS,
    DEVICES,
    FIELDS,
    MODELS,
    feasible_tps,
)


def key(row: dict[str, str]) -> tuple:
    return (
        row["model"],
        row["device"],
        int(row["batch"]),
        int(row["context_length"]),
        row["baseline"],
        int(row["tp"]),
    )


def expected_keys(model: str) -> set[tuple]:
    expected = set()
    for device in DEVICES:
        for batch in BATCHES:
            for context in CONTEXTS:
                for baseline, _sparsity, tier in BASELINES:
                    tps, _footprint = feasible_tps(
                        model, device, batch, context, tier
                    )
                    for tp in tps:
                        expected.add(
                            (model, device, batch, context, baseline, tp)
                        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part-dir", type=Path,
        default=Path("results/sweep_b200_final_parts"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/sweep_b200_final.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("results/sweep_b200_final_merge_audit.json"),
    )
    args = parser.parse_args()

    merged: dict[tuple, dict[str, str]] = {}
    shard_audit = {}
    incomplete = {}
    for index, model in enumerate(MODELS):
        path = args.part_dir / f"{index:02d}.csv"
        counters = Counter()
        valid: dict[tuple, dict[str, str]] = {}
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                counters["raw"] += 1
                if None in row or any(field not in row for field in FIELDS):
                    counters["malformed"] += 1
                    continue
                if row.get("model") != model:
                    counters["wrong_model"] += 1
                    continue
                if row.get("status") != "OK":
                    counters["non_ok"] += 1
                    continue
                try:
                    row_key = key(row)
                    float(row["tpot_p50_ms"])
                    float(row["tpot_p99_ms"])
                except (KeyError, TypeError, ValueError):
                    counters["malformed"] += 1
                    continue
                if row_key in valid:
                    counters["duplicate_ok"] += 1
                valid[row_key] = {field: row[field] for field in FIELDS}

        expected = expected_keys(model)
        missing = expected - set(valid)
        unexpected = set(valid) - expected
        if missing or unexpected:
            incomplete[model] = {
                "missing": len(missing),
                "unexpected": len(unexpected),
            }
        for unexpected_key in unexpected:
            valid.pop(unexpected_key)
        merged.update(valid)
        shard_audit[model] = {
            **dict(counters),
            "valid_unique_ok": len(valid),
            "expected": len(expected),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    payload = {
        "parts": str(args.part_dir),
        "output": str(args.out),
        "complete": not incomplete,
        "rows": len(merged),
        "shards": shard_audit,
        "incomplete": incomplete,
    }
    args.audit.write_text(json.dumps(payload, indent=2) + "\n")
    if incomplete:
        raise SystemExit(
            "Refusing partial merge; incomplete shards: "
            + ", ".join(f"{model}: {counts}" for model, counts in incomplete.items())
        )

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row_key in sorted(merged):
            writer.writerow(merged[row_key])
    tmp.replace(args.out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
