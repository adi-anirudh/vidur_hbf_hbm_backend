#!/usr/bin/env python3
"""Run the exhaustive sweep as independent model shards, then merge atomically."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_sweep_b200 import FIELDS, MODELS


ROOT = Path(__file__).resolve().parent
DRIVER = ROOT / "run_sweep_b200.py"


def run_shard(index: int, model: str, part_dir: Path, workers: int) -> Path:
    out = part_dir / f"{index:02d}.csv"
    command = [
        sys.executable, str(DRIVER),
        "--workers", str(workers),
        "--timeout", "900",
        "--only-model", model,
        "--out", str(out),
    ]
    process = subprocess.run(command, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f"shard {index} {model} exited {process.returncode}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-models", type=int, default=3)
    parser.add_argument("--workers-per-model", type=int, default=8)
    parser.add_argument(
        "--part-dir", type=Path,
        default=ROOT / "results" / "sweep_b200_final_parts",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results" / "sweep_b200_final.csv",
    )
    args = parser.parse_args()
    args.part_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.parallel_models) as pool:
        futures = {
            pool.submit(
                run_shard, index, model, args.part_dir,
                args.workers_per_model,
            ): (index, model)
            for index, model in enumerate(MODELS)
        }
        for future in as_completed(futures):
            index, model = futures[future]
            path = future.result()
            print(f"SHARD DONE {index:02d} {model} -> {path}", flush=True)

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    total = 0
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, model in enumerate(MODELS):
            part = args.part_dir / f"{index:02d}.csv"
            for row in csv.DictReader(part.open()):
                writer.writerow(row)
                total += 1
    tmp.replace(args.out)
    print(f"MERGED {total} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
