#!/usr/bin/env python3
"""Generate the paper's explicit analytical Blackwell NVLink profile.

The model is intentionally simple and auditable:

  all-reduce bytes/GPU = 2 * (N - 1) / N * payload bytes
  time = fixed device-side latency + bytes/GPU / link bandwidth

NVIDIA specifies 1.8 TB/s bidirectional NVLink bandwidth per Blackwell GPU.
The 3 us device-side latency is a nominal point; paper sensitivity varies it.
Vidur adds its existing 20 us CPU launch overhead separately.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "profiling" / "network" / "blackwell_nvl8"
FIELDS = [
    "",
    "time_stats.all_reduce.min",
    "time_stats.all_reduce.max",
    "time_stats.all_reduce.mean",
    "time_stats.all_reduce.median",
    "time_stats.all_reduce.std",
    "rank",
    "num_workers",
    "size",
    "collective",
    "devices_per_node",
    "max_devices_per_node",
]


def all_reduce_ms(
    size_bytes: int, workers: int, bandwidth_gbps: float, latency_us: float
) -> float:
    ring_bytes = 2.0 * (workers - 1) / workers * size_bytes
    return latency_us / 1000.0 + ring_bytes / (bandwidth_gbps * 1e6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bandwidth-gbps", type=float, default=1800.0)
    parser.add_argument("--latency-us", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sizes = range(2048, 8 * 1024 * 1024 + 1, 8192)
    path = args.output_dir / "all_reduce.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        row_id = 0
        for workers in (2, 4, 8):
            for size in sizes:
                value = all_reduce_ms(
                    size, workers, args.bandwidth_gbps, args.latency_us
                )
                writer.writerow({
                    "": row_id,
                    "time_stats.all_reduce.min": value,
                    "time_stats.all_reduce.max": value,
                    "time_stats.all_reduce.mean": value,
                    "time_stats.all_reduce.median": value,
                    "time_stats.all_reduce.std": 0.0,
                    "rank": 0,
                    "num_workers": workers,
                    "size": size,
                    "collective": "all_reduce",
                    # Vidur uses this field as the number of participating GPUs
                    # for an intra-node collective, not the node's physical size.
                    "devices_per_node": workers,
                    "max_devices_per_node": 8,
                })
                row_id += 1

    print(path)


if __name__ == "__main__":
    main()
