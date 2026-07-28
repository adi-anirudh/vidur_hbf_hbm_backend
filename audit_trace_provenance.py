#!/usr/bin/env python3
"""Emit a machine-readable audit of every profile consumed by the paper sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from run_sweep_b200 import MODELS, TP_CHOICES


ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "data" / "profiling"
OUT = ROOT / "results" / "trace_provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path, tp_column: str) -> dict:
    frame = pd.read_csv(path)
    result = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "tensor_parallel_degrees": sorted(
            int(value) for value in frame[tp_column].dropna().unique()
        ),
    }
    if "attention_backend" in frame:
        result["attention_backends"] = sorted(
            str(value) for value in frame["attention_backend"].dropna().unique()
        )
    return result


def main() -> None:
    models = {}
    for model in MODELS:
        directory = PROFILE_ROOT / "compute" / "blackwell" / model
        mlp = directory / "mlp.csv"
        attention = directory / "attention.csv"
        if not mlp.exists() or not attention.exists():
            raise FileNotFoundError(f"incomplete Blackwell profile for {model}")
        models[model] = {
            "device_label": "blackwell",
            "collection_stack": "Sarathi-Serve Vidur profiler",
            "mlp": inspect_csv(mlp, "num_tensor_parallel_workers"),
            "attention": inspect_csv(
                attention, "num_tensor_parallel_workers"
            ),
        }

    collective = PROFILE_ROOT / "network" / "blackwell_nvl8" / "all_reduce.csv"
    report = {
        "schema_version": 1,
        "paper_models": models,
        "requested_tensor_parallel_degrees": list(TP_CHOICES),
        "resolution": {
            "compute_template":
                "data/profiling/compute/{DEVICE}/{MODEL}/{mlp,attention}.csv",
            "resolved_device": "blackwell",
            "cross_device_compute_fallback": False,
        },
        "tensor_parallel_collective": {
            "path": str(collective.relative_to(ROOT)),
            "sha256": sha256(collective),
            "rows": int(len(pd.read_csv(collective))),
            "status": (
                "analytical ring model at 1.8 TB/s/GPU plus an exposed "
                "3 us device-side latency; Vidur adds 20 us launch overhead"
            ),
            "generator": "generate_blackwell_network_profile.py",
            "paper_action": "report fixed-latency sensitivity",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
