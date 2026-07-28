#!/usr/bin/env python3
"""Run and archive the two low-level timing validations used by the paper.

1. Per-request/layer/block traces versus the predictor's plane aggregation,
   through the same scheduler.
2. The Python plane scheduler versus the independent C++ HBFSim model over a
   1--1024-way parallelism sweep at matched geometry and timing.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "results"
PROGRAMS = {
    "aggregation_vs_trace": ROOT / "compare_analytical_vs_trace.py",
    "plane_scheduler_vs_hbfsim": ROOT / "validate_vs_hbfsim.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(name: str, program: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(program)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    raw = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    (OUT / f"validation_{name}.log").write_text(raw)
    if proc.returncode:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    result = {
        "program": str(program.relative_to(ROOT)),
        "program_sha256": sha256(program),
        "raw_log": f"results/validation_{name}.log",
    }
    if name == "aggregation_vs_trace":
        match = re.search(r"worst \|error\| = ([\d.]+)%", raw)
        result.update({
            "metric": "decode-step latency",
            "comparison": "aggregate-per-plane predictor vs per-read trace",
            "worst_absolute_error_percent": (
                float(match.group(1)) if match else None
            ),
            "status": "pass_same_scheduler_consistency_check",
            "paper_use": (
                "validates aggregation only; does not independently validate "
                "the NAND timing model"
            ),
        })
    else:
        match = re.search(
            r"max \|err\| = ([\d.]+)%\s+mean \|err\| = ([\d.]+)%", raw
        )
        maximum = float(match.group(1)) if match else None
        mean = float(match.group(2)) if match else None
        accepted = maximum is not None and maximum <= 5.0
        result.update({
            "metric": "aggregate read bandwidth",
            "comparison": "Python PlaneScheduler vs independent C++ HBFSim",
            "parallelism_units": [1, 4, 16, 64, 256, 1024],
            "max_absolute_error_percent": maximum,
            "mean_absolute_error_percent": mean,
            "status": (
                "pass" if accepted
                else "rejected_configuration_mismatch_not_used_for_paper"
            ),
            "paper_use": accepted,
        })
    return result


def main() -> None:
    payload = {
        "schema_version": 1,
        "python": sys.version,
        "hbf_config": "configs/hbf_paper.toml",
        "hbf_config_sha256": sha256(ROOT / "configs/hbf_paper.toml"),
        "validations": {},
    }
    for name, program in PROGRAMS.items():
        print(f"running {name}...", flush=True)
        payload["validations"][name] = run(name, program)
    path = OUT / "validation_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
