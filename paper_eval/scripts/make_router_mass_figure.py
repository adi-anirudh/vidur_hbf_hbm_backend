#!/usr/bin/env python3
"""Plot cumulative expert-ranked mass for the sweep's synthetic router."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import corrected_ctx_workload_sweep as base  # noqa: E402
import corrected_microbatch_sweep as micro  # noqa: E402

MICROBATCHES = 2
WORKLOADS = (32, 128, 512, 1024)
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
COLORS = ("#4c78a8", "#59a14f", "#f28e2b", "#d62728")


def cumulative_mass_samples(sessions: int) -> np.ndarray:
    """Return sample-by-rank cumulative selected-gate-mass curves."""
    samples = []
    slices = micro.microbatch_slices(sessions, MICROBATCHES)
    for trial in base.routing_samples(sessions):
        for layer in trial:
            for start, end in slices:
                mass = np.zeros(base.N_EXPERTS, dtype=np.float64)
                for experts, weights in zip(
                    layer.routes[start:end], layer.gate_weights[start:end]
                ):
                    mass[np.asarray(experts)] += np.asarray(weights)
                mass /= mass.sum()  # Average selected router mass per token.
                samples.append(np.cumsum(np.sort(mass)[::-1]))
    curves = np.asarray(samples)
    assert curves.shape == (
        base.ROUTING_TRIALS * base.N_MOE * MICROBATCHES,
        base.N_EXPERTS,
    )
    assert np.allclose(curves[:, -1], 1.0)
    return curves


def summarize(curves: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": curves.mean(axis=0),
        "p10": np.percentile(curves, 10, axis=0),
        "p90": np.percentile(curves, 90, axis=0),
    }


def first_rank(curve: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(curve, threshold, side="left") + 1)


def threshold_counts(curves: np.ndarray, threshold: float) -> np.ndarray:
    return np.argmax(curves >= threshold, axis=1) + 1


def make_plot(all_curves: dict[int, np.ndarray], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ranks = np.arange(1, base.N_EXPERTS + 1)
    for sessions, color in zip(WORKLOADS, COLORS):
        stats = summarize(all_curves[sessions])
        k98 = first_rank(stats["mean"], 0.98)
        ax.plot(
            ranks,
            stats["mean"],
            color=color,
            linewidth=2.4,
            label=f"S={sessions} ({sessions // 2} tokens/MB), K₉₈={k98}",
        )
        ax.fill_between(
            ranks, stats["p10"], stats["p90"], color=color, alpha=0.10,
            linewidth=0,
        )
    ax.plot(
        ranks, ranks / base.N_EXPERTS, "--", color="#777777", linewidth=1.3,
        label="Uniform mass",
    )
    for threshold in (0.90, 0.95, 0.98):
        ax.axhline(threshold, color="#999999", linestyle=":", linewidth=0.9)
        ax.text(257, threshold, f"{threshold:.0%}", va="center", fontsize=8)
    ax.set(xlim=(1, 256), ylim=(0, 1.015))
    ax.set_xticks([1, 16, 32, 64, 96, 128, 160, 192, 224, 256])
    y = np.linspace(0, 1, 11)
    ax.set_yticks(y, [f"{value:.0%}" for value in y])
    ax.set_xlabel("number of experts retained, ranked by aggregate gate mass")
    ax.set_ylabel("cumulative selected router mass")
    ax.set_title(
        "Router-mass concentration across 256 experts\n"
        "Mean with 10th–90th percentile bands across layers, trials, and microbatches"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.text(
        0.5, 0.01,
        "SYNTHETIC ROUTER USED BY THE EVALUATOR — NOT A DEEPSEEK-V3 TRACE. "
        "Mass is before top-1 protection.",
        ha="center", fontsize=9, color="#9c1c1c", fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(all_curves: dict[int, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sessions", "microbatches", "sessions_per_microbatch", "expert_rank",
            "mean_cumulative_mass", "p10_cumulative_mass", "p90_cumulative_mass",
        ])
        for sessions in WORKLOADS:
            stats = summarize(all_curves[sessions])
            for rank in range(base.N_EXPERTS):
                writer.writerow([
                    sessions, MICROBATCHES, sessions // MICROBATCHES, rank + 1,
                    f"{stats['mean'][rank]:.8f}", f"{stats['p10'][rank]:.8f}",
                    f"{stats['p90'][rank]:.8f}",
                ])


def write_summary(all_curves: dict[int, np.ndarray], path: Path) -> None:
    lines = [
        "# Synthetic router cumulative-mass summary at μ=2", "",
        "> This characterizes the router generator used by the analytical sweep. It is not a DeepSeek-V3 measurement.",
        "> Counts use router mass before adding top-1-protected experts.", "",
        "Each distribution contains `4 trials × 58 MoE layers × 2 microbatches = 464` samples.", "",
        "| Sessions | Tokens/MB | Target | Experts on mean curve | Median/sample | 10th–90th percentile |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for sessions in WORKLOADS:
        curves = all_curves[sessions]
        mean = curves.mean(axis=0)
        for target in THRESHOLDS:
            counts = threshold_counts(curves, target)
            lines.append(
                f"| {sessions} | {sessions // 2} | {target:.0%} | "
                f"{first_rank(mean, target)} | {np.median(counts):.0f} | "
                f"{np.percentile(counts, 10):.0f}–{np.percentile(counts, 90):.0f} |"
            )
    lines += ["", "Gate logits are normalized over each token's selected top-8 experts; mass outside that selected set is unavailable in the current synthetic trace.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        default=str(SCRIPT_DIR.parent / "figures" / "current" / "fig1_router_mass_concentration"),
    )
    parser.add_argument(
        "--csv", default=str(SCRIPT_DIR.parent / "results" / "synthetic_router_mass_cdf_mu2.csv"),
    )
    parser.add_argument(
        "--summary", default=str(SCRIPT_DIR.parent / "results" / "synthetic_router_mass_cdf_mu2.md"),
    )
    args = parser.parse_args()
    all_curves = {sessions: cumulative_mass_samples(sessions) for sessions in WORKLOADS}
    make_plot(all_curves, Path(args.figure))
    write_csv(all_curves, Path(args.csv))
    write_summary(all_curves, Path(args.summary))
    print(f"wrote {Path(args.figure).with_suffix('.png')}")
    print(f"wrote {Path(args.figure).with_suffix('.pdf')}")
    print(f"wrote {args.csv}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
