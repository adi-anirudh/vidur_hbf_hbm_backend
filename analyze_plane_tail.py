#!/usr/bin/env python3
"""Statistical characterization of the plane-load tail.

Consumes the P=1024 raw plane-load and per-layer/query round vectors collected
from real Llama-3.1-8B attention over PG-19.  The analysis distinguishes:

* the distribution across all planes;
* the fixed sink-plane impulse (plane 0) from the remaining selected traffic;
* the paired serial-round reduction for every recorded layer/query event.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publication_style as ps

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/adityaan/splash_accuracy_tests/planeload_1024.json")
OUT_JSON = ROOT / "results" / "plane_tail_statistics.json"
OUT_LOAD = ROOT / "results" / "plane_tail_loads.csv"
OUT_ROUNDS = ROOT / "results" / "plane_tail_rounds.csv"
OUT_FIG = ROOT / "results" / "plots" / "plane_tail_statistics"


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.percentile(values, q))
        for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)
    }


def load_stats(load: np.ndarray) -> dict:
    normalized = load / load.mean()
    sorted_load = np.sort(load)
    return {
        "planes": int(load.size),
        "mean_load": float(load.mean()),
        "coefficient_of_variation": float(normalized.std()),
        "max_over_mean": float(normalized.max()),
        "percentiles_over_mean": percentiles(normalized),
        "fraction_within_10_percent_of_mean": float(
            np.mean(np.abs(normalized - 1) <= 0.10)
        ),
        "fraction_within_25_percent_of_mean": float(
            np.mean(np.abs(normalized - 1) <= 0.25)
        ),
        "fraction_within_50_percent_of_mean": float(
            np.mean(np.abs(normalized - 1) <= 0.50)
        ),
        "fraction_at_or_above_1_25x_mean": float(
            np.mean(normalized >= 1.25)
        ),
        "fraction_at_or_above_1_5x_mean": float(
            np.mean(normalized >= 1.50)
        ),
        "fraction_at_or_above_2x_mean": float(
            np.mean(normalized >= 2.0)
        ),
        "top_1_percent_load_share": float(
            sorted_load[-int(np.ceil(0.01 * load.size)):].sum() / load.sum()
        ),
        "top_5_percent_load_share": float(
            sorted_load[-int(np.ceil(0.05 * load.size)):].sum() / load.sum()
        ),
    }


def round_stats(rounds: np.ndarray) -> dict:
    return {
        "events": int(rounds.size),
        "mean": float(rounds.mean()),
        "std": float(rounds.std()),
        "p50": float(np.percentile(rounds, 50)),
        "p90": float(np.percentile(rounds, 90)),
        "p95": float(np.percentile(rounds, 95)),
        "p99": float(np.percentile(rounds, 99)),
        "p99_9": float(np.percentile(rounds, 99.9)),
        "max": float(rounds.max()),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(10_000)
    # Batch the bootstrap to avoid a large 10K x N temporary allocation.
    for start in range(0, len(means), 250):
        stop = min(start + 250, len(means))
        sample = rng.integers(0, len(values), (stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    return [float(x) for x in np.percentile(means, (2.5, 97.5))]


def main() -> None:
    raw = json.loads(SOURCE.read_text())
    if int(raw["planes"]) != 1024:
        raise SystemExit(f"expected P=1024, found {raw['planes']}")
    for name in ("stripe", "splash"):
        if len(raw[name]["load"]) != 1024:
            raise SystemExit(f"{name}: malformed load vector")

    loads = {
        name: np.asarray(raw[name]["load"], dtype=float)
        for name in ("stripe", "splash")
    }
    rounds = {
        name: np.asarray(raw[name]["rounds"], dtype=float)
        for name in ("stripe", "splash")
    }
    if rounds["stripe"].shape != rounds["splash"].shape:
        raise SystemExit("round vectors are not paired")

    plane0 = int(np.argmax(loads["splash"]))
    keep = np.arange(1024) != plane0
    delta = rounds["stripe"] - rounds["splash"]
    order = np.argsort(loads["splash"])[::-1]

    summary = {
        "schema_version": 1,
        "source": str(SOURCE),
        "setup": {
            key: raw[key]
            for key in (
                "ctx", "frac", "planes", "page_size",
                "budget_pages", "ideal_rounds",
            )
        },
        "sample_contract": {
            "layer_query_events": int(delta.size),
            "attention_heads_per_event": 32,
            "head_layer_query_selections": int(delta.size * 32),
            "windows": 2,
            "queries_per_window": 64,
            "layers": 32,
            "round_definition": (
                "maximum pages assigned to any plane in one layer/query event, "
                "after summing the 32 query heads"
            ),
        },
        "global_striped": {
            "plane_load": load_stats(loads["stripe"]),
            "serial_rounds": round_stats(rounds["stripe"]),
        },
        "splash": {
            "plane_load_all": load_stats(loads["splash"]),
            "fixed_sink_plane": {
                "plane_id": plane0,
                "load_over_all_plane_mean": float(
                    loads["splash"][plane0] / loads["splash"].mean()
                ),
                "second_highest_over_mean": float(
                    loads["splash"][order[1]] / loads["splash"].mean()
                ),
                "interpretation": (
                    "the sole >1.5x impulse is plane 0, which receives the "
                    "compulsory sink page in every selection"
                ),
            },
            "plane_load_excluding_fixed_sink": load_stats(
                loads["splash"][keep]
            ),
            "serial_rounds": round_stats(rounds["splash"]),
        },
        "paired_round_reduction": {
            "events": int(delta.size),
            "fraction_splash_fewer_rounds": float(np.mean(delta > 0)),
            "fraction_equal_rounds": float(np.mean(delta == 0)),
            "fraction_splash_more_rounds": float(np.mean(delta < 0)),
            "mean_rounds_removed": float(delta.mean()),
            "median_rounds_removed": float(np.median(delta)),
            "p5_p95_rounds_removed": [
                float(x) for x in np.percentile(delta, (5, 95))
            ],
            "bootstrap_95_percent_ci_mean_rounds_removed": (
                bootstrap_mean_ci(delta)
            ),
            "mean_round_speedup": float(
                rounds["stripe"].mean() / rounds["splash"].mean()
            ),
            "p99_round_speedup": float(
                np.percentile(rounds["stripe"], 99)
                / np.percentile(rounds["splash"], 99)
            ),
            "max_round_speedup": float(
                rounds["stripe"].max() / rounds["splash"].max()
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    with OUT_LOAD.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "plane", "global_striped_load", "global_striped_over_mean",
            "splash_load", "splash_over_mean", "is_fixed_sink_plane",
        ])
        for plane in range(1024):
            writer.writerow([
                plane,
                loads["stripe"][plane],
                loads["stripe"][plane] / loads["stripe"].mean(),
                loads["splash"][plane],
                loads["splash"][plane] / loads["splash"].mean(),
                int(plane == plane0),
            ])
    with OUT_ROUNDS.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "event", "global_striped_rounds", "splash_rounds",
            "rounds_removed",
        ])
        for event, (global_round, splash_round) in enumerate(zip(
            rounds["stripe"], rounds["splash"],
        )):
            writer.writerow([
                event, global_round, splash_round, global_round - splash_round,
            ])

    ps.apply()
    global_c = ps.COLORS["dense"]
    splash_c = ps.COLORS["splash"]
    sink_c = ps.COLORS["token"]
    plt.rcParams.update({"axes.titlesize":7.0,"axes.labelsize":6.4,"xtick.labelsize":5.8,"ytick.labelsize":5.8,"legend.fontsize":5.5})
    fig, axes = plt.subplots(4, 1, figsize=(3.4, 6.9))

    rank = 100 * (np.arange(1024) + 1) / 1024
    for name, color, label in (
        ("stripe", global_c, "Global top-k + stripe"),
        ("splash", splash_c, "SPLASH quota"),
    ):
        normalized = np.sort(loads[name] / loads[name].mean())
        axes[0].plot(rank, normalized, color=color, lw=1.6, label=label)
    axes[0].axhspan(0.75, 1.25, color="#777", alpha=0.10)
    axes[0].axhline(1, color="#555", ls=":", lw=0.9)
    axes[0].set_xlabel("Planes at or below load percentile (%)")
    axes[0].set_ylabel("Plane load / mean")
    axes[0].set_title("(a) Almost all planes stay near the mean")
    axes[0].legend(frameon=False)
    ps.finish_axis(axes[0])

    tail = order[:32]
    tail_load = loads["splash"][tail] / loads["splash"].mean()
    colors = [sink_c if plane == plane0 else splash_c for plane in tail]
    axes[1].bar(np.arange(32), tail_load, color=colors, width=0.82)
    axes[1].axhline(1.25, color="#555", ls=":", lw=0.9)
    axes[1].set_xticks([0, 7, 15, 23, 31])
    axes[1].set_xticklabels(["1", "8", "16", "24", "32"])
    axes[1].set_xlabel("Rank among 32 busiest planes")
    axes[1].set_ylabel("Plane load / mean")
    axes[1].set_title("(b) The maximum is one sink-plane impulse")
    axes[1].annotate(
        f"plane {plane0}: {tail_load[0]:.2f}×",
        xy=(0, tail_load[0]), xytext=(5, 1.85),
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    axes[1].annotate(
        f"next: {tail_load[1]:.2f}×",
        xy=(1, tail_load[1]), xytext=(9, 1.48),
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    ps.finish_axis(axes[1], grid_axis="y")

    thresholds = np.array([1.1, 1.25, 1.5, 1.75, 2.0])
    for name, color, label in (
        ("stripe", global_c, "Global top-k + stripe"),
        ("splash", splash_c, "SPLASH quota"),
    ):
        normalized = loads[name] / loads[name].mean()
        fraction = [max(np.mean(normalized >= t), 1e-4) for t in thresholds]
        axes[2].plot(
            thresholds, 100 * np.asarray(fraction), "-o",
            color=color, lw=1.6, ms=3.5, label=label,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Load threshold (× mean)")
    axes[2].set_ylabel("Planes exceeding threshold (%)")
    axes[2].set_title("(c) Imbalance is confined to a tiny tail")
    ps.finish_axis(axes[2])

    for name, color, label in (
        ("stripe", global_c, "Global top-k + stripe"),
        ("splash", splash_c, "SPLASH quota"),
    ):
        values = np.sort(rounds[name])
        cdf = 100 * (np.arange(values.size) + 1) / values.size
        axes[3].plot(values, cdf, color=color, lw=1.7, label=label)
    axes[3].axvline(
        np.percentile(rounds["stripe"], 99),
        color=global_c, ls=":", lw=0.9,
    )
    axes[3].axvline(
        np.percentile(rounds["splash"], 99),
        color=splash_c, ls=":", lw=0.9,
    )
    axes[3].set_xlabel("Serial read rounds / layer-query event")
    axes[3].set_ylabel("Events completed (%)")
    axes[3].set_title("(d) Quotas eliminate the serial-read tail")
    axes[3].legend(frameon=False, loc="lower right")
    ps.finish_axis(axes[3])
    axes[3].text(
        0.03, 0.72,
        "p99: 98 → 64 rounds\n"
        "99.93% of events improve",
        transform=axes[3].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_FIG}.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)

    print(OUT_JSON)
    print(
        "SPLASH: "
        f"{100 * summary['splash']['plane_load_all']['fraction_within_25_percent_of_mean']:.2f}% "
        "within ±25%; "
        f"{100 * summary['splash']['plane_load_all']['fraction_at_or_above_1_5x_mean']:.3f}% "
        "at/above 1.5x"
    )
    print(
        "Rounds: "
        f"{100 * summary['paired_round_reduction']['fraction_splash_fewer_rounds']:.2f}% "
        "of paired events improve; mean reduction "
        f"{summary['paired_round_reduction']['mean_rounds_removed']:.2f}"
    )


if __name__ == "__main__":
    main()
