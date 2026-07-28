#!/usr/bin/env python3
"""Build a provisional H100-anchored energy sensitivity for the B200 sweep.

This is intentionally separate from the paper's existing analytical energy
figure and from a future same-system B200 measurement.  It uses:

* BVidur TPOT and coefficient-free HBM/HBF traffic counts;
* the published H100 attention dynamic-power anchor;
* the published H100 activity-attributed HBM joules/byte; and
* the existing analytical HBF-array energy coefficient.

GPU board energy is power times simulated execution time. HBM is separated by
traffic, and GPU logic is the board-energy residual. HBF is outside the GPU
board boundary and is added independently. SRAM bridge, link, NMP/top-k, FTL,
and background energy remain absent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep_capacity import HBM_GB, weight_bytes
import publication_style as ps


MODELS = [
    ("Qwen/Qwen3-235B-A22B", "Qwen3-235B"),
    ("mistralai/Mixtral-8x22B-v0.1", "Mixtral-8x22B"),
    ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
    ("meta-llama/Meta-Llama-3.1-405B", "Llama3.1-405B"),
]
CONTEXT_LABELS = {
    131072: "128K",
    196608: "192K",
    262144: "256K",
    393216: "384K",
    524288: "512K",
    786432: "768K",
    1048576: "1M",
    2097152: "2M",
}
BASELINES = ["Dense", "Naive", "Sparse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project provisional hybrid energy")
    parser.add_argument(
        "--traffic",
        type=Path,
        default=Path("results/energy_b200/sweep_component_traffic.csv"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/energy_h100/source_calibrated_separation.json"),
    )
    parser.add_argument("--slo-ms", type=float, default=100.0)
    parser.add_argument(
        "--active-power-w",
        type=float,
        default=None,
        help="Per-GPU active power; defaults to the published H100 anchor",
    )
    parser.add_argument(
        "--hbf-array-pj-per-byte",
        type=float,
        default=64.0,
        help="Existing 8 pJ/bit HBF-array coefficient",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/energy_b200/provisional_hybrid_energy.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/energy_b200/provisional_hybrid_summary.json"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("figures/energy_normalized_hybrid_provisional"),
    )
    return parser.parse_args()


def geomean(values: pd.Series) -> float:
    positive = [float(value) for value in values if value > 0]
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def select_operating_points(frame: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    model_names = {name for name, _ in MODELS}
    selected = frame.loc[
        frame["model"].isin(model_names)
        & frame["baseline"].isin(BASELINES)
        & frame["tpot_p50_ms"].le(slo_ms)
    ].copy()
    selected["model_weight_bytes"] = selected["model"].map(weight_bytes)
    # HBF holds KV, not model weights. Retain only rows whose sharded weights
    # and the sweep's 4 GB activation reserve fit in B200 HBM.
    selected["weight_placement_feasible"] = (
        selected["model_weight_bytes"] / selected["tp"] + 4e9
        <= HBM_GB["blackwell"] * 1e9
    )
    selected = selected.loc[selected["weight_placement_feasible"]].copy()
    selected["goodput_per_gpu"] = (
        selected["batch"]
        / (selected["tpot_p50_ms"] / 1000.0)
        / selected["tp"]
    )
    selected.sort_values("goodput_per_gpu", ascending=False, inplace=True)
    return selected.drop_duplicates(["model", "context_length", "baseline"])


def add_energy(
    frame: pd.DataFrame,
    active_power_w: float,
    hbm_pj_per_byte: float,
    hbf_pj_per_byte: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["hbm_bytes_per_output_token_total"] = (
        (
            result["hbm_read_bytes_per_output_token"]
            + result["hbm_write_bytes_per_output_token"]
        )
        * result["tp"]
        + result["model_weight_bytes"] / result["batch"]
    )
    result["hbf_physical_bytes_per_output_token_total"] = (
        result["hbf_physical_bytes_per_output_token"] * result["tp"]
    )
    result["gpu_board_j_per_output_token"] = (
        active_power_w
        * result["tp"]
        * (result["tpot_p50_ms"] / 1000.0)
        / result["batch"]
    )
    result["hbm_j_per_output_token"] = (
        result["hbm_bytes_per_output_token_total"] * hbm_pj_per_byte * 1e-12
    )
    result["gpu_logic_residual_j_per_output_token"] = (
        result["gpu_board_j_per_output_token"]
        - result["hbm_j_per_output_token"]
    )
    if result["gpu_logic_residual_j_per_output_token"].lt(0).any():
        bad = result.loc[
            result["gpu_logic_residual_j_per_output_token"].lt(0),
            ["model", "context_length", "baseline", "batch", "tp"],
        ]
        raise SystemExit(f"negative GPU-logic residual:\n{bad.to_string(index=False)}")
    result["hbf_array_j_per_output_token"] = (
        result["hbf_physical_bytes_per_output_token_total"]
        * hbf_pj_per_byte
        * 1e-12
    )
    result["modeled_total_j_per_output_token"] = (
        result["gpu_board_j_per_output_token"]
        + result["hbf_array_j_per_output_token"]
    )
    result["hbm_fraction_of_gpu_board"] = (
        result["hbm_j_per_output_token"]
        / result["gpu_board_j_per_output_token"]
    )
    result["hbf_array_fraction_of_modeled_total"] = (
        result["hbf_array_j_per_output_token"]
        / result["modeled_total_j_per_output_token"]
    )
    result["active_power_w_per_gpu_assumption"] = active_power_w
    result["hbm_pj_per_byte_assumption"] = hbm_pj_per_byte
    result["hbf_array_pj_per_byte_assumption"] = hbf_pj_per_byte
    result["energy_status"] = "provisional_hybrid_not_measured_b200"

    sparse = result.loc[result["baseline"].eq("Sparse")].set_index(
        ["model", "context_length"]
    )["modeled_total_j_per_output_token"]
    keys = pd.MultiIndex.from_frame(result[["model", "context_length"]])
    sparse_energy = sparse.reindex(keys).to_numpy()
    result["normalized_tokens_per_joule_to_splash"] = (
        sparse_energy / result["modeled_total_j_per_output_token"].to_numpy()
    )
    return result


def summarize(
    selected: pd.DataFrame,
    calibration: dict[str, object],
    hbf_pj_per_byte: float,
    slo_ms: float,
) -> dict[str, object]:
    calculation = calibration["calculation"]
    sensitivity = calculation["cap_sensitivity_400_to_700_w"]
    hbm_values = [
        float(sensitivity["activity_attributed_hbm_pj_per_byte_min"]),
        float(calculation["activity_attributed_hbm_pj_per_byte"]),
        float(sensitivity["activity_attributed_hbm_pj_per_byte_max"]),
    ]
    power_values = [450.0, 700.0, 1000.0]
    hbf_values = [
        0.5 * hbf_pj_per_byte,
        hbf_pj_per_byte,
        1.5 * hbf_pj_per_byte,
    ]
    cases = []
    for power in power_values:
        for hbm_pj in hbm_values:
            for hbf_pj in hbf_values:
                energy = add_energy(selected, power, hbm_pj, hbf_pj)
                pivot = energy.pivot(
                    index=["model", "context_length"],
                    columns="baseline",
                    values="modeled_total_j_per_output_token",
                )
                cases.append(
                    {
                        "active_power_w_per_gpu": power,
                        "hbm_pj_per_byte": hbm_pj,
                        "hbf_array_pj_per_byte": hbf_pj,
                        "dense_over_splash_energy_geomean": geomean(
                            pivot["Dense"] / pivot["Sparse"]
                        ),
                        "naive_over_splash_energy_geomean": geomean(
                            pivot["Naive"] / pivot["Sparse"]
                        ),
                        "median_total_j_per_output_token": float(
                            energy["modeled_total_j_per_output_token"].median()
                        ),
                        "mean_hbm_fraction_of_gpu_board": float(
                            energy["hbm_fraction_of_gpu_board"].mean()
                        ),
                        "mean_hbf_array_fraction_of_modeled_total": float(
                            energy["hbf_array_fraction_of_modeled_total"].mean()
                        ),
                        "negative_gpu_logic_rows": int(
                            energy["gpu_logic_residual_j_per_output_token"].lt(0).sum()
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "status": "provisional_hybrid_not_measured_b200",
        "selected_operating_points": len(selected),
        "selection": (
            f"maximum tokens/s/GPU under the {slo_ms:g} ms TPOT SLO after requiring "
            "weights plus 4 GB reserve to fit in per-GPU HBM"
        ),
        "included": [
            "GPU board energy from active-power assumption times BVidur TPOT",
            "HBM energy from weights and hot-KV traffic",
            "HBF array energy from physical HBF bytes",
        ],
        "omitted": [
            "HBF SRAM bridge",
            "HBF dedicated link",
            "NMP/top-k and FTL logic",
            "background/static system energy",
            "cooling and host energy",
        ],
        "sensitivity_cases": cases,
    }


def plot(frame: pd.DataFrame, prefix: Path) -> None:
    colors = {
        "Dense": ps.COLORS["dense"],
        "Naive": ps.COLORS["token"],
        "Sparse": ps.COLORS["splash"],
    }
    labels = {"Dense": "H3", "Naive": "Naive Sparse", "Sparse": "SPLASH"}
    contexts = list(CONTEXT_LABELS)
    ps.apply()
    ps.apply()
    plt.rcParams.update({
        "axes.labelsize": 6.6, "xtick.labelsize": 5.8, "ytick.labelsize": 6.0,
        "legend.fontsize": 6.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.3), sharex=True, sharey=True)
    x = np.arange(len(contexts))
    width = 0.26
    for ax, (model, model_label) in zip(axes.flat, MODELS):
        group = frame.loc[frame["model"].eq(model)]
        for offset, baseline in enumerate(BASELINES):
            indexed = group.loc[group["baseline"].eq(baseline)].set_index(
                "context_length"
            )
            values = [
                float(indexed.loc[context, "normalized_tokens_per_joule_to_splash"])
                for context in contexts
            ]
            ax.bar(
                x + (offset - 1) * width, values, width,
                color=colors[baseline], edgecolor=ps.COLORS["ink"],
                linewidth=ps.BAR_EDGE_WIDTH,
                hatch="///" if baseline == "Sparse" else None,
                label=labels[baseline], zorder=3,
            )
        ax.axhline(1.0, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.015, 0.94, model_label, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.6, fontweight="bold")
        ps.finish_axis(ax, grid_axis="y")
    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels([CONTEXT_LABELS[c] for c in contexts], rotation=45,
                           fontsize=5.8)
    fig.supylabel("Normalized tokens/J", fontsize=6.8, x=0.02)
    fig.supxlabel("Context length", fontsize=6.6, y=0.005)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 1.03), frameon=False, handlelength=1.2,
        columnspacing=1.2, fontsize=6.5,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.85, bottom=0.2,
                        hspace=0.2, wspace=0.28)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(suffix))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text())
    active_power_w = args.active_power_w or float(
        calibration["attention_energy_per_token"]["measured_dynamic_board_power_w"]
    )
    hbm_pj_per_byte = float(
        calibration["calculation"]["activity_attributed_hbm_pj_per_byte"]
    )
    traffic = pd.read_csv(args.traffic)
    selected = select_operating_points(traffic, args.slo_ms)
    result = add_energy(
        selected,
        active_power_w,
        hbm_pj_per_byte,
        args.hbf_array_pj_per_byte,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = summarize(
        selected,
        calibration,
        args.hbf_array_pj_per_byte,
        args.slo_ms,
    )
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    plot(result, args.figure_prefix)
    pivot = result.pivot(
        index=["model", "context_length"],
        columns="baseline",
        values="modeled_total_j_per_output_token",
    )
    print(f"Wrote {len(result)} rows to {args.output}")
    print(
        "Energy ratios: "
        f"Dense/SPLASH={geomean(pivot['Dense'] / pivot['Sparse']):.2f}x, "
        f"Naive/SPLASH={geomean(pivot['Naive'] / pivot['Sparse']):.2f}x"
    )


if __name__ == "__main__":
    main()
