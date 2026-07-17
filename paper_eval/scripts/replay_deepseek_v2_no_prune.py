#!/usr/bin/env python3
"""Held-out, no-pruning replay of DeepSeek-V2-Lite routing traces.

This script deliberately produces outputs outside ``figures/current``.  It
uses the first half of trace sequences to build a profile-optimized fixed
expert placement and the second half to compare that placement with dynamic
per-microbatch LPT assignment.  The replay models routed-expert weight reads
and expert-token compute; it is not an end-to-end latency simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
DEFAULT_TRACE = (
    PAPER_DIR
    / "figures"
    / "current"
    / "deepseek_trace_run"
    / "deepseek_traces.npz"
)
DEFAULT_OUTDIR = PAPER_DIR / "experiments" / "deepseek_v2_lite_no_prune"


@dataclass(frozen=True)
class ReplayConfig:
    hidden_size: int = 2048
    moe_intermediate_size: int = 1408
    dtype_bytes: int = 2
    memory_bandwidth_gbps: float = 1024.0
    gpu_flops: float = 1_000e12
    profile_fraction: float = 0.5

    @property
    def expert_weight_bytes(self) -> int:
        # Gate, up, and down matrices.
        return 3 * self.hidden_size * self.moe_intermediate_size * self.dtype_bytes

    @property
    def expert_read_ms(self) -> float:
        return self.expert_weight_bytes / (self.memory_bandwidth_gbps * 1e9) * 1e3

    @property
    def expert_token_flops(self) -> int:
        return 6 * self.hidden_size * self.moe_intermediate_size

    @property
    def expert_token_compute_ms(self) -> float:
        return self.expert_token_flops / self.gpu_flops * 1e3


@dataclass
class SummaryRow:
    microbatch_size: int
    moe_gpus: int
    eval_decode_steps: int
    active_experts_mean: float
    active_experts_p95: float
    max_expert_tokens_mean: float
    expert_token_skew_mean: float
    expert_token_skew_p95: float
    static_max_fetched_experts_mean: float
    dynamic_max_fetched_experts_mean: float
    static_max_routed_tokens_mean: float
    dynamic_max_routed_tokens_mean: float
    static_moe_stack_ms_mean: float
    dynamic_moe_stack_ms_mean: float
    static_moe_stack_ms_p95: float
    dynamic_moe_stack_ms_p95: float
    mean_latency_reduction_pct: float
    p95_latency_reduction_pct: float
    dynamic_slower_layer_fraction: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def scalar(npz: np.lib.npyio.NpzFile, key: str):
    return npz[key].item()


def validate_trace(npz: np.lib.npyio.NpzFile) -> dict:
    required = {
        "selected_experts",
        "routing_weights",
        "token_ids",
        "moe_layer_idx",
        "n_routed_experts",
        "top_k",
        "n_shared_experts",
        "model",
        "revision",
        "dtype",
        "weight_semantics",
    }
    missing = required.difference(npz.files)
    if missing:
        raise ValueError(f"trace is missing keys: {sorted(missing)}")

    experts = npz["selected_experts"]
    weights = npz["routing_weights"]
    token_ids = npz["token_ids"]
    layers = npz["moe_layer_idx"]
    n_experts = int(scalar(npz, "n_routed_experts"))
    top_k = int(scalar(npz, "top_k"))

    if experts.ndim != 4:
        raise ValueError(f"selected_experts must be rank 4, got {experts.shape}")
    if experts.shape != weights.shape:
        raise ValueError("selected_experts and routing_weights shapes differ")
    if experts.shape[:2] != token_ids.shape:
        raise ValueError("token_ids do not match sequence/token trace dimensions")
    if experts.shape[2] != layers.size or experts.shape[3] != top_k:
        raise ValueError("trace shape disagrees with layer/top-k metadata")
    if int(experts.min()) < 0 or int(experts.max()) >= n_experts:
        raise ValueError("selected expert ID outside configured range")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("routing weights must be finite and nonnegative")
    if np.any(np.diff(np.sort(experts, axis=-1), axis=-1) == 0):
        raise ValueError("a token/layer contains duplicate selected experts")

    return {
        "model": str(scalar(npz, "model")),
        "revision": str(scalar(npz, "revision")),
        "dtype": str(scalar(npz, "dtype")),
        "n_sequences": int(experts.shape[0]),
        "sequence_length": int(experts.shape[1]),
        "n_moe_layers": int(experts.shape[2]),
        "n_routed_experts": n_experts,
        "top_k": top_k,
        "n_shared_experts": int(scalar(npz, "n_shared_experts")),
        "moe_layer_idx": layers.astype(int).tolist(),
        "weight_semantics": str(scalar(npz, "weight_semantics")),
        "weight_ordering": "not assumed; no-pruning replay uses selected IDs only",
    }


def route_count_tensor(selected: np.ndarray, microbatch_size: int, n_experts: int) -> np.ndarray:
    """Return [decode_group, position, layer, expert] routed-token counts."""
    n_seq, seq_len, n_layers, top_k = selected.shape
    if n_seq % microbatch_size:
        raise ValueError(f"{n_seq} sequences not divisible by microbatch size {microbatch_size}")

    groups = n_seq // microbatch_size
    routed = (
        selected.reshape(groups, microbatch_size, seq_len, n_layers, top_k)
        .transpose(0, 2, 3, 1, 4)
        .reshape(groups * seq_len * n_layers, microbatch_size * top_k)
    )
    counts = np.zeros((routed.shape[0], n_experts), dtype=np.uint16)
    for row, expert_ids in enumerate(routed):
        counts[row] = np.bincount(expert_ids, minlength=n_experts)
    return counts.reshape(groups, seq_len, n_layers, n_experts)


def lpt_owner(job_costs: np.ndarray, n_gpus: int) -> np.ndarray:
    """Assign jobs to identical GPUs using deterministic longest-processing-time first."""
    loads = np.zeros(n_gpus, dtype=np.float64)
    owners = np.full(job_costs.size, -1, dtype=np.int16)
    order = np.lexsort((np.arange(job_costs.size), -job_costs))
    for job in order:
        gpu = int(np.argmin(loads))
        owners[job] = gpu
        loads[gpu] += float(job_costs[job])
    return owners


def profile_static_owners(
    profile_counts: np.ndarray,
    n_gpus: int,
    config: ReplayConfig,
) -> np.ndarray:
    """Build one fixed, profile-LPT expert placement per MoE layer."""
    n_layers = profile_counts.shape[2]
    n_experts = profile_counts.shape[3]
    owners = np.empty((n_layers, n_experts), dtype=np.int16)
    for layer in range(n_layers):
        layer_counts = profile_counts[:, :, layer, :].reshape(-1, n_experts)
        expected_cost = (
            np.mean(layer_counts > 0, axis=0) * config.expert_read_ms
            + np.mean(layer_counts, axis=0) * config.expert_token_compute_ms
        )
        owners[layer] = lpt_owner(expected_cost, n_gpus)
    return owners


def assign_static(
    counts: np.ndarray,
    owners: np.ndarray,
    config: ReplayConfig,
) -> tuple[float, int, int]:
    active = np.flatnonzero(counts)
    loads = np.zeros(int(owners.max()) + 1, dtype=np.float64)
    fetches = np.zeros_like(loads, dtype=np.int32)
    token_calls = np.zeros_like(loads, dtype=np.int32)
    jobs = config.expert_read_ms + counts[active] * config.expert_token_compute_ms
    assigned = owners[active]
    np.add.at(loads, assigned, jobs)
    np.add.at(fetches, assigned, 1)
    np.add.at(token_calls, assigned, counts[active])
    return float(loads.max(initial=0.0)), int(fetches.max(initial=0)), int(token_calls.max(initial=0))


def assign_dynamic(
    counts: np.ndarray,
    n_gpus: int,
    config: ReplayConfig,
) -> tuple[float, int, int]:
    active = np.flatnonzero(counts)
    jobs = config.expert_read_ms + counts[active] * config.expert_token_compute_ms
    order = np.lexsort((active, -jobs))
    loads = np.zeros(n_gpus, dtype=np.float64)
    fetches = np.zeros(n_gpus, dtype=np.int32)
    token_calls = np.zeros(n_gpus, dtype=np.int32)
    for offset in order:
        gpu = int(np.argmin(loads))
        expert = int(active[offset])
        loads[gpu] += float(jobs[offset])
        fetches[gpu] += 1
        token_calls[gpu] += int(counts[expert])
    return float(loads.max(initial=0.0)), int(fetches.max(initial=0)), int(token_calls.max(initial=0))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def replay_one(
    profile_counts: np.ndarray,
    eval_counts: np.ndarray,
    microbatch_size: int,
    n_gpus: int,
    n_shared_experts: int,
    config: ReplayConfig,
) -> SummaryRow:
    owners = profile_static_owners(profile_counts, n_gpus, config)
    groups, seq_len, n_layers, _ = eval_counts.shape

    active_experts = np.count_nonzero(eval_counts, axis=-1)
    max_expert_tokens = np.max(eval_counts, axis=-1)
    mean_active_tokens = np.divide(
        np.sum(eval_counts, axis=-1),
        active_experts,
        out=np.zeros_like(active_experts, dtype=np.float64),
        where=active_experts > 0,
    )
    token_skew = np.divide(
        max_expert_tokens,
        mean_active_tokens,
        out=np.zeros_like(mean_active_tokens),
        where=mean_active_tokens > 0,
    )

    shape = (groups, seq_len, n_layers)
    static_routed_ms = np.zeros(shape, dtype=np.float64)
    dynamic_routed_ms = np.zeros(shape, dtype=np.float64)
    static_fetch_max = np.zeros(shape, dtype=np.float64)
    dynamic_fetch_max = np.zeros(shape, dtype=np.float64)
    static_token_max = np.zeros(shape, dtype=np.float64)
    dynamic_token_max = np.zeros(shape, dtype=np.float64)

    for group in range(groups):
        for position in range(seq_len):
            for layer in range(n_layers):
                counts = eval_counts[group, position, layer]
                s_ms, s_fetch, s_tokens = assign_static(counts, owners[layer], config)
                d_ms, d_fetch, d_tokens = assign_dynamic(counts, n_gpus, config)
                static_routed_ms[group, position, layer] = s_ms
                dynamic_routed_ms[group, position, layer] = d_ms
                static_fetch_max[group, position, layer] = s_fetch
                dynamic_fetch_max[group, position, layer] = d_fetch
                static_token_max[group, position, layer] = s_tokens
                dynamic_token_max[group, position, layer] = d_tokens

    # Shared experts are balanced and common to both methods. Each participating
    # GPU reads the shared weights once and handles at most ceil(B/M) tokens.
    shared_ms_per_layer = (
        n_shared_experts * config.expert_read_ms
        + math.ceil(microbatch_size / n_gpus)
        * n_shared_experts
        * config.expert_token_compute_ms
    )
    static_layer_ms = static_routed_ms + shared_ms_per_layer
    dynamic_layer_ms = dynamic_routed_ms + shared_ms_per_layer
    static_stack_ms = np.sum(static_layer_ms, axis=-1).reshape(-1)
    dynamic_stack_ms = np.sum(dynamic_layer_ms, axis=-1).reshape(-1)

    static_mean = float(np.mean(static_stack_ms))
    dynamic_mean = float(np.mean(dynamic_stack_ms))
    static_p95 = percentile(static_stack_ms, 95)
    dynamic_p95 = percentile(dynamic_stack_ms, 95)

    return SummaryRow(
        microbatch_size=microbatch_size,
        moe_gpus=n_gpus,
        eval_decode_steps=int(static_stack_ms.size),
        active_experts_mean=float(np.mean(active_experts)),
        active_experts_p95=percentile(active_experts, 95),
        max_expert_tokens_mean=float(np.mean(max_expert_tokens)),
        expert_token_skew_mean=float(np.mean(token_skew)),
        expert_token_skew_p95=percentile(token_skew, 95),
        static_max_fetched_experts_mean=float(np.mean(static_fetch_max)),
        dynamic_max_fetched_experts_mean=float(np.mean(dynamic_fetch_max)),
        static_max_routed_tokens_mean=float(np.mean(static_token_max)),
        dynamic_max_routed_tokens_mean=float(np.mean(dynamic_token_max)),
        static_moe_stack_ms_mean=static_mean,
        dynamic_moe_stack_ms_mean=dynamic_mean,
        static_moe_stack_ms_p95=static_p95,
        dynamic_moe_stack_ms_p95=dynamic_p95,
        mean_latency_reduction_pct=100.0 * (1.0 - dynamic_mean / static_mean),
        p95_latency_reduction_pct=100.0 * (1.0 - dynamic_p95 / static_p95),
        dynamic_slower_layer_fraction=float(np.mean(dynamic_layer_ms > static_layer_ms)),
    )


def write_summary(rows: list[SummaryRow], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "load_balance_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    lines = [
        "# DeepSeek-V2-Lite no-pruning replay summary",
        "",
        "Held-out evaluation: first 128 sequences profile fixed placement; last 128 evaluate both methods.",
        "",
        "| B | MoE GPUs | Active experts | Token skew | Static mean (ms) | Dynamic mean (ms) | Mean reduction | p95 reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.microbatch_size} | {row.moe_gpus} | "
            f"{row.active_experts_mean:.2f} | {row.expert_token_skew_mean:.2f}× | "
            f"{row.static_moe_stack_ms_mean:.3f} | {row.dynamic_moe_stack_ms_mean:.3f} | "
            f"{row.mean_latency_reduction_pct:.2f}% | {row.p95_latency_reduction_pct:.2f}% |"
        )
    (results_dir / "load_balance_summary.md").write_text("\n".join(lines) + "\n")


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_routing_characterization(rows: list[SummaryRow], figures_dir: Path) -> None:
    # Routing-only metrics do not depend on the number of MoE GPUs.
    by_batch = {}
    for row in rows:
        by_batch.setdefault(row.microbatch_size, row)
    batches = sorted(by_batch)
    active_mean = [by_batch[b].active_experts_mean for b in batches]
    active_p95 = [by_batch[b].active_experts_p95 for b in batches]
    skew_mean = [by_batch[b].expert_token_skew_mean for b in batches]
    skew_p95 = [by_batch[b].expert_token_skew_p95 for b in batches]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    axes[0].plot(batches, active_mean, "o-", linewidth=2.2, label="Mean")
    axes[0].plot(batches, active_p95, "s--", linewidth=1.8, label="p95")
    axes[0].axhline(64, color="#777777", linestyle=":", linewidth=1.2, label="All 64 experts")
    axes[0].set_ylabel("active routed experts / layer")
    axes[0].set_title("Expert union saturates with microbatch size")

    axes[1].plot(batches, skew_mean, "o-", linewidth=2.2, color="#d62728", label="Mean")
    axes[1].plot(batches, skew_p95, "s--", linewidth=1.8, color="#9467bd", label="p95")
    axes[1].axhline(1, color="#777777", linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("max / mean active-expert token count")
    axes[1].set_title("Token load remains imbalanced")

    for ax in axes:
        ax.set_xlabel("sequences per microbatch")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, [str(batch) for batch in batches])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Measured DeepSeek-V2-Lite routing · WikiText-103 · no pruning")
    fig.tight_layout()
    save_figure(fig, figures_dir, "fig1_routing_characterization")


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    batches: list[int],
    gpu_counts: list[int],
    title: str,
    vmax: float,
):
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(gpu_counts)), gpu_counts)
    ax.set_yticks(range(len(batches)), batches)
    ax.set_xlabel("MoE GPUs")
    ax.set_ylabel("sequences per microbatch")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=9)
    return image


def plot_dynamic_gain(rows: list[SummaryRow], figures_dir: Path) -> None:
    batches = sorted({row.microbatch_size for row in rows})
    gpu_counts = sorted({row.moe_gpus for row in rows})
    lookup = {(row.microbatch_size, row.moe_gpus): row for row in rows}
    mean_gain = np.array(
        [[lookup[(batch, gpu)].mean_latency_reduction_pct for gpu in gpu_counts] for batch in batches]
    )
    p95_gain = np.array(
        [[lookup[(batch, gpu)].p95_latency_reduction_pct for gpu in gpu_counts] for batch in batches]
    )
    vmax = max(1.0, math.ceil(float(max(np.max(mean_gain), np.max(p95_gain)))))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    image = heatmap(axes[0], mean_gain, batches, gpu_counts, "Mean MoE-stack latency reduction", vmax)
    heatmap(axes[1], p95_gain, batches, gpu_counts, "p95 MoE-stack latency reduction", vmax)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("dynamic versus profile-fixed reduction (%)")
    fig.suptitle("Full replication enables per-microbatch assignment · no pruning")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.84, wspace=0.14)
    save_figure(fig, figures_dir, "fig2_dynamic_vs_profile_fixed")


def plot_balance_components(rows: list[SummaryRow], figures_dir: Path) -> None:
    batches = sorted({row.microbatch_size for row in rows})
    gpu_counts = sorted({row.moe_gpus for row in rows})
    lookup = {(row.microbatch_size, row.moe_gpus): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(gpu_counts)))
    for color, gpu in zip(colors, gpu_counts):
        fetch_reduction = [
            100.0
            * (
                1.0
                - lookup[(batch, gpu)].dynamic_max_fetched_experts_mean
                / lookup[(batch, gpu)].static_max_fetched_experts_mean
            )
            for batch in batches
        ]
        token_reduction = [
            100.0
            * (
                1.0
                - lookup[(batch, gpu)].dynamic_max_routed_tokens_mean
                / lookup[(batch, gpu)].static_max_routed_tokens_mean
            )
            for batch in batches
        ]
        axes[0].plot(batches, fetch_reduction, "o-", color=color, linewidth=2, label=f"M={gpu}")
        axes[1].plot(batches, token_reduction, "o-", color=color, linewidth=2, label=f"M={gpu}")

    axes[0].set_title("Slowest-GPU expert-fetch reduction")
    axes[0].set_ylabel("dynamic versus profile-fixed reduction (%)")
    axes[1].set_title("Slowest-GPU routed-token reduction")
    for ax in axes:
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1.1)
        ax.set_xlabel("sequences per microbatch")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, [str(batch) for batch in batches])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("What dynamic assignment balances · measured routing · no pruning")
    fig.tight_layout()
    save_figure(fig, figures_dir, "fig3_fetch_vs_compute_balance")


def write_readme(
    outdir: Path,
    trace_path: Path,
    trace_meta: dict,
    config: ReplayConfig,
    rows: list[SummaryRow],
) -> None:
    best = max(rows, key=lambda row: row.mean_latency_reduction_pct)
    lines = [
        "# DeepSeek-V2-Lite real-trace, no-pruning experiment",
        "",
        "This directory is intentionally separate from `paper_eval/figures/current`. Nothing here is a current DeepSeek-V3 paper result.",
        "",
        "## What this tests",
        "",
        "The replay compares profile-optimized fixed expert ownership against per-microbatch dynamic LPT assignment. Every selected routed expert executes; there is no pruning. The first half of sequences profiles fixed placement and the second half is held out for evaluation.",
        "",
        "## What this does not test",
        "",
        "- It is not a DeepSeek-V3 trace or a 256-expert result.",
        "- It does not model attention, network communication, scheduler overhead, or end-to-end TBT.",
        "- It does not validate routing at long context; the trace length is 256 tokens.",
        "- V2-Lite's complete routed-expert collection fits in conventional HBM, so this validates scheduling behavior rather than HBF capacity necessity.",
        "",
        "## Inputs and assumptions",
        "",
        f"- Trace: `{trace_path}`",
        f"- Trace SHA-256: `{sha256(trace_path)}`",
        f"- Model revision: `{trace_meta['revision']}`",
        f"- Trace collection dtype: `{trace_meta['dtype']}`.",
        f"- Expert dimensions: hidden `{config.hidden_size}`, intermediate `{config.moe_intermediate_size}`.",
        f"- Modeled expert storage: `{config.dtype_bytes}` bytes/parameter; one routed expert is `{config.expert_weight_bytes / 1e6:.3f}` MB.",
        f"- HBM/HBF analytical bandwidth: `{config.memory_bandwidth_gbps:g}` GB/s.",
        f"- Analytical compute: `{config.gpu_flops / 1e12:g}` TFLOP/s.",
        f"- At this roofline, one expert read costs the same time as approximately `{config.expert_read_ms / config.expert_token_compute_ms:.1f}` expert-token computations.",
        "- Routed experts are indivisible jobs; hot-expert splitting is disabled.",
        "- Shared-expert work is included as the same balanced cost for both methods.",
        "",
        "## Files",
        "",
        "- `run_config.json`: exact replay inputs and assumptions.",
        "- `results/load_balance_summary.csv`: machine-readable results.",
        "- `results/load_balance_summary.md`: compact human-readable table.",
        "- `figures/fig1_routing_characterization.*`: active-expert union and token skew.",
        "- `figures/fig2_dynamic_vs_profile_fixed.*`: mean and p95 modeled MoE-stack reductions.",
        "- `figures/fig3_fetch_vs_compute_balance.*`: slowest-GPU fetch and routed-token reductions.",
        "",
        "## Headline within this limited experiment",
        "",
        f"The largest mean modeled reduction is `{best.mean_latency_reduction_pct:.2f}%` at microbatch size `{best.microbatch_size}` with `{best.moe_gpus}` MoE GPUs. Interpret this only as a trace-driven scheduling result under the assumptions above.",
        "Under the selected roofline, expert reads dominate expert-token compute. Dynamic assignment continues to reduce routed-token imbalance at large microbatches, but that does not materially reduce modeled latency once nearly every expert is fetched.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig \\",
        "  .venv/bin/python paper_eval/scripts/replay_deepseek_v2_no_prune.py",
        "```",
        "",
    ]
    (outdir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--microbatch-sizes", type=parse_int_list, default=(8, 16, 32, 64, 128))
    parser.add_argument("--moe-gpus", type=parse_int_list, default=(2, 4, 8, 16))
    parser.add_argument("--bandwidth-gbps", type=float, default=1024.0)
    parser.add_argument("--gpu-tflops", type=float, default=1000.0)
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        choices=(1, 2),
        default=2,
        help="modeled expert-weight storage bytes per parameter (1=FP8, 2=BF16)",
    )
    args = parser.parse_args()

    config = ReplayConfig(
        dtype_bytes=args.dtype_bytes,
        memory_bandwidth_gbps=args.bandwidth_gbps,
        gpu_flops=args.gpu_tflops * 1e12,
    )
    if config.memory_bandwidth_gbps <= 0 or config.gpu_flops <= 0:
        parser.error("bandwidth and GPU throughput must be positive")

    with np.load(args.trace, allow_pickle=False) as npz:
        trace_meta = validate_trace(npz)
        selected = np.array(npz["selected_experts"], copy=True)

    n_seq = selected.shape[0]
    profile_n = int(n_seq * config.profile_fraction)
    if profile_n <= 0 or profile_n >= n_seq:
        raise ValueError("profile split must leave nonempty profile and evaluation sets")
    profile_selected = selected[:profile_n]
    eval_selected = selected[profile_n:]
    n_experts = trace_meta["n_routed_experts"]

    rows = []
    for microbatch_size in args.microbatch_sizes:
        if profile_n % microbatch_size or (n_seq - profile_n) % microbatch_size:
            raise ValueError(
                f"microbatch size {microbatch_size} must divide both {profile_n}-sequence splits"
            )
        print(f"building route counts for B={microbatch_size}", flush=True)
        profile_counts = route_count_tensor(profile_selected, microbatch_size, n_experts)
        eval_counts = route_count_tensor(eval_selected, microbatch_size, n_experts)
        for n_gpus in args.moe_gpus:
            if n_gpus > n_experts:
                raise ValueError("MoE GPU count cannot exceed routed expert count")
            print(f"  replaying M={n_gpus}", flush=True)
            rows.append(
                replay_one(
                    profile_counts,
                    eval_counts,
                    microbatch_size,
                    n_gpus,
                    trace_meta["n_shared_experts"],
                    config,
                )
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.outdir / "results")
    plot_routing_characterization(rows, args.outdir / "figures")
    plot_dynamic_gain(rows, args.outdir / "figures")
    plot_balance_components(rows, args.outdir / "figures")

    run_config = {
        "experiment": "DeepSeek-V2-Lite real-trace no-pruning replay",
        "trace_path": str(args.trace),
        "trace_sha256": sha256(args.trace),
        "trace_metadata": trace_meta,
        "replay": asdict(config),
        "derived": {
            "expert_weight_bytes": config.expert_weight_bytes,
            "expert_read_ms": config.expert_read_ms,
            "expert_token_flops": config.expert_token_flops,
            "expert_token_compute_ms": config.expert_token_compute_ms,
            "expert_read_to_token_compute_ratio": (
                config.expert_read_ms / config.expert_token_compute_ms
            ),
            "profile_sequences": profile_n,
            "evaluation_sequences": n_seq - profile_n,
            "microbatch_sizes": list(args.microbatch_sizes),
            "moe_gpu_counts": list(args.moe_gpus),
        },
        "scope": {
            "pruning": False,
            "attention": False,
            "network_communication": False,
            "scheduler_overhead": False,
            "hot_expert_splitting": False,
            "result_kind": "trace-driven analytical MoE load-balance replay",
        },
    }
    (args.outdir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    write_readme(args.outdir, args.trace, trace_meta, config, rows)
    print(f"wrote isolated experiment under {args.outdir}")


if __name__ == "__main__":
    main()
