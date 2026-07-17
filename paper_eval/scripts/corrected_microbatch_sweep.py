#!/usr/bin/env python3
"""Per-microbatch correction to the HBM-attention/HBF-MoE analytical sweep.

Unlike corrected_ctx_workload_sweep.py, routing, pruning, expert assignment, HBF
reads, and MoE computation are evaluated independently for every microbatch.
There is no hot-expert HBM tier and no persistent cross-microbatch expert cache:
HBM and HBF both use 1024 GB/s, and every active expert in every microbatch pays
an HBF read.

For one microbatch, attention and MoE are serial.  With >=2 microbatches, the
58 alternating attention/MoE layers reach the dependency-pipeline steady-state
bound max(total attention-pool work, total MoE-pool work), as validated by the
repository's dependency-driven pipeline simulator.  Weight reads are still
charged once per microbatch, so excessive microbatching is not free.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import corrected_ctx_workload_sweep as base  # noqa: E402


MICROBATCH_COUNTS = (1, 2, 4, 8, 16)
MAX_MOE_GPUS = 32
MAX_TOTAL_GPUS = 128


@dataclass(frozen=True)
class MicrobatchMoeProfile:
    execution_ms: float
    aggregate_visit_bytes: float
    mean_retained_experts: float
    mean_routes_per_token: float

    def work_ms(self, attention_gpus: int, moe_gpus: int) -> float:
        links = max(1, min(attention_gpus, moe_gpus))
        return self.execution_ms + base.bytes_to_ms(
            self.aggregate_visit_bytes, links * base.LINK_BW_GBPS
        )


@dataclass(frozen=True)
class MBPoint:
    context: int
    sessions: int
    retained_mass: float
    system: str
    gpus: int
    attention_gpus: int
    moe_gpus: int
    microbatches: int
    attention_ms: float
    moe_ms: float
    tbt_ms: float
    mean_retained_experts: float
    mean_routes_per_token: float


def microbatch_slices(sessions: int, microbatches: int) -> tuple[tuple[int, int], ...]:
    assert 1 <= microbatches <= sessions
    quotient, remainder = divmod(sessions, microbatches)
    bounds = []
    start = 0
    for index in range(microbatches):
        size = quotient + (1 if index < remainder else 0)
        bounds.append((start, start + size))
        start += size
    assert start == sessions
    return tuple(bounds)


def slice_routing(layer: base.LayerRouting, start: int, end: int) -> base.LayerRouting:
    return base.LayerRouting(layer.routes[start:end], layer.gate_weights[start:end])


@lru_cache(maxsize=1)
def microbatch_pruned_trials(
    sessions: int, retained_mass_milli: int, microbatches: int
) -> tuple[tuple[tuple[base.PrunedLayer, ...], ...], ...]:
    """Return trial -> layer -> microbatch, pruned independently per microbatch."""
    retained_mass = retained_mass_milli / 1000.0
    slices = microbatch_slices(sessions, microbatches)
    trials = []
    for trial in base.routing_samples(sessions):
        layers = []
        for layer in trial:
            layers.append(
                tuple(
                    base.prune_layer(slice_routing(layer, start, end), retained_mass)
                    for start, end in slices
                )
            )
        trials.append(tuple(layers))
    return tuple(trials)


@lru_cache(maxsize=None)
def microbatch_moe_profile(
    sessions: int,
    retained_mass_milli: int,
    microbatches: int,
    n_gpus: int,
    dynamic: bool,
) -> MicrobatchMoeProfile:
    """Total MoE-pool work for one decode iteration, summed over microbatches."""
    trial_execution = []
    trial_visit_bytes = []
    retained_counts = []
    routes_per_token = []
    slices = microbatch_slices(sessions, microbatches)
    mb_sizes = [end - start for start, end in slices]

    for trial in microbatch_pruned_trials(
        sessions, retained_mass_milli, microbatches
    ):
        execution = 0.0
        visit_bytes = 0.0
        for layer_index, layer_microbatches in enumerate(trial):
            for mb_index, layer in enumerate(layer_microbatches):
                owner = (
                    base.dynamic_lpt_owner(layer, n_gpus, base.BW_HBF_GBPS)
                    if dynamic
                    else base.fixed_owner(layer_index, n_gpus)
                )
                loads = [0.0] * n_gpus
                for expert in layer.retained:
                    read = base.bytes_to_ms(
                        base.PER_EXPERT_BYTES, base.BW_HBF_GBPS
                    )
                    compute = base.flops_to_ms(
                        layer.token_counts[expert]
                        * 6
                        * base.D
                        * base.MOE_INTERMEDIATE
                    )
                    loads[owner[expert]] += read + compute

                # Shared expert is replicated and read once on every GPU for
                # every microbatch. It is not retained as a hot HBM object.
                shared = base.bytes_to_ms(
                    base.SHARED_WEIGHT_BYTES_LAYER, base.BW_HBF_GBPS
                )
                shared += base.flops_to_ms(
                    (mb_sizes[mb_index] / n_gpus)
                    * 6
                    * base.D
                    * base.MOE_INTERMEDIATE
                )
                execution += max(loads) + shared

                visits = 0
                retained_routes = 0
                for token_route in layer.routes:
                    visits += len({owner[expert] for expert in token_route})
                    retained_routes += len(token_route)
                visit_bytes += 2 * visits * base.D * base.ACT_BYTES
                retained_counts.append(len(layer.retained))
                routes_per_token.append(retained_routes / mb_sizes[mb_index])
        trial_execution.append(execution)
        trial_visit_bytes.append(visit_bytes)

    return MicrobatchMoeProfile(
        execution_ms=float(np.mean(trial_execution)),
        aggregate_visit_bytes=float(np.mean(trial_visit_bytes)),
        mean_retained_experts=float(np.mean(retained_counts)),
        mean_routes_per_token=float(np.mean(routes_per_token)),
    )


def attention_work_ms(
    context: int,
    sessions: int,
    attention_gpus: int,
    microbatches: int,
    bw_gbps: float,
) -> float:
    """Total attention-pool work; model weights are reread per microbatch."""
    total = 0.0
    for start, end in microbatch_slices(sessions, microbatches):
        batch_per_gpu = math.ceil((end - start) / attention_gpus)
        total += base.attention_stage_ms(context, batch_per_gpu, bw_gbps)
    return total


def pipeline_tbt(attention_work: float, moe_work: float, microbatches: int) -> float:
    if microbatches == 1:
        return attention_work + moe_work
    return max(attention_work, moe_work)


def colocated_point(
    context: int,
    sessions: int,
    n_gpus: int,
    microbatches: int,
    hbf: bool,
) -> MBPoint | None:
    if not 1 <= n_gpus <= base.N_EXPERTS:
        return None
    batch_per_gpu = math.ceil(sessions / n_gpus)
    cap = base.HBF_CAP_GB if hbf else base.HBM_CAP_GB
    bw = base.BW_HBF_GBPS if hbf else base.BW_HBM_GBPS
    if not base.colocated_capacity_ok(context, batch_per_gpu, n_gpus, cap):
        return None
    attention = attention_work_ms(
        context, sessions, n_gpus, microbatches, bw
    )
    profile = microbatch_moe_profile(
        sessions, 1000, microbatches, n_gpus, False
    )
    moe = profile.work_ms(n_gpus, n_gpus)
    # Same GPUs execute both stages: no cross-pool overlap.
    return MBPoint(
        context=context,
        sessions=sessions,
        retained_mass=1.0,
        system=base.SYSTEM_HBF if hbf else base.SYSTEM_HBM,
        gpus=n_gpus,
        attention_gpus=n_gpus,
        moe_gpus=n_gpus,
        microbatches=microbatches,
        attention_ms=attention,
        moe_ms=moe,
        tbt_ms=attention + moe,
        mean_retained_experts=profile.mean_retained_experts,
        mean_routes_per_token=profile.mean_routes_per_token,
    )


def heterogeneous_point(
    context: int,
    sessions: int,
    retained_mass: float,
    total_gpus: int,
    microbatches: int,
    moe_gpus: int,
    dynamic: bool,
) -> MBPoint | None:
    attention_gpus = total_gpus - moe_gpus
    if attention_gpus < 1 or moe_gpus < 1:
        return None
    if not base.moe_capacity_ok(moe_gpus, dynamic):
        return None
    capacity = base.attention_capacity_sessions(context, base.HBM_CAP_GB)
    if capacity < 1 or math.ceil(sessions / attention_gpus) > capacity:
        return None

    attention = attention_work_ms(
        context,
        sessions,
        attention_gpus,
        microbatches,
        base.BW_HBM_GBPS,
    )
    profile = microbatch_moe_profile(
        sessions,
        int(round(retained_mass * 1000)),
        microbatches,
        moe_gpus,
        dynamic,
    )
    moe = profile.work_ms(attention_gpus, moe_gpus)
    return MBPoint(
        context=context,
        sessions=sessions,
        retained_mass=retained_mass,
        system=base.SYSTEM_DYNAMIC if dynamic else base.SYSTEM_STATIC,
        gpus=total_gpus,
        attention_gpus=attention_gpus,
        moe_gpus=moe_gpus,
        microbatches=microbatches,
        attention_ms=attention,
        moe_ms=moe,
        tbt_ms=pipeline_tbt(attention, moe, microbatches),
        mean_retained_experts=profile.mean_retained_experts,
        mean_routes_per_token=profile.mean_routes_per_token,
    )


def precompute_session(sessions: int) -> None:
    """Build compact profiles, dropping large pruned-route objects after each c/B."""
    print(f"precomputing per-microbatch profiles: S={sessions}", flush=True)
    for retained in base.RETAINED_MASS:
        milli = int(round(retained * 1000))
        for microbatches in MICROBATCH_COUNTS:
            for n_gpus in range(1, MAX_MOE_GPUS + 1):
                microbatch_moe_profile(
                    sessions, milli, microbatches, n_gpus, False
                )
                microbatch_moe_profile(
                    sessions, milli, microbatches, n_gpus, True
                )
            # Colocated baselines can use more than MAX_MOE_GPUS.
            if retained == 1.0:
                for n_gpus in range(MAX_MOE_GPUS + 1, MAX_TOTAL_GPUS + 1):
                    microbatch_moe_profile(
                        sessions, milli, microbatches, n_gpus, False
                    )
            microbatch_pruned_trials.cache_clear()
    base.routing_samples.cache_clear()


def enumerate_points() -> list[MBPoint]:
    points = []
    for sessions in base.CONCURRENCIES:
        precompute_session(sessions)
        for context in base.CONTEXTS:
            print(f"  enumerating C={context // 1024}K", flush=True)
            for total_gpus in range(2, MAX_TOTAL_GPUS + 1):
                for microbatches in MICROBATCH_COUNTS:
                    for hbf in (False, True):
                        point = colocated_point(
                            context, sessions, total_gpus, microbatches, hbf
                        )
                        if point:
                            points.append(point)
                    max_m = min(MAX_MOE_GPUS, total_gpus - 1)
                    for retained in base.RETAINED_MASS:
                        for dynamic in (False, True):
                            for moe_gpus in range(1, max_m + 1):
                                point = heterogeneous_point(
                                    context,
                                    sessions,
                                    retained,
                                    total_gpus,
                                    microbatches,
                                    moe_gpus,
                                    dynamic,
                                )
                                if point:
                                    points.append(point)
    return points


def group_points(points: Sequence[MBPoint]):
    grouped = {}
    for point in points:
        grouped.setdefault(
            (point.context, point.sessions, point.retained_mass, point.system), []
        ).append(point)
    return grouped


def best_within(points: Iterable[MBPoint], budget: int) -> MBPoint | None:
    eligible = [point for point in points if point.gpus <= budget]
    return min(
        eligible,
        key=lambda p: (p.tbt_ms, p.gpus, p.microbatches, p.moe_gpus),
    ) if eligible else None


def min_slo(points: Iterable[MBPoint], slo_ms: float) -> MBPoint | None:
    eligible = [point for point in points if point.tbt_ms <= slo_ms]
    return min(
        eligible,
        key=lambda p: (p.gpus, p.tbt_ms, p.microbatches),
    ) if eligible else None


def choose_budgets(grouped) -> dict[int, int]:
    budgets = {}
    for sessions in base.CONCURRENCIES:
        candidates = grouped[(32768, sessions, 1.0, base.SYSTEM_STATIC)]
        point = min_slo(candidates, 100.0)
        chosen = point.gpus if point else min(candidates, key=lambda p: p.tbt_ms).gpus
        budgets[sessions] = min(MAX_TOTAL_GPUS, max(4, math.ceil(chosen / 4) * 4))
    return budgets


def write_csv(points: Sequence[MBPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "context_tokens", "context_k", "sessions", "retained_mass", "system",
        "gpus", "attention_gpus", "moe_gpus", "microbatches", "attention_ms",
        "moe_ms", "tbt_ms", "mean_retained_experts", "mean_routes_per_token",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow({
                "context_tokens": p.context,
                "context_k": p.context // 1024,
                "sessions": p.sessions,
                "retained_mass": f"{p.retained_mass:.2f}",
                "system": p.system,
                "gpus": p.gpus,
                "attention_gpus": p.attention_gpus,
                "moe_gpus": p.moe_gpus,
                "microbatches": p.microbatches,
                "attention_ms": f"{p.attention_ms:.6f}",
                "moe_ms": f"{p.moe_ms:.6f}",
                "tbt_ms": f"{p.tbt_ms:.6f}",
                "mean_retained_experts": f"{p.mean_retained_experts:.6f}",
                "mean_routes_per_token": f"{p.mean_routes_per_token:.6f}",
            })


def fmt(point: MBPoint | None) -> str:
    if point is None:
        return "—"
    if point.system in (base.SYSTEM_HBM, base.SYSTEM_HBF):
        return f"{point.tbt_ms:.1f} ({point.gpus}G, μ{point.microbatches})"
    return (
        f"{point.tbt_ms:.1f} "
        f"({point.attention_gpus}A:{point.moe_gpus}M, μ{point.microbatches})"
    )


def write_markdown(points: Sequence[MBPoint], path: Path, budgets: dict[int, int]) -> None:
    grouped = group_points(points)
    lines = [
        "# Corrected per-microbatch context × workload sweep",
        "",
        "> Routing, pruning, assignment, HBF reads, and MoE compute are performed independently per microbatch.",
        "> HBM and HBF both use 1024 GB/s. There is no hot-expert HBM tier and no cross-microbatch expert cache.",
        "> Routing and pruning remain synthetic; results are analytical rather than measured.",
        "",
        "## Semantics",
        "",
        f"- Microbatch counts: `{', '.join(map(str, MICROBATCH_COUNTS))}`.",
        "- Every active/retained expert is read from HBF again for every microbatch that uses it.",
        "- Pruning and integer expert assignment see only the current microbatch.",
        "- One microbatch is serial; two or more use the dependency-pipeline steady-state `max(A-work, M-work)`.",
        "- Attention weights are also reread per microbatch, preventing free over-microbatching.",
        "",
        "## Fixed budgets",
        "",
        "Selected from the unpruned balanced-static heterogeneous baseline at 32K/100 ms, without observing dynamic results.",
        "",
        "| Sessions | Budget |",
        "|---:|---:|",
    ]
    for sessions, budget in budgets.items():
        lines.append(f"| {sessions} | {budget} |")

    lines.extend([
        "",
        "## Best result after sweeping microbatch count",
        "",
        "`μN` is the selected number of microbatches.",
        "",
        "| Ctx | S | Budget | HBM coloc | HBF coloc | Static c=1 | Dynamic c=1 | Static c=.98 | Dynamic c=.98 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for context in base.CONTEXTS:
        for sessions in base.CONCURRENCIES:
            budget = budgets[sessions]
            cells = []
            for retained, system in (
                (1.0, base.SYSTEM_HBM),
                (1.0, base.SYSTEM_HBF),
                (1.0, base.SYSTEM_STATIC),
                (1.0, base.SYSTEM_DYNAMIC),
                (0.98, base.SYSTEM_STATIC),
                (0.98, base.SYSTEM_DYNAMIC),
            ):
                cells.append(
                    fmt(best_within(grouped.get((context, sessions, retained, system), []), budget))
                )
            lines.append(
                f"| {context // 1024}K | {sessions} | {budget} | "
                + " | ".join(cells)
                + " |"
            )

    for slo in base.SLOS_MS:
        lines.extend([
            "",
            f"## Minimum GPUs at {slo:.0f} ms after sweeping microbatch count",
            "",
            "| Ctx | S | HBM coloc | HBF coloc | Static c=1 | Dynamic c=1 | Static c=.98 | Dynamic c=.98 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for context in base.CONTEXTS:
            for sessions in base.CONCURRENCIES:
                cells = []
                for retained, system in (
                    (1.0, base.SYSTEM_HBM),
                    (1.0, base.SYSTEM_HBF),
                    (1.0, base.SYSTEM_STATIC),
                    (1.0, base.SYSTEM_DYNAMIC),
                    (0.98, base.SYSTEM_STATIC),
                    (0.98, base.SYSTEM_DYNAMIC),
                ):
                    point = min_slo(
                        grouped.get((context, sessions, retained, system), []), slo
                    )
                    cells.append("—" if point is None else f"{point.gpus} (μ{point.microbatches})")
                lines.append(
                    f"| {context // 1024}K | {sessions} | "
                    + " | ".join(cells)
                    + " |"
                )

    lines.extend([
        "",
        "## Microbatch sensitivity at fixed budgets (dynamic c=.98)",
        "",
        "| μbatches | Median TBT (ms) | Median retained experts/μbatch | Feasible cells |",
        "|---:|---:|---:|---:|",
    ])
    for microbatches in MICROBATCH_COUNTS:
        tbts = []
        retained = []
        for context in base.CONTEXTS:
            for sessions in base.CONCURRENCIES:
                budget = budgets[sessions]
                candidates = [
                    p
                    for p in grouped[(context, sessions, 0.98, base.SYSTEM_DYNAMIC)]
                    if p.gpus <= budget and p.microbatches == microbatches
                ]
                if candidates:
                    point = min(candidates, key=lambda p: p.tbt_ms)
                    tbts.append(point.tbt_ms)
                    retained.append(point.mean_retained_experts)
        lines.append(
            f"| {microbatches} | {np.median(tbts):.1f} | "
            f"{np.median(retained):.1f} | {len(tbts)} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- No cross-microbatch expert reuse; this intentionally follows the equal-bandwidth/no-hot-tier requirement.",
        "- The two-or-more-microbatch overlap is a steady-state bound; control and scheduling overhead remain zero.",
        "- Pruning quality is not measured, and routing is synthetic.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def validate() -> None:
    assert base.BW_HBM_GBPS == base.BW_HBF_GBPS == 1024.0
    assert microbatch_slices(32, 16)[0] == (0, 2)
    assert sum(end - start for start, end in microbatch_slices(33, 8)) == 33
    assert pipeline_tbt(10, 20, 1) == 30
    assert pipeline_tbt(10, 20, 2) == 20
    # Rereading weights makes total attention work nondecreasing with μbatch count.
    one = attention_work_ms(8192, 128, 8, 1, 1024.0)
    two = attention_work_ms(8192, 128, 8, 2, 1024.0)
    assert two >= one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR.parent / "results"),
    )
    args = parser.parse_args()
    validate()
    points = enumerate_points()
    grouped = group_points(points)
    budgets = choose_budgets(grouped)
    outdir = Path(args.output_dir)
    csv_path = outdir / "corrected_microbatch_sweep.csv"
    md_path = outdir / "corrected_microbatch_sweep.md"
    write_csv(points, csv_path)
    write_markdown(points, md_path, budgets)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
