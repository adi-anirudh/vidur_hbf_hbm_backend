"""
Shared MoE-on-flash timing math.

Single source of truth for the sparse-MoE expert-weight-on-HBF model used by
both:
  * attn_moe_batch_sweep.py            (single-GPU microbenchmark, "path B")
  * HBFLinearRegressionExecutionTimePredictor  (full cluster simulator, "path A")

Every formula here is lifted verbatim from attn_moe_batch_sweep.py so the two
paths cannot drift. Model:

  * Expert weights (gated SwiGLU: gate + up + down) live on HBF flash.
  * A decode step with batch B activates E_unique(B) distinct experts per
    layer (uniform-random or worst-case routing); their weights are streamed
    from flash.
  * MoE phase per layer = combine(flash load, expert GEMM compute), where
    combine is `+` (serial) or `max` (overlapped).

Flash timing has two variants, mirroring the sweep script:
  * flash_time_bw_ms    — conservative bandwidth bound (bytes / peak BW).
                          The sweep's plotted headline.
  * flash_time_sched_ms — the cycle-accurate NAND plane-scheduler shortcut
                          (submit_plane_reads); optimistic tRC-streaming
                          reference, identical to the KV-read path in
                          hbf_execution_time_predictor.
"""

from __future__ import annotations

import math


# ──────────────────────────────────────────────────────────────────────────────
# Routing: expected number of unique experts activated by B tokens (top_k of E)
# ──────────────────────────────────────────────────────────────────────────────

def unique_experts_uniform(B: float, E: int, k: int) -> float:
    """Expected distinct experts under uniform-random routing (balls in bins)."""
    return E * (1.0 - (1.0 - k / E) ** B)


def unique_experts_worst(B: float, E: int, k: int) -> float:
    """Adversarial routing: every token hits fresh experts until E saturates."""
    return min(B * k, E)


# ──────────────────────────────────────────────────────────────────────────────
# Byte / FLOP constants (gated SwiGLU expert: gate(D→He) + up(D→He) + down(He→D))
# ──────────────────────────────────────────────────────────────────────────────

def per_expert_bytes(D: int, He: int, be: int) -> int:
    """Flash bytes of one expert's weights: (2*D*He + He*D) * be = 3*D*He*be."""
    return (2 * D * He + He * D) * be


def expert_flops_per_token(D: int, He: int) -> int:
    """Expert GEMM FLOPs per routed token = 2*MACs over gate+up+down."""
    return 6 * D * He


def moe_flash_bytes(
    B: float,
    E: int,
    k: int,
    D: int,
    He: int,
    be: int,
    shared_He: int = 0,
    worst: bool = False,
) -> float:
    """Total flash bytes for one MoE layer at decode batch B: routed unique
    experts plus the always-on shared expert (0 when shared_He == 0)."""
    n_unique = unique_experts_worst(B, E, k) if worst else unique_experts_uniform(B, E, k)
    return n_unique * per_expert_bytes(D, He, be) + per_expert_bytes(D, shared_He, be)


def moe_compute_ms(
    B: float,
    k: int,
    D: int,
    He: int,
    flops_per_ms: float,
    shared_He: int = 0,
) -> float:
    """Expert GEMM time per MoE layer: routed (B*k experts-worth of tokens)
    plus the always-on shared expert applied to every token."""
    routed = B * k * expert_flops_per_token(D, He)
    shared = B * expert_flops_per_token(D, shared_He)
    return (routed + shared) / flops_per_ms


def flops_per_ms(fp16_tflops: float) -> float:
    """Peak FLOPs per millisecond for a device SKU."""
    return fp16_tflops * 1e12 / 1e3


def combine(flash_ms: float, compute_ms: float, overlap: bool) -> float:
    """Serial (flash + compute) or perfectly overlapped (max)."""
    return max(flash_ms, compute_ms) if overlap else flash_ms + compute_ms


# ──────────────────────────────────────────────────────────────────────────────
# Flash timing via the repo's NAND plane scheduler / bandwidth bound
# ──────────────────────────────────────────────────────────────────────────────

def flash_time_bw_ms(backend, total_bytes: float) -> float:
    """Conservative bound: total_bytes / peak flash bandwidth (tR-bound).
    bytes / (GB/s = bytes/ns) -> ns -> ms."""
    if total_bytes <= 0:
        return 0.0
    bpns = backend.peak_bandwidth_gbps()  # bytes/ns
    return (total_bytes / bpns) / 1e6


def flash_time_sched_ms(backend, total_bytes: float) -> float:
    """Time to read `total_bytes` from flash, striped uniformly across all
    planes, via the cycle-accurate plane scheduler (submit_plane_reads
    shortcut).

    NOTE: this shortcut issues all per-plane pages sequentially from block 0,
    which takes the page-buffer cache-read fast path (tRC) for pages within a
    block, so the *effective* bandwidth exceeds the tR-bound headline. It is
    exactly what hbf_execution_time_predictor uses for KV reads. Use
    flash_time_bw_ms() for the conservative bandwidth-bound cross-check."""
    if total_bytes <= 0:
        return 0.0
    page = backend._cfg.subarray.page_size_bytes
    total_planes = backend._mapper.total_planes
    pages = max(1, math.ceil(total_bytes / page))
    per_plane = max(1, math.ceil(pages / total_planes))
    plane_pages = {p: per_plane for p in range(total_planes)}
    backend.submit_plane_reads(plane_pages, token_id=0)
    comps = backend.drain()
    if not comps:
        return 0.0
    return max(c.latency_ns for c in comps) / 1e6
