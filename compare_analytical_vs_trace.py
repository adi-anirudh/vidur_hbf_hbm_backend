# -*- coding: utf-8 -*-
"""
compare_analytical_vs_trace.py

Measures the error of the predictor's *analytical* decode-step flash model
(one aggregate read per plane, submit_plane_reads) against the *trace-level*
cycle-accurate reference (one read per (seq, layer, KV-block), submit_decode_step)
through the SAME PlaneScheduler.

Both paths share the scheduler, so the only difference is the analytical
aggregation. Goal: drive the relative error below 1%.
"""
from __future__ import annotations

import math
from vidur.memory_backends.hbf.backend import HBFSimBackend
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy
from vidur.memory_backends.base import MemRequest, OpType, DataClass

TOML = "configs/hbf_paper.toml"

# (label, n_kv, h_dim, n_lay)
MODELS = [
    ("Llama-3-8B", 8, 128, 32),
    ("Llama-2-70b", 8, 128, 80),
    ("phi-2",       32, 80, 32),
    ("Qwen-72B",    64, 128, 80),
]
BATCHES   = [4, 8, 32]
CONTEXTS  = [4096, 16384, 65536]
SPARSITY  = [1.0, 0.1]
BLOCK_TOKENS = 16

# Trace reference submits one read per (seq, layer, selected-block); cap the
# total so the per-block drain stays tractable. Larger configs are skipped.
MAX_TRACE_READS = 200_000


def kv_block_bytes(n_kv, h_dim):
    per_tok = 2 * n_kv * h_dim * 2          # K+V, fp16
    return BLOCK_TOKENS * per_tok           # bytes per KV block, per layer


def make_backend(num_layers):
    return HBFSimBackend(
        config_path=TOML,
        placement_policy=PlacementPolicy.STRIPE_ACROSS_PLANES,
        num_layers=num_layers,
    )


def trace_level_ms(n_kv, h_dim, n_lay, batch, ctx, sparsity):
    """Per-(seq, layer, block) reads with real STRIPE mapping. Returns ms."""
    be = make_backend(n_lay)
    blk_bytes = kv_block_bytes(n_kv, h_dim)
    n_blocks  = max(1, math.ceil(ctx / BLOCK_TOKENS))
    k_sel     = max(1, math.ceil(n_blocks * sparsity))   # top-K blocks per seq/layer
    gid = 0
    for seq in range(batch):
        for layer in range(n_lay):
            for j in range(k_sel):
                # distinct global block id → STRIPE spreads across planes,
                # each KV block lands in its own NAND block.
                be.submit(MemRequest(
                    req_id=be._next_req_id(),
                    op=OpType.READ,
                    size_bytes=blk_bytes,
                    data_class=DataClass.COLD_KV,
                    layer_id=layer,
                    sequence_id=seq,
                    token_id=0,
                    block_id=gid,
                ))
                gid += 1
    comps = be.drain()
    return (max(c.latency_ns for c in comps) / 1e6) if comps else 0.0


def analytical_ms(n_kv, h_dim, n_lay, batch, ctx, sparsity):
    """Replicates predictor baseline branch (hbm_kv_fraction == 0). Returns ms."""
    be = make_backend(n_lay)
    blk_bytes    = kv_block_bytes(n_kv, h_dim)
    page_size    = be._cfg.subarray.page_size_bytes
    total_planes = be._mapper.total_planes
    n_blocks     = max(1, math.ceil(ctx / BLOCK_TOKENS))

    # Per-request top-K: each of `batch` sequences selects ceil(n_blocks*sparsity)
    # blocks per layer, then stripe the aggregate across planes (matches predictor).
    k_per_seq = max(1, math.ceil(n_blocks * sparsity))
    total_selected_blocks = batch * k_per_seq * n_lay
    blocks_per_plane_sparse = max(1, math.ceil(total_selected_blocks / total_planes))
    pages_per_plane = max(1, math.ceil(blocks_per_plane_sparse * blk_bytes / page_size))
    plane_pages  = {p: pages_per_plane for p in range(total_planes)}
    plane_blocks = {p: blocks_per_plane_sparse for p in range(total_planes)}
    be.submit_plane_reads(plane_pages, token_id=0, plane_block_counts=plane_blocks)
    comps = be.drain()
    return (max(c.latency_ns for c in comps) / 1e6) if comps else 0.0


def main():
    print(f"{'model':<12} {'B':>4} {'ctx':>7} {'spars':>6} "
          f"{'trace_ms':>10} {'anly_ms':>10} {'err_%':>9}")
    print("-" * 64)
    worst = 0.0
    for (label, n_kv, h_dim, n_lay) in MODELS:
        for batch in BATCHES:
            for ctx in CONTEXTS:
                for sp in SPARSITY:
                    n_blocks = max(1, math.ceil(ctx / BLOCK_TOKENS))
                    k_sel    = max(1, math.ceil(n_blocks * sp))
                    n_reads  = batch * n_lay * k_sel
                    if n_reads > MAX_TRACE_READS:
                        print(f"{label:<12} {batch:>4} {ctx:>7} {sp:>6.2f} "
                              f"{'(skip: '+str(n_reads)+' reads)':>32}")
                        continue
                    t = trace_level_ms(n_kv, h_dim, n_lay, batch, ctx, sp)
                    a = analytical_ms(n_kv, h_dim, n_lay, batch, ctx, sp)
                    err = (a - t) / t * 100.0 if t > 0 else 0.0
                    worst = max(worst, abs(err))
                    print(f"{label:<12} {batch:>4} {ctx:>7} {sp:>6.2f} "
                          f"{t:>10.4f} {a:>10.4f} {err:>+8.2f}%")
    print("-" * 64)
    print(f"worst |error| = {worst:.2f}%")


if __name__ == "__main__":
    main()
