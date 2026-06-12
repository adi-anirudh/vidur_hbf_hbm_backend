# -*- coding: utf-8 -*-
"""
stress_analytical_vs_trace.py

Adversarial stress test of the analytical decode model vs the trace-level sim.

Unlike compare_analytical_vs_trace.py (which uses idealised, model-friendly
inputs), this varies the access pattern in ways that can break the analytical
model's core assumption of perfectly uniform per-plane load:

  * block-id allocation:  contiguous-unique | per-seq-contiguous | scattered | recent-window
  * context lengths:      uniform | ragged (per-sequence)
  * token_id / row_offset: zero | per-seq-varied
  * sparsity:             0.03 .. 1.0
  * KV block size:        normal | huge (pages_per_kv_block > pages_per_block)

For each scenario we run BOTH paths through the same PlaneScheduler and report
the relative error, plus the realised per-plane load imbalance (max/mean), which
is the mechanism that drives any divergence.
"""
from __future__ import annotations

import math
import random
from vidur.memory_backends.hbf.backend import HBFSimBackend
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy
from vidur.memory_backends.base import MemRequest, OpType, DataClass

TOML = "configs/hbf_paper.toml"
BLOCK_TOKENS = 16
MAX_TRACE_READS = 250_000


def make_backend(num_layers, policy=PlacementPolicy.STRIPE_ACROSS_PLANES):
    return HBFSimBackend(config_path=TOML, placement_policy=policy,
                         num_layers=num_layers)


def kv_block_bytes(n_kv, h_dim):
    return BLOCK_TOKENS * (2 * n_kv * h_dim * 2)


def build_reads(n_kv, h_dim, n_lay, seq_ctx, sparsity, alloc, rng,
                token_mode):
    """
    Produce the list of (seq, layer, block_id, token_id) reads for one decode
    step, plus per-(seq) selected-block counts.

    seq_ctx: list of per-sequence context lengths (ragged allowed).
    alloc:   how global block ids are assigned.
    token_mode: 'zero' or 'varied' row offsets.
    """
    reads = []
    total_selected_per_layer = 0
    next_global = 0
    # Pre-reserve a contiguous id range per (seq) so allocation schemes differ.
    seq_ranges = []
    for ctx in seq_ctx:
        nb = max(1, math.ceil(ctx / BLOCK_TOKENS))
        seq_ranges.append((next_global, nb))
        next_global += nb
    pool_size = max(1, next_global)

    for s, ctx in enumerate(seq_ctx):
        base, nb = seq_ranges[s]
        k = max(1, math.ceil(nb * sparsity))
        total_selected_per_layer += k
        # choose which of the seq's nb blocks are selected (positions)
        if alloc == "recent":
            sel_local = list(range(nb - k, nb))            # most-recent window
        elif alloc == "scattered":
            sel_local = rng.sample(range(nb), k)           # random positions
        else:
            sel_local = list(range(k))                     # first-k
        tok = 0 if token_mode == "zero" else rng.randrange(0, 4096)
        for layer in range(n_lay):
            for loc in sel_local:
                if alloc == "contiguous_unique":
                    # every (seq,layer,block) gets a globally unique id
                    bid = len(reads)
                elif alloc == "perseq_contiguous":
                    # layers reuse the seq's id range -> same planes per layer
                    bid = base + loc
                elif alloc == "scattered":
                    bid = (base + loc) * 7 + layer * 131  # spread-ish, non-uniform
                else:  # recent / first-k -> per-seq contiguous global ids
                    bid = base + loc
                reads.append((s, layer, bid, tok))
    return reads, total_selected_per_layer


def trace_ms(reads, blk_bytes, n_lay, policy):
    be = make_backend(n_lay, policy)
    for (s, layer, bid, tok) in reads:
        be.submit(MemRequest(
            req_id=be._next_req_id(), op=OpType.READ, size_bytes=blk_bytes,
            data_class=DataClass.COLD_KV, layer_id=layer, sequence_id=s,
            token_id=tok, block_id=bid))
    # realised per-plane load (block count) for imbalance diagnostics
    plane_load = {}
    for (s, layer, bid, tok) in reads:
        addr = be._mapper.map_block(block_id=bid, layer_id=layer,
                                    sequence_id=s, token_id=tok)
        plane_load[addr.plane_id] = plane_load.get(addr.plane_id, 0) + 1
    comps = be.drain()
    lat = (max(c.latency_ns for c in comps) / 1e6) if comps else 0.0
    total_planes = be._mapper.total_planes
    mx = max(plane_load.values()) if plane_load else 0
    mean = len(reads) / total_planes
    return lat, (mx / mean if mean > 0 else 1.0)


def analytical_ms(total_selected_per_layer, n_lay, blk_bytes, num_layers_be):
    be = make_backend(num_layers_be)
    page_size    = be._cfg.subarray.page_size_bytes
    total_planes = be._mapper.total_planes
    total_selected_blocks = total_selected_per_layer * n_lay
    bpp = max(1, math.ceil(total_selected_blocks / total_planes))
    pages_per_plane = max(1, math.ceil(bpp * blk_bytes / page_size))
    plane_pages  = {p: pages_per_plane for p in range(total_planes)}
    plane_blocks = {p: bpp for p in range(total_planes)}
    be.submit_plane_reads(plane_pages, token_id=0, plane_block_counts=plane_blocks)
    comps = be.drain()
    return (max(c.latency_ns for c in comps) / 1e6) if comps else 0.0


def main():
    rng = random.Random(1234)
    # (label, n_kv, h_dim, n_lay)
    MODELS = [("Llama-3-8B", 8, 128, 32), ("Qwen-72B", 64, 128, 80),
              ("huge-blk", 8, 128, 8)]   # huge-blk uses big block size below
    BATCHES  = [4, 8, 16]
    CTX_BASE = [4096, 16384]
    SPARS    = [0.03, 0.1, 0.25, 0.5, 1.0]
    ALLOCS   = ["contiguous_unique", "perseq_contiguous", "scattered", "recent"]
    CTXMODE  = ["uniform", "ragged"]
    TOKMODE  = ["zero", "varied"]

    print(f"{'model':<11}{'alloc':<19}{'ctx':<8}{'tok':<7}{'B':>3}{'base':>7}"
          f"{'sp':>6}{'trace':>9}{'anly':>9}{'err%':>8}{'imbal':>7}")
    print("-" * 100)
    worst = 0.0
    worst_row = None
    rows = 0
    for (label, n_kv, h_dim, n_lay) in MODELS:
        blk_bytes = kv_block_bytes(n_kv, h_dim)
        if label == "huge-blk":
            blk_bytes = 2 * 1024 * 1024   # 2 MB KV block -> 512 pages > 256 ppb
        for alloc in ALLOCS:
            for ctxmode in CTXMODE:
                for tokmode in TOKMODE:
                    for batch in BATCHES:
                        for base in CTX_BASE:
                            for sp in SPARS:
                                if ctxmode == "uniform":
                                    seq_ctx = [base] * batch
                                else:
                                    seq_ctx = [rng.randint(base // 2, base)
                                               for _ in range(batch)]
                                # cheap pre-count to skip oversized traces
                                est = sum(max(1, math.ceil(
                                    max(1, math.ceil(c / BLOCK_TOKENS)) * sp))
                                    for c in seq_ctx) * n_lay
                                if est > MAX_TRACE_READS:
                                    continue
                                reads, tspl = build_reads(
                                    n_kv, h_dim, n_lay, seq_ctx, sp, alloc,
                                    rng, tokmode)
                                t, imbal = trace_ms(reads, blk_bytes, n_lay,
                                                    PlacementPolicy.STRIPE_ACROSS_PLANES)
                                a = analytical_ms(tspl, n_lay, blk_bytes, n_lay)
                                err = (a - t) / t * 100.0 if t > 0 else 0.0
                                rows += 1
                                if abs(err) > abs(worst):
                                    worst = err
                                    worst_row = (label, alloc, ctxmode, tokmode,
                                                 batch, base, sp, t, a, imbal)
                                if abs(err) > 1.0:
                                    print(f"{label:<11}{alloc:<19}{ctxmode:<8}"
                                          f"{tokmode:<7}{batch:>3}{base:>7}{sp:>6.2f}"
                                          f"{t:>9.4f}{a:>9.4f}{err:>+7.1f}%{imbal:>7.2f}")
    print("-" * 100)
    print(f"scenarios tested: {rows}")
    print(f"worst error: {worst:+.2f}%  ->  {worst_row}")


if __name__ == "__main__":
    main()
