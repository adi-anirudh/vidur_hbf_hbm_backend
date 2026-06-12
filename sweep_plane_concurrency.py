# -*- coding: utf-8 -*-
"""
sweep_plane_concurrency.py

Sweeps the degree of plane concurrency — from fully utilising all available
planes (reads spread across every plane, maximum parallelism) down to all
accesses landing on a single plane (fully serial) — and reports the error of
the fast analytical flash model vs the cycle-accurate PlaneScheduler.

Reads are realistic decode KV-block reads: uniform size (one KV block each),
distinct NAND block per read on a plane, token_id=0 (so multi-plane merging is
maximally active — the regime where merge-coupling could bite).

Pass --hetero to add size heterogeneity (mixes KV-block sizes) to probe the
merge-coupling skew directly.
"""
from __future__ import annotations

import argparse
import math

from vidur.memory_backends.base import OpType
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy, PlaneAddressMapper
from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader
from vidur.memory_backends.hbf.plane_scheduler import NANDRequest, PlaneScheduler
from vidur.memory_backends.hbf.fast_flash_model import fast_decode_latency_ns

TOML = "configs/hbf_paper.toml"


def scheduler_latency_ns(reads, cfg, mapper):
    sched = PlaneScheduler(die_cfg=cfg.nand_die, sa_cfg=cfg.subarray,
                           stack_cfg=cfg.nand_stack)
    rid = 0
    for r in reads:
        addr = mapper.map_block(block_id=max(0, r["block_id"]),
                                layer_id=r.get("layer_id", 0),
                                sequence_id=r.get("sequence_id", 0),
                                token_id=r.get("token_id", 0))
        sched.submit(NANDRequest(req_id=rid, op=r["op"], addr=addr,
                                 size_bytes=r["size_bytes"], issued_at=0.0,
                                 num_blocks=r.get("num_blocks", 0)))
        rid += 1
    comps = sched.drain()
    return max((c.latency_ns for c in comps), default=0.0)


def build_reads(n_reads, n_planes_used, total_planes, kv_pages, page_size,
                hetero, rng):
    """n_reads KV-block reads landing on exactly n_planes_used planes, each
    read on a distinct NAND block of its plane."""
    reads = []
    occ = [0] * n_planes_used      # reads already placed on each used plane
    for i in range(n_reads):
        p = i % n_planes_used
        j = occ[p]; occ[p] += 1
        block_id = p + total_planes * j      # plane p, distinct local block j
        pages = kv_pages
        if hetero:
            pages = rng.choice([kv_pages, kv_pages // 2 or 1, kv_pages * 2])
        reads.append(dict(op=OpType.READ, size_bytes=pages * page_size,
                          block_id=block_id, layer_id=0, sequence_id=0,
                          token_id=0, num_blocks=0))
    return reads


def main():
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=int, default=7680)
    ap.add_argument("--kv_pages", type=int, default=16)  # 64KB KV block
    ap.add_argument("--hetero", action="store_true")
    args = ap.parse_args()
    rng = random.Random(7)

    cfg = HBFSimConfigLoader(TOML).load()
    mapper = PlaneAddressMapper(sa_cfg=cfg.subarray, die_cfg=cfg.nand_die,
                                stack_cfg=cfg.nand_stack,
                                policy=PlacementPolicy.STRIPE_ACROSS_PLANES,
                                num_layers=64)
    total_planes = mapper.total_planes
    page_size = cfg.subarray.page_size_bytes

    # concurrency levels: full -> 1 (powers of two plus the exact total)
    levels = []
    p = total_planes
    while p >= 1:
        levels.append(p)
        p //= 2
    if levels[-1] != 1:
        levels.append(1)

    print(f"total_planes={total_planes}  reads={args.reads}  "
          f"kv_pages={args.kv_pages}  hetero={args.hetero}")
    print(f"{'planes_used':>12}{'reads/plane':>12}{'sched_ms':>11}"
          f"{'fast_ms':>11}{'err_%':>9}")
    print("-" * 56)
    worst = 0.0
    for lv in levels:
        reads = build_reads(args.reads, lv, total_planes, args.kv_pages,
                            page_size, args.hetero, rng)
        s = scheduler_latency_ns(reads, cfg, mapper) / 1e6
        f = fast_decode_latency_ns(reads, cfg, mapper) / 1e6
        err = (f - s) / s * 100.0 if s > 0 else 0.0
        worst = max(worst, abs(err))
        print(f"{lv:>12}{math.ceil(args.reads/lv):>12}{s:>11.4f}"
              f"{f:>11.4f}{err:>+8.2f}%")
    print("-" * 56)
    print(f"worst |error| across concurrency sweep = {worst:.3f}%")


if __name__ == "__main__":
    main()
