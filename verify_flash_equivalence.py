# -*- coding: utf-8 -*-
"""
verify_flash_equivalence.py

Exhaustively checks that the fast analytical flash model
(fast_decode_latency_ns) returns EXACTLY the same decode-step wall-clock as the
cycle-accurate PlaneScheduler.drain(), across adversarially-generated access
patterns. The goal is zero mismatches over the whole space.

Dimensions varied per random pattern:
  * number of reads (1 .. many)
  * block_id          (collisions, scatter, contiguous, shared-across-layers)
  * layer/sequence    (re-reads of the same block → tRC cache hits)
  * token_id          (row_offset → multi-plane merge groups)
  * size_bytes        (sub-page, single page, multi-page, > pages_per_block)
  * num_blocks tag    (aggregate per-plane reads)
  * submission order  (shuffled)
  * NAND config       (1 subarray/plane AND multi-subarray/plane geometries)

Any mismatch is printed with the exact reproducing pattern.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from vidur.memory_backends.base import MemRequest, OpType, DataClass
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy, PlaneAddressMapper
from vidur.memory_backends.hbf.config_loader import (
    HBFSimConfig, NANDDieConfig, NANDStackConfig, SubarrayConfig,
)
from vidur.memory_backends.hbf.plane_scheduler import NANDRequest, PlaneScheduler
from vidur.memory_backends.hbf.fast_flash_model import fast_decode_latency_ns


# ---------------------------------------------------------------------------
# Reference: cycle-accurate scheduler drain
# ---------------------------------------------------------------------------

def scheduler_latency_ns(reads, cfg, mapper):
    sched = PlaneScheduler(die_cfg=cfg.nand_die, sa_cfg=cfg.subarray,
                           stack_cfg=cfg.nand_stack)
    rid = 0
    for r in reads:
        addr = mapper.map_block(
            block_id=max(0, r["block_id"]), layer_id=r.get("layer_id", 0),
            sequence_id=r.get("sequence_id", 0), token_id=r.get("token_id", 0))
        sched.submit(NANDRequest(
            req_id=rid, op=r["op"], addr=addr, size_bytes=r["size_bytes"],
            issued_at=0.0, num_blocks=r.get("num_blocks", 0)))
        rid += 1
    comps = sched.drain()
    return max((c.latency_ns for c in comps), default=0.0)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_cfg(planes_per_die, dies_per_ch, channels, subarrays_per_plane,
              pages_per_block=256, page_size=4096, plane_cap=0,
              max_plane_batch=96, enable_mpc=True,
              policy=PlacementPolicy.STRIPE_ACROSS_PLANES):
    num_subarrays = planes_per_die * subarrays_per_plane
    blocks_per_plane = 256
    sa = SubarrayConfig(page_size_bytes=page_size, pages_per_block=pages_per_block,
                        tR_ns=4096.0, tRC_ns=30.0)
    die = NANDDieConfig(
        num_planes=planes_per_die, num_subarrays=num_subarrays,
        blocks_per_plane=blocks_per_plane,
        max_concurrent_per_plane=plane_cap,
        enable_multi_plane_cmds=enable_mpc, max_plane_batch=max_plane_batch)
    stack = NANDStackConfig(num_channels=channels, num_dies_per_channel=dies_per_ch)
    cfg = HBFSimConfig(subarray=sa, nand_die=die, nand_stack=stack)
    mapper = PlaneAddressMapper(sa_cfg=sa, die_cfg=die, stack_cfg=stack,
                                policy=policy, num_layers=64)
    return cfg, mapper


CONFIGS = {
    "paper_1sa":      dict(planes_per_die=96, dies_per_ch=8, channels=1,
                           subarrays_per_plane=1),
    "small_1sa":      dict(planes_per_die=4, dies_per_ch=2, channels=1,
                           subarrays_per_plane=1),
    "multi_sa":       dict(planes_per_die=4, dies_per_ch=2, channels=1,
                           subarrays_per_plane=4),
    "multi_sa_cap1":  dict(planes_per_die=4, dies_per_ch=2, channels=1,
                           subarrays_per_plane=4, plane_cap=1),
    "no_merge":       dict(planes_per_die=8, dies_per_ch=1, channels=1,
                           subarrays_per_plane=1, enable_mpc=False),
}


# ---------------------------------------------------------------------------
# Random pattern generator
# ---------------------------------------------------------------------------

# KV-block page counts for real models (per-token bytes × 16 tokens / 4096):
#   Llama-3/2 (GQA8)=16, phi-2=40, Qwen-72B (64 KV heads)=128, big-MHA>256
MODEL_KV_PAGES = [16, 40, 128, 320]


def gen_pattern(rng, ppb, uniform=False, ops="read", realistic=False):
    n = rng.choice([1, 1, 2, 5, 10, 30, 80, 200, 600])
    if realistic:
        # realistic decode: every read is a distinct cold KV block (globally
        # unique id) of one model's uniform size -> latency-homogeneous.
        block_mode = "distinct"
        size_mode = "uniform"
        uni_pages = rng.choice(MODEL_KV_PAGES)
    else:
        block_mode = rng.choice(["scatter", "collide", "contig", "shared_layers",
                                 "few_planes"])
        if uniform:
            size_mode = "uniform"
            uni_pages = rng.choice(MODEL_KV_PAGES)
        else:
            size_mode = rng.choice(["subpage", "1page", "multi", "cross_block",
                                    "huge"])
    if realistic:
        # a decode step issues every KV read at ONE token index -> one row_offset
        tok_mode = "step"
        step_tok = rng.randint(0, 4096)
    else:
        tok_mode = rng.choice(["zero", "small", "varied"])
    nseq  = rng.choice([1, 2, 4, 16])
    nlay  = rng.choice([1, 2, 8, 32])
    reads = []
    base = rng.randint(0, 50)
    for i in range(n):
        if block_mode == "scatter":
            bid = rng.randint(0, 5000)
        elif block_mode == "collide":
            bid = 0
        elif block_mode == "contig":
            bid = base + i
        elif block_mode == "few_planes":
            bid = rng.choice([0, 1, 2, 3]) + base
        elif block_mode == "distinct":
            bid = base + i * 1009          # globally unique -> distinct cold block
        else:  # shared_layers — same block re-read across layers
            bid = base + (i % 3)
        if size_mode == "uniform":
            size = 4096 * uni_pages
        elif size_mode == "subpage":
            size = rng.randint(1, 4095)
        elif size_mode == "1page":
            size = 4096
        elif size_mode == "multi":
            size = 4096 * rng.randint(2, 8)
        elif size_mode == "cross_block":
            size = 4096 * (ppb + rng.randint(1, 10))
        else:  # huge
            size = 4096 * rng.randint(ppb, ppb * 4)
        if ops == "read":
            op = OpType.READ
        else:
            op = rng.choice([OpType.READ, OpType.READ, OpType.WRITE,
                             OpType.PREFETCH])
        if tok_mode == "zero":
            tok = 0
        elif tok_mode == "step":
            tok = step_tok
        elif tok_mode == "small":
            tok = rng.randint(0, 3)
        else:
            tok = rng.randint(0, 4096)
        reads.append(dict(
            op=op, size_bytes=size, block_id=bid,
            layer_id=rng.randrange(nlay), sequence_id=rng.randrange(nseq),
            token_id=tok,
            num_blocks=(0 if uniform else rng.choice([0, 0, 0, rng.randint(1, 8)]))))
    if rng.random() < 0.5:
        rng.shuffle(reads)
    return reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform", action="store_true",
                    help="all reads in a step share one model's KV-block size")
    ap.add_argument("--ops", choices=["read", "mixed"], default="read",
                    help="'mixed' interleaves WRITE/PREFETCH with READ")
    ap.add_argument("--realistic", action="store_true",
                    help="distinct cold KV blocks + uniform size (real decode)")
    ap.add_argument("--tol_pct", type=float, default=0.0,
                    help="report mismatch only if rel error exceeds this %")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    POLICIES = [PlacementPolicy.STRIPE_ACROSS_PLANES,
                PlacementPolicy.PACK_BY_SEQUENCE,
                PlacementPolicy.INTERLEAVE_BY_TOKEN]
    total = 0
    mismatches = []
    worst_pct = 0.0
    for cfg_name, cfg_kw in CONFIGS.items():
        for policy in POLICIES:
            cfg, mapper = build_cfg(policy=policy, **cfg_kw)
            ppb = cfg.subarray.pages_per_block
            cfg_mismatch = 0
            for _ in range(args.iters):
                reads = gen_pattern(rng, ppb, uniform=args.uniform, ops=args.ops,
                                    realistic=args.realistic)
                ref  = scheduler_latency_ns(reads, cfg, mapper)
                fast = fast_decode_latency_ns(reads, cfg, mapper)
                total += 1
                pct = abs(ref - fast) / ref * 100.0 if ref > 0 else 0.0
                worst_pct = max(worst_pct, pct)
                if pct > args.tol_pct + 1e-9:
                    cfg_mismatch += 1
                    if len(mismatches) < 6:
                        mismatches.append((cfg_name, policy.value, ref, fast, reads))
            tag = "OK" if cfg_mismatch == 0 else f"*** {cfg_mismatch} > tol ***"
            print(f"{cfg_name:<14} {policy.value:<10} {args.iters} patterns   {tag}")

    print("-" * 60)
    print(f"worst relative error: {worst_pct:.3f}%")
    print(f"total patterns: {total}   mismatches: {sum(1 for _ in mismatches) if mismatches else 0}"
          f"{'' if not mismatches else ' (showing first ' + str(len(mismatches)) + ')'}")
    for (cfg_name, pol, ref, fast, reads) in mismatches:
        print(f"\n[{cfg_name}/{pol}] ref={ref:.3f} fast={fast:.3f}  n={len(reads)}")
        for r in reads[:40]:
            print("   ", r)


if __name__ == "__main__":
    main()
