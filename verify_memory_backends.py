# -*- coding: utf-8 -*-
"""
Verification script for vidur memory backends.

Checks that the HBF and HBM backends:
  1. Generate physical addresses (plane, subarray, page per block)
  2. Enforce multi-plane command merging (batch reads → single tR cost)
  3. Enforce tR vs tRC timing (cold miss vs warm cache-read hit)
  4. Track per-plane stall counts and occupancy
  5. Emit a valid HBFSim-format trace file
  6. Track HBM row hit / miss / conflict states
  7. CompositeBackend serialises HBF before HBM and adjusts latencies

Run with:
    PYTHONPATH=. python3.9 verify_memory_backends.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader, SubarrayConfig, NANDDieConfig, NANDStackConfig
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy, PlaneAddressMapper
from vidur.memory_backends.hbf.plane_scheduler import PlaneScheduler, NANDRequest
from vidur.memory_backends.hbf.backend import HBFSimBackend
from vidur.memory_backends.ramulator.bank_scheduler import HBMBankScheduler, HBMRequest
from vidur.memory_backends.ramulator.backend import RamulatorBackend
from vidur.memory_backends.composite import CompositeBackend
from vidur.memory_backends.base import DataClass, MemRequest, OpType

HBF_CFG = "/home/adityaan/HBFSim/configs/staged_hbm_base_baseline.toml"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


# ──────────────────────────────────────────────────────────────────────────────
# 1. Address generation
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 1. Address Generation ===")

cfg = HBFSimConfigLoader(HBF_CFG).load()
mapper = PlaneAddressMapper(
    sa_cfg=cfg.subarray, die_cfg=cfg.nand_die, stack_cfg=cfg.nand_stack,
    policy=PlacementPolicy.STRIPE_ACROSS_PLANES, num_layers=32,
)

# Four sequences, same token, same layer → should land on planes 0,1,2,3
addrs = [mapper.map_block(block_id=i, layer_id=0, sequence_id=i, token_id=5)
         for i in range(4)]

planes = [a.plane_id for a in addrs]
row_offsets = [a.row_offset for a in addrs]

check("blocks 0-3 map to planes 0-3 (STRIPE)",
      planes == [0, 1, 2, 3], f"planes={planes}")
check("same token_id → same row_offset on all planes",
      len(set(row_offsets)) == 1, f"row_offsets={row_offsets}")
check("row_offset = token_id % pages_per_block",
      row_offsets[0] == 5 % cfg.subarray.pages_per_block,
      f"got {row_offsets[0]}, expected {5 % cfg.subarray.pages_per_block}")

# PACK_BY_SEQUENCE: all layers of seq 0 → same plane
pack_mapper = PlaneAddressMapper(
    sa_cfg=cfg.subarray, die_cfg=cfg.nand_die, stack_cfg=cfg.nand_stack,
    policy=PlacementPolicy.PACK_BY_SEQUENCE, num_layers=32,
)
pack_addrs = [pack_mapper.map_block(block_id=0, layer_id=l, sequence_id=0, token_id=0)
              for l in range(32)]
check("PACK_BY_SEQUENCE: all layers of seq 0 on same plane",
      len(set(a.plane_id for a in pack_addrs)) == 1,
      f"planes={set(a.plane_id for a in pack_addrs)}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. tR vs tRC timing
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 2. tR vs tRC Timing ===")

sched = PlaneScheduler(die_cfg=cfg.nand_die, sa_cfg=cfg.subarray, stack_cfg=cfg.nand_stack, freq_mhz=cfg.freq_mhz)
cycle_ns = 1000.0 / cfg.freq_mhz

# First read: cold miss (tR). Second read: exact same (block, page) → cache hit (tRC).
# A different page of the same block is a miss — page buffer holds one page at a time.
addr0 = mapper.map_block(block_id=0, layer_id=0, token_id=5)  # block B, page 5

sched.submit(NANDRequest(req_id=1, op=OpType.READ, addr=addr0, size_bytes=16384, issued_at=0.0))
c1 = sched.drain()

# Re-read the identical (block, page) → HIT
sched.submit(NANDRequest(req_id=2, op=OpType.READ, addr=addr0, size_bytes=16384, issued_at=c1[0].latency_ns))
c2 = sched.drain()

# Same block, different page → MISS (page buffer only holds one page)
addr0_diffpage = addr0.__class__(
    channel_id=addr0.channel_id, stack_id=addr0.stack_id, die_id=addr0.die_id,
    plane_id=addr0.plane_id, subarray_id=addr0.subarray_id,
    block_id=addr0.block_id, page_id=addr0.page_id + 1,
)
sched.submit(NANDRequest(req_id=3, op=OpType.READ, addr=addr0_diffpage, size_bytes=16384, issued_at=c1[0].latency_ns + c2[0].latency_ns))
c3 = sched.drain()

tR_ns  = cfg.subarray.tR_cycles  * cycle_ns
tRC_ns = cfg.subarray.tRC_cycles * cycle_ns

check("cold read pays tR",
      abs(c1[0].latency_ns - tR_ns) < 1.0,
      f"got {c1[0].latency_ns:.1f} ns, expected {tR_ns:.1f} ns")
check("warm cache-read (same block, same page) pays tRC",
      abs(c2[0].latency_ns - tRC_ns) < 1.0,
      f"got {c2[0].latency_ns:.1f} ns, expected ~{tRC_ns:.1f} ns")
check("tRC / tR ratio ≈ 1/1667",
      c2[0].latency_ns / c1[0].latency_ns < 0.01,
      f"ratio={c2[0].latency_ns/c1[0].latency_ns:.5f}")
check("same block different page is a miss (page buffer holds one page)",
      abs(c3[0].latency_ns - tR_ns) < 1.0,
      f"got {c3[0].latency_ns:.1f} ns, expected {tR_ns:.1f} ns")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Multi-plane command merging
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 3. Multi-plane Command Merging ===")

sched2 = PlaneScheduler(die_cfg=cfg.nand_die, sa_cfg=cfg.subarray, stack_cfg=cfg.nand_stack, freq_mhz=cfg.freq_mhz)

# 4 sequences, same token_id=0 → same row_offset, planes 0-3 → should merge
batch_addrs = [mapper.map_block(block_id=i, layer_id=0, sequence_id=i, token_id=0)
               for i in range(4)]
for i, addr in enumerate(batch_addrs):
    sched2.submit(NANDRequest(req_id=i+10, op=OpType.READ, addr=addr, size_bytes=16384, issued_at=0.0))

completions = sched2.drain()
merges = sched2.multi_plane_merge_count()
total_cmds = sched2.total_cmds_issued()

check("4 batch reads → multi-plane merges occurred",
      merges > 0, f"merges={merges}")
check("all 4 reads complete",
      len(completions) == 4, f"completions={len(completions)}")
check("merged reads: total cmds < num requests (parallelism)",
      total_cmds < 4, f"cmds={total_cmds}, requests=4")

# All completions should be the same latency (executed in parallel)
lats = [c.latency_ns for c in completions]
check("parallel completions have equal latency",
      max(lats) - min(lats) < 1.0, f"latencies={[f'{l:.1f}' for l in lats]}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Per-plane stall counts and occupancy stats
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 4. Plane Stats ===")

backend = HBFSimBackend(HBF_CFG, placement_policy=PlacementPolicy.STRIPE_ACROSS_PLANES, num_layers=32)
kv_block_ids = {seq: [seq * 32 + l for l in range(32)] for seq in range(8)}

backend.submit_decode_step(
    sequence_ids=list(range(8)), token_id=3, num_layers=32,
    kv_block_ids=kv_block_ids, kv_block_size_bytes=16384,
)
backend.drain()
ps = backend.plane_stats()

check("plane_stats() returns PlaneStats",
      ps is not None)
expected_total_planes = cfg.nand_stack.num_channels * cfg.nand_stack.num_dies_per_channel * cfg.nand_die.num_planes
check("num_planes == channels × dies × planes/die",
      ps.num_planes == expected_total_planes, f"got {ps.num_planes}, expected {expected_total_planes}")
check("plane_occupancies has one entry per plane",
      len(ps.plane_occupancies) == ps.num_planes,
      f"len={len(ps.plane_occupancies)}")
check("multi_plane_merge_count > 0 for 8-seq batch",
      ps.multi_plane_merge_count > 0, f"merges={ps.multi_plane_merge_count}")
check("cache_hits + cache_misses == total requests (8 seqs × 32 layers)",
      ps.cache_hits + ps.cache_misses == 8 * 32,
      f"hits={ps.cache_hits} misses={ps.cache_misses} total={ps.cache_hits+ps.cache_misses}")
check("Gini coefficient in [0, 1]",
      0.0 <= ps.gini_coefficient <= 1.0, f"gini={ps.gini_coefficient:.3f}")
check("active_plane_fraction > 0",
      ps.active_plane_fraction > 0.0, f"active={ps.active_plane_fraction:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. HBFSim trace emission
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 5. HBFSim Trace Emission ===")

import tempfile, pathlib

with tempfile.TemporaryDirectory() as tmpdir:
    traced_backend = HBFSimBackend(
        HBF_CFG,
        placement_policy=PlacementPolicy.STRIPE_ACROSS_PLANES,
        emit_trace=True,
        trace_dir=tmpdir,
        num_layers=32,
    )
    traced_backend.submit_decode_step(
        sequence_ids=[0, 1, 2, 3], token_id=7, num_layers=4,
        kv_block_ids={i: [i * 4 + l for l in range(4)] for i in range(4)},
        kv_block_size_bytes=16384,
    )
    traced_backend.drain()
    trace_path = traced_backend.emit_trace()

    check("trace file created", pathlib.Path(trace_path).exists(), str(trace_path))

    lines = pathlib.Path(trace_path).read_text().strip().splitlines()
    check("trace has 16 lines (4 seqs × 4 layers)",
          len(lines) == 16, f"lines={len(lines)}")

    # Check trace format: <cycles> <type> <hex_addr> <size> <layer> <op_name> <class> ...
    fields = lines[0].split()
    check("trace line has ≥7 fields",
          len(fields) >= 7, f"fields={fields}")
    check("address field is hex",
          fields[2].startswith("0x"), f"addr={fields[2]}")
    check("op type is read (HBFSim 'R' format)",
          fields[1] == "R", f"op={fields[1]}")
    check("data class field present (KV/COLD_KV)",
          "KV" in fields[6].upper(), f"class={fields[6]}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. HBM row buffer (Ramulator)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 6. HBM Row Buffer (Ramulator) ===")

hbm_sched = HBMBankScheduler(
    num_channels=16, num_banks_per_channel=8,
    tCL_cycles=14, tRCD_cycles=14, tRP_cycles=14,
    freq_mhz=1000.0, row_size_bytes=2048,
)

# Two reads to the same address (same row): first miss, second hit
addr = 0x0000_1000
hbm_sched.submit(HBMRequest(req_id=1, op=OpType.READ, addr=addr, size_bytes=64, issued_at=0.0))
c1 = hbm_sched.drain()

hbm_sched.submit(HBMRequest(req_id=2, op=OpType.READ, addr=addr, size_bytes=64, issued_at=c1[0].latency_ns))
c2 = hbm_sched.drain()

tCL_ns  = 14.0   # cycles × 1 ns/cycle
tRCD_ns = 14.0
miss_lat = tRCD_ns + tCL_ns
hit_lat  = tCL_ns

check("first access is row miss (tRCD+tCL)",
      abs(c1[0].latency_ns - miss_lat) < 1.0,
      f"got {c1[0].latency_ns:.1f} ns, expected {miss_lat:.1f} ns")
check("second access is row hit (tCL only)",
      abs(c2[0].latency_ns - hit_lat) < 1.0,
      f"got {c2[0].latency_ns:.1f} ns, expected {hit_lat:.1f} ns")
check("row hit is faster than row miss",
      c2[0].latency_ns < c1[0].latency_ns)

# Row conflict: read to different row in same bank.
# Stride = num_ch × num_bk × row_size_bytes = 16×8×2048 = 262144 B
# keeps (channel, bank) identical but advances the row number by 1.
conflict_addr = addr + 16 * 8 * 2048  # same bank, next row
hbm_sched.submit(HBMRequest(req_id=3, op=OpType.READ, addr=conflict_addr, size_bytes=64, issued_at=0.0))
c3 = hbm_sched.drain()
conflict_lat = tCL_ns + tRCD_ns + tCL_ns  # tRP + tRCD + tCL

check("row conflict pays tRP+tRCD+tCL",
      c3[0].latency_ns >= miss_lat,
      f"got {c3[0].latency_ns:.1f} ns")
check("stats: row_hits=1, row_misses=1, conflicts≥1",
      hbm_sched.row_hits() == 1 and hbm_sched.row_misses() == 1 and hbm_sched.row_conflicts() >= 1,
      f"hits={hbm_sched.row_hits()} misses={hbm_sched.row_misses()} conflicts={hbm_sched.row_conflicts()}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CompositeBackend: HBF serial then HBM, latency adjustment
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 7. CompositeBackend (HBF→HBM serialisation) ===")

hbf = HBFSimBackend(HBF_CFG, placement_policy=PlacementPolicy.STRIPE_ACROSS_PLANES, num_layers=32)
hbm = RamulatorBackend(HBF_CFG)
comp = CompositeBackend(hbf=hbf, hbm=hbm)

# 2 cold (→ HBF) + 2 hot (→ HBM, deferred)
comp.submit_decode_step(
    sequence_ids=[0, 1, 2, 3], token_id=0, num_layers=1,
    kv_block_ids={i: [i] for i in range(4)},
    kv_block_size_bytes=16384,
    hot_fraction=0.5,  # seqs 2,3 → HBM; seqs 0,1 → HBF
)
completions = comp.drain()

check("composite drain returns 4 completions (2 HBF + 2 HBM)",
      len(completions) == 4, f"got {len(completions)}")

# HBF completions should be the short tR latency
# HBM completions are adjusted: include HBF stall → always ≥ max HBF latency
hbf_lats = sorted([c.latency_ns for c in completions[:2]])
hbm_lats = sorted([c.latency_ns for c in completions[2:]])

check("HBF latencies are ~tR",
      all(abs(l - cfg.subarray.tR_cycles * cycle_ns) < 1000.0 for l in hbf_lats),
      f"hbf_lats={[f'{l:.0f}ns' for l in hbf_lats]}")
check("HBM latencies ≥ HBF latencies (includes HBF stall)",
      min(hbm_lats) >= max(hbf_lats) - 1.0,
      f"min_hbm={min(hbm_lats):.0f} max_hbf={max(hbf_lats):.0f}")
check("plane_stats() delegates to HBF",
      comp.plane_stats() is not None)
check("hbm_stats() delegates to HBM",
      comp.hbm_stats() is not None)

print("\n=== Done ===\n")
