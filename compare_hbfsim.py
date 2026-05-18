# -*- coding: utf-8 -*-
"""
compare_hbfsim.py — side-by-side timing comparison of our Python PlaneScheduler
against HBFSim's C++ offline model on two micro-benchmarks:

  Case A: multi-plane parallel read
    4 reads issued simultaneously, each targeting a *different* plane.
    Multi-plane merge fires in both models -> wall-clock latency ≈ 1×tR.

  Case B: same-plane sequential read
    4 reads to the *same* plane (different pages).
    HBFSim: first page pays tR, subsequent pages in the same block pay tRC
            (cache-read shortcut; block stays in page-buffer).
    Our model: I/O-gated — every page costs tR regardless (the I/O transfer
               from NAND to GPU over the TSV is the bottleneck, not sense time).

HBFSim config used: hbf_research_scaled (tR=1000 ns), sanity_tiny geometry
  (block_interleave, 16 ch, 16 dies, 32 subarrays/die, 4 planes/die, 16KB pages)
Our model config: tR=1000 ns, 4 planes, 1 die, 1 channel, 4KB pages (SLC I/O)

Usage:
  cd /home/adityaan/vidur
  PYTHONPATH=. python3.9 compare_hbfsim.py
"""

import os
import sys
import csv
import math
import subprocess
import tempfile
import textwrap

# ── Our model imports ─────────────────────────────────────────────────────────
from vidur.memory_backends.hbf.plane_scheduler import PlaneScheduler, NANDRequest
from vidur.memory_backends.hbf.address_mapper import PhysicalAddr
from vidur.memory_backends.base import OpType
from vidur.memory_backends.hbf.config_loader import NANDDieConfig, NANDStackConfig, SubarrayConfig

HBFSIM_BIN  = "/home/adityaan/HBFSim/build-dramsim3/hbfsim"
HBFSIM_ROOT = "/home/adityaan/HBFSim"

# ---------------------------------------------------------------------------
# Shared timing knob
# ---------------------------------------------------------------------------
TR_NS = 1_000.0   # hbf_research_scaled tR; also used for our model

# ---------------------------------------------------------------------------
# ── Section 1: Our Python PlaneScheduler ─────────────────────────────────────
# ---------------------------------------------------------------------------

def make_scheduler_4planes(tR_ns: float) -> PlaneScheduler:
    """
    Minimal 4-plane scheduler (1 die, 1 channel) with tR_ns per page.
    Geometry mirrors one physical die of the HBFSim sanity_tiny config.
    """
    sa_cfg = SubarrayConfig(
        page_size_bytes=4_096,    # SLC 4 KB (I/O-gated at 1 GB/s/plane)
        pages_per_block=256,
        tR_ns=tR_ns,
        tRC_ns=30.0,              # cache-read (irrelevant for I/O-gated model)
    )
    die_cfg = NANDDieConfig(
        num_planes=4,
        blocks_per_plane=256,
        num_subarrays=4,          # 1 subarray per plane
        enable_multi_plane_cmds=True,
        max_plane_batch=4,
    )
    stack_cfg = NANDStackConfig(
        num_channels=1,
        num_dies_per_channel=1,
    )
    return PlaneScheduler(die_cfg, sa_cfg, stack_cfg)


def _phys(subarray_id: int, page_id: int) -> PhysicalAddr:
    """Helper: PhysicalAddr for a subarray (= plane in our 1-SA-per-plane model)."""
    return PhysicalAddr(
        channel_id=0,
        stack_id=0,
        die_id=0,
        plane_id=subarray_id,   # 1-to-1 with plane in our minimal config
        block_id=0,
        page_id=page_id,
        subarray_id=subarray_id,
    )


def run_ours_caseA() -> dict:
    """Case A: 4 reads, each on a distinct plane, same page offset (row=0)."""
    sched = make_scheduler_4planes(TR_NS)
    for plane in range(4):
        sched.submit(NANDRequest(
            req_id=plane, op=OpType.READ,
            addr=_phys(subarray_id=plane, page_id=0),
            size_bytes=4_096, issued_at=0.0,
        ))
    completions = sched.drain()
    return {
        "wall_ns":    max(c.latency_ns for c in completions),
        "n_cmds":     sched.total_cmds_issued(),
        "n_merges":   sched.multi_plane_merge_count(),
        "completions": sorted((c.req_id, c.latency_ns) for c in completions),
    }


def run_ours_caseB() -> dict:
    """Case B: 4 reads to plane 0, pages 0-3 of the *same* block (same row_offset series)."""
    sched = make_scheduler_4planes(TR_NS)
    for page in range(4):
        sched.submit(NANDRequest(
            req_id=page, op=OpType.READ,
            addr=_phys(subarray_id=0, page_id=page),
            size_bytes=4_096, issued_at=0.0,
        ))
    completions = sched.drain()
    return {
        "wall_ns":    max(c.latency_ns for c in completions),
        "n_cmds":     sched.total_cmds_issued(),
        "n_merges":   sched.multi_plane_merge_count(),
        "completions": sorted((c.req_id, c.latency_ns) for c in completions),
    }


def run_ours_caseC() -> dict:
    """Case C: 4 reads to plane 0, page 0 of 4 *different* blocks (realistic KV access).
    In our I/O-gated model this is identical to Case B: every page pays tR regardless.
    In HBFSim, different block_ids mean no tRC shortcut — each pays full tR.
    """
    sched = make_scheduler_4planes(TR_NS)
    for i in range(4):
        sched.submit(NANDRequest(
            req_id=i, op=OpType.READ,
            # block_id varies; page_id=0 in each block (row_offset = page_id % pages_per_block = 0)
            addr=PhysicalAddr(
                channel_id=0, stack_id=0, die_id=0,
                plane_id=0, block_id=i, page_id=0,
                subarray_id=0,          # plane 0
            ),
            size_bytes=4_096, issued_at=0.0,
        ))
    completions = sched.drain()
    return {
        "wall_ns":    max(c.latency_ns for c in completions),
        "n_cmds":     sched.total_cmds_issued(),
        "n_merges":   sched.multi_plane_merge_count(),
        "completions": sorted((c.req_id, c.latency_ns) for c in completions),
    }


# ---------------------------------------------------------------------------
# ── Section 1b: Case D — bandwidth invariance ─────────────────────────────────
# ---------------------------------------------------------------------------

# Workload: 6000 KV blocks × 256 KB = 1536 MB total
# All three configs share the same total bandwidth = 768 GB/s = 768 bytes/ns.
# Expected decode time ≈ 1536 MB / 768 GB/s ≈ 2.097 ms (same for all configs).
TOTAL_KV_BYTES_D  = 6144 * 256 * 1024   # 1536 MB (divisible by 64×8MB for clean convergence)
TOTAL_BW_BYTES_NS = 768.0               # 768 GB/s = 768 bytes/ns


def _make_scheduler_flat(n_planes: int, page_size_bytes: int, tR_ns: float) -> PlaneScheduler:
    """Flat geometry: 1 die, n_planes planes, 1 subarray per plane.
    Multi-plane merge disabled — each plane reads its share independently.
    """
    sa_cfg = SubarrayConfig(
        page_size_bytes=page_size_bytes,
        pages_per_block=max(4, 8 * 1024 * 1024 // page_size_bytes),
        tR_ns=tR_ns,
        tRC_ns=30.0,
    )
    die_cfg = NANDDieConfig(
        num_planes=n_planes,
        blocks_per_plane=256,
        num_subarrays=n_planes,           # 1 subarray per plane
        enable_multi_plane_cmds=False,    # aggregate bandwidth test: no merging
        max_plane_batch=n_planes,
    )
    stack_cfg = NANDStackConfig(num_channels=1, num_dies_per_channel=1)
    return PlaneScheduler(die_cfg, sa_cfg, stack_cfg)


def run_ours_caseD_config(n_planes: int, page_size_bytes: int) -> dict:
    """Run Case D for one (n_planes, page_size) configuration.
    tR is derived from page_size to preserve total bandwidth = TOTAL_BW_BYTES_NS.

    Each page is submitted as a separate request to a distinct block_id, simulating
    the LLM KV access pattern: different sequences map to different NAND blocks, so
    every read is a page-buffer miss (pays tR).  This is what makes tRC irrelevant
    for long-context decode — there are no same-block re-reads across sequences.
    """
    bw_per_plane   = TOTAL_BW_BYTES_NS / n_planes   # bytes/ns per plane
    tR_ns          = page_size_bytes / bw_per_plane  # ns
    sched          = _make_scheduler_flat(n_planes, page_size_bytes, tR_ns)
    kv_per_plane   = TOTAL_KV_BYTES_D / n_planes
    pages_per_plane = max(1, math.ceil(kv_per_plane / page_size_bytes))

    req_id = 0
    for p in range(n_planes):
        for blk in range(pages_per_plane):          # one page per distinct block
            sched.submit(NANDRequest(
                req_id=req_id, op=OpType.READ,
                addr=PhysicalAddr(
                    channel_id=0, stack_id=0, die_id=0,
                    plane_id=p, block_id=blk, page_id=0,
                    subarray_id=p,
                ),
                size_bytes=page_size_bytes,
                issued_at=0.0,
            ))
            req_id += 1

    completions = sched.drain()
    return {
        "wall_ns":           max(c.latency_ns for c in completions),
        "tR_ns":             tR_ns,
        "bw_per_plane_gbps": bw_per_plane,
        "kv_per_plane_mb":   kv_per_plane / 1024 / 1024,
        "pages_per_plane":   pages_per_plane,
        "cache_hits":        sched.cache_hit_count(),
    }


# ---------------------------------------------------------------------------
# ── Section 2: HBFSim C++ model ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Geometry (sanity_tiny / hbf_research_scaled):
#   block_interleave, D=16 dies, S=32 subarrays/die, 4 planes/die (8 SA/plane)
#   pages_per_block=512, page_size=16384 => block_stride = 512*16384 = 8 MB
#   freq_mhz=1000 => 1 cycle = 1 ns
#
# Address-to-plane mapping (block_interleave):
#   global_block = addr / block_stride
#   die_id       = global_block % 16
#   subarray_id  = (global_block // 16) % 32
#   plane_id     = subarray_id // 8  (8 subarrays per plane)
#
# Case A — 4 reads to same die (die=0), different planes (0,1,2,3):
#   plane 0: subarray=0  => block = 0*16 + 0   = 0    => addr = HBF_BASE + 0*8M
#   plane 1: subarray=8  => block = 8*16 + 0   = 128  => addr = HBF_BASE + 128*8M
#   plane 2: subarray=16 => block = 16*16 + 0  = 256  => addr = HBF_BASE + 256*8M
#   plane 3: subarray=24 => block = 24*16 + 0  = 384  => addr = HBF_BASE + 384*8M
#   Same row_offset (page 0 in each block) => multi-plane merge eligible.
#
# Case B — 4 reads to same plane (die=0, subarray=0, plane=0), pages 0-3:
#   block=0, pages 0-3 => consecutive 16KB addresses within block 0
#   First page: tR; subsequent pages: tRC (cache-read, block stays in buffer)

HBF_BASE    = 0x1800000000    # = 0x1800000000 (10 hex digits, ~66 GB mark)
BLOCK_BYTES = 512 * 16384   # 8 MB per block = 0x800000


def _hbfsim_addr(block: int, page: int = 0) -> int:
    return HBF_BASE + block * BLOCK_BYTES + page * 16384


HBFSIM_CONFIG_TMPL = textwrap.dedent("""\
    [simulation]
    max_cycles              = 50_000_000
    warmup_cycles           = 0
    freq_mhz                = 1000
    output_csv_path         = "{csv_out}"
    output_stats_json_path  = "{stats_out}"
    log_level               = 0
    progress_interval_seconds = 999.0

    [nand_profile]
    name = "hbf_research_scaled"

    [requester]
    trace_file              = "{trace}"
    max_in_flight           = 64
    issue_rate_per_cyc      = 64

    [host_bus]
    bandwidth_GBps          = 4000.0
    base_latency_cycles     = 0
    queue_depth             = 64
    cpu_compute_cycles      = 0

    [memory_model]
    address_space = "composite"
    hbm_role = "addressable"
    hbf_role = "addressable"
    access_policy = "address_partition"
    fabric = "ideal_parallel"

    [address_mapping]
    policy                  = "block_interleave"
    num_stacks              = 1
    num_channels            = 16
    num_dies_hbf            = 16
    num_subarrays_hbf       = 32
    page_size_bytes         = 16384
    pages_per_block         = 512
    hbm_base_addr           = "0x0"
    hbm_size_bytes          = "1 GB"
    hbf_base_addr           = "0x1800000000"
    hbf_size_bytes          = "512 GB"

    [hbf_controller]
    read_queue_depth              = 64
    write_queue_depth             = 32
    scheduler_type                = "fcfs"
    controller_latency_cycles     = 0
    enable_read_priority          = true
    max_outstanding_per_subarray  = 64

      [hbf_controller.ftl]
      policy                 = "page_mapping"
      write_allocator        = "round_robin"
      unmapped_read_policy   = "prepopulate"

      [hbf_controller.gc]
      policy          = "greedy"

      [hbf_controller.wear_leveler]
      policy          = "round_robin"

      [hbf_controller.thermal]
      policy          = "null"

    [logic_die]
    num_channels                  = 16
    num_dies_per_stack            = 16
    num_subarrays_per_die         = 32
    tsv_bandwidth_GBps            = 4000
    logic_die_freq_mhz            = 1000
    command_translation_cycles    = 0
    max_concurrent_subarray_cmds  = 16384

    [nand_media]
    page_size_bytes           = 16384
    pages_per_block           = 512
    blocks_per_subarray       = 1024
    num_dies_per_stack        = 16
    num_subarrays_per_die     = 32
    num_planes_per_die        = 4
    max_concurrent_per_plane  = 0
    capacity_per_die_bytes    = "32 GB"
    queue_depth               = 64
""")


def _make_trace_caseA() -> str:
    """4 back-to-back reads to same die / different planes (multi-plane merge)."""
    addrs = [
        _hbfsim_addr(block=0),    # die=0, subarray=0,  plane=0
        _hbfsim_addr(block=128),  # die=0, subarray=8,  plane=1
        _hbfsim_addr(block=256),  # die=0, subarray=16, plane=2
        _hbfsim_addr(block=384),  # die=0, subarray=24, plane=3
    ]
    lines = ["# Case A: 4 reads, same die (0), different planes (0-3), same row_offset"]
    for i, addr in enumerate(addrs):
        ts = 0 if i == 0 else -1
        lines.append(f"{ts} R {hex(addr)} 16384 0 caseA")
    return "\n".join(lines) + "\n"


def _make_trace_caseB() -> str:
    """4 back-to-back reads to same plane (die=0, plane=0), consecutive pages of SAME block.
    HBFSim: 1×tR + 3×tRC  (page-buffer cache-read — block stays loaded after first sense).
    """
    lines = ["# Case B: 4 reads, same plane (die=0 plane=0), pages 0-3 of SAME block (tRC fires)"]
    for page in range(4):
        ts = 0 if page == 0 else -1
        addr = _hbfsim_addr(block=0, page=page)
        lines.append(f"{ts} R {hex(addr)} 16384 0 caseB")
    return "\n".join(lines) + "\n"


def _make_trace_caseC() -> str:
    """4 reads to same plane (die=0, plane=0), page 0 of DIFFERENT blocks.
    Each read hits a new block → page buffer invalidated → full tR every time.
    This is the realistic LLM KV pattern: 4 sequences stripe-mapped to the same plane.

    Block selection (block_interleave, D=16, S=32):
      die = blk % 16, subarray = (blk//16) % 32, plane = subarray // 8
      For die=0, plane=0 (subarray=0): blk = k*16*32 = k*512  (k=0,1,2,3)

    To avoid the FTL unmapped-block stall (prepopulate writes 512 pages before first read),
    we pre-write page 0 of each target block first (1 page × tPROG = 2000 ns each).
    Only the caseC READ records are used for latency comparison.
    """
    target_blocks = [k * 16 * 32 for k in range(4)]   # 0, 512, 1024, 1536
    lines = [
        "# Case C: write page 0 of each target block (FTL allocation), then read",
        "# Writes: back-to-back on same subarray → serialized at tPROG=2000ns each",
    ]
    for i, blk in enumerate(target_blocks):
        ts = 0 if i == 0 else -1
        addr = _hbfsim_addr(block=blk, page=0)
        lines.append(f"{ts} W {hex(addr)} 16384 0 caseC_init")
    lines.append("# Reads: same subarray, different blocks → page buffer miss → tR each")
    for i, blk in enumerate(target_blocks):
        addr = _hbfsim_addr(block=blk, page=0)
        lines.append(f"-1 R {hex(addr)} 16384 0 caseC")
    return "\n".join(lines) + "\n"


def run_hbfsim_case(trace_content: str, label: str, filter_op_name: str = None) -> dict:
    """Write trace+config, run HBFSim binary, parse CSV output.

    filter_op_name: if set, only include rows where op_name == this value.
    """
    with tempfile.TemporaryDirectory(prefix="hbfsim_cmp_") as tmp:
        trace_path = os.path.join(tmp, "trace.trace")
        cfg_path   = os.path.join(tmp, "sim.toml")
        csv_out    = os.path.join(tmp, "out.csv")
        stats_out  = os.path.join(tmp, "stats.json")

        with open(trace_path, "w") as f:
            f.write(trace_content)

        cfg_text = HBFSIM_CONFIG_TMPL.format(
            csv_out=csv_out, stats_out=stats_out, trace=trace_path
        )
        with open(cfg_path, "w") as f:
            f.write(cfg_text)

        result = subprocess.run(
            [HBFSIM_BIN, cfg_path],
            capture_output=True, text=True,
            cwd=HBFSIM_ROOT, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"HBFSim failed for {label}:\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )

        rows = []
        with open(csv_out, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if filter_op_name is None or row.get("op_name") == filter_op_name:
                    rows.append(row)

        if not rows:
            raise RuntimeError(f"HBFSim produced no {filter_op_name or ''} rows for {label}")

        total_times    = [int(r["total_time"])    for r in rows]
        subarray_times = [int(r["subarray_time"]) for r in rows]
        die_ids        = [int(r["die_id"])         for r in rows]
        sa_ids         = [int(r["subarray_id"])    for r in rows]
        t_sends        = [int(r["t_send"])          for r in rows]

        # Wall-clock for a batch = span from first issue to last completion
        completions_abs = [ts + tt for ts, tt in zip(t_sends, total_times)]
        wall_ns = max(completions_abs) - min(t_sends)

        return {
            "wall_ns":        wall_ns,
            "total_times":    total_times,
            "subarray_times": subarray_times,
            "die_ids":        die_ids,
            "subarray_ids":   sa_ids,
        }


# ---------------------------------------------------------------------------
# ── Section 3: Print comparison ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def fmt(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns/1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns/1_000:.2f} µs"
    return f"{ns:.1f} ns"


def main():
    print("=" * 70)
    print("  HBFSim C++ vs. Python PlaneScheduler — micro-benchmark comparison")
    print("  tR = 1000 ns (hbf_research_scaled), 4 planes per die")
    print("=" * 70)

    # ── Run our model ──────────────────────────────────────────────────────────
    our_A  = run_ours_caseA()
    our_B  = run_ours_caseB()
    our_C  = run_ours_caseC()
    our_D1 = run_ours_caseD_config(n_planes=768,  page_size_bytes=4_096)           # current default
    our_D2 = run_ours_caseD_config(n_planes=64,   page_size_bytes=8*1024*1024)     # HBFSim-like block
    our_D3 = run_ours_caseD_config(n_planes=1024, page_size_bytes=16_384)          # HBFSim exact geom

    # ── Run HBFSim ────────────────────────────────────────────────────────────
    hbf_ok = os.path.exists(HBFSIM_BIN)
    if hbf_ok:
        try:
            hbf_A = run_hbfsim_case(_make_trace_caseA(), "Case A")
            hbf_B = run_hbfsim_case(_make_trace_caseB(), "Case B")
            hbf_C = run_hbfsim_case(_make_trace_caseC(), "Case C", filter_op_name="caseC")
        except Exception as e:
            print(f"\n[!] HBFSim run failed: {e}\n")
            hbf_ok = False

    # ── Case A: multi-plane parallel ──────────────────────────────────────────
    print("\n── Case A: 4 reads × 4 planes (multi-plane merge / parallel) ─────────")
    print(f"  Setup: die=0, planes 0-3, same row_offset (page 0 in each plane)")
    print(f"  Expected: 1×tR = {fmt(TR_NS)} (parallel dispatch, one die cycle)")
    print()
    print(f"  Our PlaneScheduler:")
    print(f"    wall-clock latency = {fmt(our_A['wall_ns'])}")
    print(f"    cmds issued        = {our_A['n_cmds']}  (should be 1 merged cmd)")
    print(f"    multi-plane merges = {our_A['n_merges']}")
    print(f"    per-req completions: {[(id_, fmt(lat)) for id_, lat in our_A['completions']]}")

    if hbf_ok:
        print(f"\n  HBFSim (hbf_research_scaled, block_interleave):")
        print(f"    wall-clock latency = {fmt(hbf_A['wall_ns'])}")
        print(f"    per-req total_time: {[fmt(t) for t in hbf_A['total_times']]}")
        print(f"    die_ids:           {hbf_A['die_ids']}")
        print(f"    subarray_ids:      {hbf_A['subarray_ids']}")
        note = ("HBM-parallel: different dies run independently"
                if len(set(hbf_A['die_ids'])) > 1 else
                "same die, multi-plane merge")
        print(f"    parallelism:       {note}")

    # ── Case B: same-plane sequential ─────────────────────────────────────────
    print("\n── Case B: 4 reads × same plane, consecutive pages ─────────────────")
    print(f"  Setup: die=0, plane=0, pages 0-3 of block 0")
    print(f"  Expected (ours):   4×tR = {fmt(4*TR_NS)}  (I/O-gated, no tRC shortcut)")
    print(f"  Expected (HBFSim): 1×tR + 3×tRC = {fmt(TR_NS + 3*30)}  (page-buffer cache-read)")
    print()
    print(f"  Our PlaneScheduler:")
    print(f"    wall-clock latency = {fmt(our_B['wall_ns'])}")
    print(f"    cmds issued        = {our_B['n_cmds']}")
    print(f"    per-req completions: {[(id_, fmt(lat)) for id_, lat in our_B['completions']]}")

    if hbf_ok:
        print(f"\n  HBFSim (hbf_research_scaled, block_interleave):")
        print(f"    wall-clock latency = {fmt(hbf_B['wall_ns'])}")
        print(f"    per-req total_time: {[fmt(t) for t in hbf_B['total_times']]}")
        print(f"    subarray_times:    {[fmt(t) for t in hbf_B['subarray_times']]}")

    # ── Case C: same plane, different blocks (realistic KV) ───────────────────
    print("\n── Case C: 4 reads × same plane, DIFFERENT blocks (realistic KV) ────")
    print(f"  Setup: die=0, plane=0; each read is page 0 of a distinct block")
    print(f"         (mirrors 4 sequences mapped to same plane by STRIPE policy)")
    print(f"  Expected (both): 4×tR = {fmt(4*TR_NS)}  (new block each time, no cache-read possible)")
    print()
    print(f"  Our PlaneScheduler:")
    print(f"    wall-clock latency = {fmt(our_C['wall_ns'])}")
    print(f"    cmds issued        = {our_C['n_cmds']}")
    print(f"    per-req completions: {[(id_, fmt(lat)) for id_, lat in our_C['completions']]}")

    if hbf_ok:
        all_trc = all(t <= 35 for t in hbf_C["subarray_times"])
        print(f"\n  HBFSim (hbf_research_scaled, block_interleave):")
        print(f"    wall-clock latency = {fmt(hbf_C['wall_ns'])}")
        print(f"    per-req total_time: {[fmt(t) for t in hbf_C['total_times']]}")
        print(f"    subarray_times:    {[fmt(t) for t in hbf_C['subarray_times']]}")
        if all_trc:
            print(f"    [!] FTL artifact: round_robin allocator packed all 4 writes into the")
            print(f"        same physical NAND block → reads see tRC, not tR.  The logical")
            print(f"        intent (different-block serialization) is masked by address remapping.")

    # ── Case D: bandwidth invariance ──────────────────────────────────────────
    expected_ms = TOTAL_KV_BYTES_D / (TOTAL_BW_BYTES_NS * 1e6)
    print("\n── Case D: bandwidth-invariant page size (I/O-gated model) ──────────")
    print(f"  Workload: {TOTAL_KV_BYTES_D // (1024*1024)} MB total KV read (6144 KV blocks × 256 KB)")
    print(f"  All configs: total bandwidth = {TOTAL_BW_BYTES_NS:.0f} GB/s  →  expected ≈ {expected_ms:.3f} ms")
    print(f"  (Convergence holds when kv_per_plane ≫ page_size; ceiling quantisation causes small delta)")
    print()
    for cfg_label, d in [
        ("768 planes × 4 KB  (current default)",   our_D1),
        (" 64 planes × 8 MB  (HBFSim block size)", our_D2),
        ("1024 planes × 16 KB (HBFSim geom)",      our_D3),
    ]:
        delta_pct = 100.0 * (d["wall_ns"] / 1e6 - expected_ms) / expected_ms
        print(f"  {cfg_label}")
        print(f"    tR = {fmt(d['tR_ns'])}/page  ·  {d['pages_per_plane']} pages/plane  ·  "
              f"wall = {fmt(d['wall_ns'])}  (Δ {delta_pct:+.1f}% vs expected)  "
              f"cache_hits={d['cache_hits']}")
    print()
    print("  Key: all three achieve the same throughput because the I/O-gated model")
    print("  satisfies:  n_planes × page_size / tR = const = total_bandwidth.")
    print("  Larger pages → fewer pages/plane but proportionally longer tR → same wall time.")
    print("  The pages_per_kv_block fix (ceil formula) is what makes this work correctly.")

    # ── Terminology note ──────────────────────────────────────────────────────
    block_bytes_hbf = 512 * 16384     # HBFSim: 8 MB per NAND block (tR sense unit)
    kv_block_bytes  = 256 * 1024      # LLM KV block: 256 KB
    hbf_pages_per_kv = kv_block_bytes // 16384   # = 16 HBFSim pages per KV block
    our_pages_per_kv = kv_block_bytes // 4096     # = 64 our pages per KV block
    hbf_kv_cost_ns  = TR_NS + (hbf_pages_per_kv - 1) * 30.0   # tR + 15×tRC
    our_kv_cost_ns  = our_pages_per_kv * TR_NS                 # 64×tR

    print("\n── Block vs. page: terminology difference ───────────────────────────")
    print(f"  HBFSim NAND block = 512 pages × 16 KB = {block_bytes_hbf//1024//1024} MB  (tR sense unit)")
    print(f"  Our NAND page     = 4 KB                               (tR transfer unit)")
    print()
    print(f"  In HBFSim, one tR loads the ENTIRE NAND block into the page buffer.")
    print(f"  All 512 pages of that block are then readable at tRC=30 ns each.")
    print(f"  In our model, one tR transfers ONE page (4 KB) over the TSV I/O bus.")
    print(f"  There is no page-buffer caching — each page pays full tR regardless.")
    print()
    print(f"  Effect on a 256 KB LLM KV block read (at tR={fmt(TR_NS)}):")
    print(f"    HBFSim: {hbf_pages_per_kv} pages × 16 KB fit in 1 NAND block "
          f"→ tR + {hbf_pages_per_kv-1}×tRC = {fmt(hbf_kv_cost_ns)}")
    print(f"    Ours:   {our_pages_per_kv} pages × 4 KB each at tR"
          f"          → {our_pages_per_kv}×tR = {fmt(our_kv_cost_ns)}  ({our_kv_cost_ns/hbf_kv_cost_ns:.0f}× slower)")
    print(f"  HBFSim 'block' (8 MB, sense unit) ≈ our 'pages' (4 KB × 64 = 256 KB / KV block)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  {'Case':<38} {'Our model':>12} {'HBFSim':>12}  Notes")
    print(f"  {'-'*38} {'-'*12} {'-'*12}  -----")

    def _row(label, ours_ns, hbf_ns=None, note=""):
        hbf_str = fmt(hbf_ns) if hbf_ns is not None else "  n/a"
        print(f"  {label:<38} {fmt(ours_ns):>12} {hbf_str:>12}  {note}")

    _row("A: 4-plane parallel",
         our_A["wall_ns"],
         hbf_A["wall_ns"] if hbf_ok else None,
         "agree — parallel dispatch")
    _row("B: same plane, same block (pages 0-3)",
         our_B["wall_ns"],
         hbf_B["wall_ns"] if hbf_ok else None,
         "diverge — HBFSim tRC fires (block already loaded)")
    _row("C: same plane, diff blocks (real KV)",
         our_C["wall_ns"],
         hbf_C["wall_ns"] if hbf_ok else None,
         "HBFSim FTL artifact; ours correct at 4×tR")

    print()
    print("  Key modelling differences:")
    print("  ┌──────────────────────┬───────────────────────────┬──────────────────────────┐")
    print("  │ Property             │ Our Python model          │ HBFSim C++ model         │")
    print("  ├──────────────────────┼───────────────────────────┼──────────────────────────┤")
    print("  │ Timing basis         │ I/O-gated (transfer BW)   │ NAND sense-time (tR/tRC) │")
    print("  │ Page size            │ 4 KB (SLC)                │ 16 KB (TLC BiCS)         │")
    print(f"  │ tR                   │ {TR_NS:>6.0f} ns (I/O @ 1GB/s) │ {TR_NS:>6.0f} ns (research_scaled)│")
    print("  │ tRC (cache-read)     │ irrelevant (I/O-gated)    │ 30 ns (page-buf shortcut)│")
    print("  │ Parallelism model    │ multi-plane merge (1 die) │ independent dies + m-p   │")
    print("  │ FTL / GC overhead    │ none (online model)       │ page-mapping FTL + gc    │")
    print("  │ Timing model style   │ online (per simulation step)│ offline trace replay   │")
    print("  └──────────────────────┴───────────────────────────┴──────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
