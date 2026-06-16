# -*- coding: utf-8 -*-
"""
Validate our PlaneScheduler vs cycle-accurate HBFSim on AGGREGATE decode
bandwidth (the metric that feeds TPOT), across a parallelism sweep.

For n_units in a sweep, read n_units full NAND blocks (one per parallelism unit,
balanced), 510 pages each, from the HBF region. Run the SAME trace through:
  * HBFSim  (build/hbfsim) -> aggregate HBF GB/s
  * our PlaneScheduler at matched geometry + HBFSim's effective NAND timing
    (tR sense + tRC + DQ folded into tRC_eff)
and compare. Balanced full-block reads make this mapping-agnostic: BW scales
with n_units up to the plane ceiling in both models if the scheduling agrees.
"""
from __future__ import annotations
import json, os, subprocess, tempfile, sys

HBFSIM = "/home/adityaan/HBFSim/build/hbfsim"
BASE_CFG = "/home/adityaan/HBFSim/configs/validation/hbf_peak_ideal.toml"
HBF_BASE = 0x1800000000
PAGE = 16384
PPB = 510                       # pages per block (hbf_gen1_sandisk)
BLOCK_BYTES = PPB * PAGE
TOTAL_PLANES = 16 * 16 * 4      # channels x dies x planes/die = 1024

# HBFSim hbf_gen1_sandisk effective per-page timing for OUR (single-tR) model:
TR_NS = 55000.0                 # avg TLC sense (45/55/65 LSB/CSB/MSB)
TRC_DQ_NS = 30.0 + 164.0        # page-buffer hit + per-page DQ streaming
BUS_GBPS = 1600.0               # HBFSim direct_peer hbf_read_return_bw_GBps_per_stack

from vidur.memory_backends.hbf.plane_scheduler import PlaneScheduler, NANDRequest
from vidur.memory_backends.hbf.address_mapper import PhysicalAddr
from vidur.memory_backends.hbf.config_loader import NANDDieConfig, NANDStackConfig, SubarrayConfig
from vidur.memory_backends.base import OpType


def our_cfg():
    sa = SubarrayConfig(page_size_bytes=PAGE, pages_per_block=PPB, tR_ns=TR_NS, tRC_ns=TRC_DQ_NS)
    die = NANDDieConfig(num_planes=4, num_subarrays=32, blocks_per_plane=1024,
                        max_concurrent_per_plane=0, enable_multi_plane_cmds=True, max_plane_batch=4)
    stack = NANDStackConfig(num_channels=16, num_dies_per_channel=16)
    return sa, die, stack


def hbfsim_bw(n_units, tmp):
    trace = os.path.join(tmp, "t.trc"); cfg = os.path.join(tmp, "c.toml"); stats = os.path.join(tmp, "s.json")
    with open(trace, "w") as f:
        f.write("# t type addr size layer op data_class seq window token state stream compute tier_copy\n")
        for u in range(n_units):
            for p in range(PPB):
                addr = HBF_BASE + u * BLOCK_BYTES + p * PAGE
                f.write(f"0 R 0x{addr:x} {PAGE} 0 decode_kv_read COLD_KV 0 0 0 decode 0 0 0\n")
    base = open(BASE_CFG).read()
    base = base.replace('trace_file = ""', f'trace_file = "{trace}"')
    base = base.replace('output_stats_json_path = "output/validation_hbf_peak_ideal_stats.json"',
                        f'output_stats_json_path = "{stats}"')
    import re
    base = re.sub(r"expected_requests = \d+", f"expected_requests = {n_units*PPB}", base)
    open(cfg, "w").write(base)
    p = subprocess.run([HBFSIM, cfg], capture_output=True, text=True, timeout=300,
                       cwd="/home/adityaan/HBFSim")
    bw = None
    for line in (p.stdout + p.stderr).splitlines():
        if "aggregate bandwidth" in line:
            bw = float(line.split("aggregate bandwidth:")[1].split("GB/s")[0])
    return bw


def our_bw(n_units):
    sa, die, stack = our_cfg()
    sched = PlaneScheduler(die_cfg=die, sa_cfg=sa, stack_cfg=stack)
    spp = die.num_subarrays // die.num_planes
    rid = 0
    for u in range(n_units):
        plane = u % TOTAL_PLANES
        local_block = u // TOTAL_PLANES
        sa_id = plane * spp + (local_block % spp)
        addr = PhysicalAddr(channel_id=0, stack_id=0, die_id=0, plane_id=plane,
                            block_id=local_block, page_id=0, subarray_id=sa_id)
        sched.submit(NANDRequest(req_id=rid, op=OpType.READ, addr=addr,
                                 size_bytes=PPB * PAGE, issued_at=0.0))
        rid += 1
    comps = sched.drain()
    wall_ns = max(c.latency_ns for c in comps) if comps else 0.0
    bytes_read = n_units * PPB * PAGE
    # shared read-return bus cap: wall can't beat bytes / bus_BW
    wall_ns = max(wall_ns, bytes_read / BUS_GBPS)
    return (bytes_read / wall_ns) if wall_ns > 0 else 0.0   # bytes/ns = GB/s


def main():
    print(f"{'n_units':>8}{'bytes_MB':>10}{'HBFSim_GBps':>13}{'ours_GBps':>11}{'err_%':>9}")
    print("-" * 52)
    errs = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in [1, 4, 16, 64, 256, 1024]:
            h = hbfsim_bw(n, tmp)
            o = our_bw(n)
            e = (o - h) / h * 100 if h else 0.0
            errs.append(abs(e))
            mb = n * PPB * PAGE / 1e6
            print(f"{n:>8}{mb:>10.0f}{h if h else float('nan'):>13.1f}{o:>11.1f}{e:>+8.1f}%")
    print("-" * 52)
    print(f"max |err| = {max(errs):.1f}%   mean |err| = {sum(errs)/len(errs):.1f}%")


if __name__ == "__main__":
    main()
