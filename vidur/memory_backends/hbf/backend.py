"""
HBFSim memory backend.

Drives the PlaneScheduler with KV requests from chiplastic and optionally
emits HBFSim-compatible trace files for offline full-fidelity validation.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from vidur.memory_backends.base import (
    DataClass, HBMStats, MemCompletion, MemoryBackend, MemRequest, OpType, PlaneStats,
)
from vidur.memory_backends.hbf.address_mapper import PhysicalAddr, PlacementPolicy, PlaneAddressMapper
from vidur.memory_backends.hbf.config_loader import HBFSimConfig, HBFSimConfigLoader
from vidur.memory_backends.hbf.plane_scheduler import NANDRequest, PlaneScheduler


class HBFSimBackend(MemoryBackend):
    """
    Plane-accurate HBF memory backend.

    Two modes controlled by emit_trace:
      False (default) — online Python plane model, in-process, no subprocess.
      True            — additionally writes a HBFSim-compatible trace file
                        at trace_dir for offline full-fidelity validation.

    Placement policies:
      STRIPE_ACROSS_PLANES  — best for large-batch decode (maximises multi-plane merges)
      PACK_BY_SEQUENCE      — best for prefetch pipelines (exploits tRC cache-read path)
      INTERLEAVE_BY_TOKEN   — balanced; good for mixed workloads
    """

    def __init__(
        self,
        config_path: str | Path,
        placement_policy: PlacementPolicy = PlacementPolicy.STRIPE_ACROSS_PLANES,
        num_layers: int = 32,
        emit_trace: bool = False,
        trace_dir:  str | Path = "hbfsim_traces",
    ):
        cfg = HBFSimConfigLoader(config_path).load()
        self._cfg = cfg
        self._emit_trace = emit_trace
        self._trace_dir  = Path(trace_dir)

        self._scheduler = PlaneScheduler(
            die_cfg=cfg.nand_die,
            sa_cfg=cfg.subarray,
            stack_cfg=cfg.nand_stack,
        )
        self._mapper = PlaneAddressMapper(
            sa_cfg=cfg.subarray,
            die_cfg=cfg.nand_die,
            stack_cfg=cfg.nand_stack,
            policy=placement_policy,
            num_layers=num_layers,
        )

        self._current_time_ns: float = 0.0
        self._req_counter: int = 0
        self._pending_meta: Dict[int, MemRequest] = {}  # req_id → original request

        # Trace file state
        self._trace_rows: List[dict] = []

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    def submit(self, req: MemRequest) -> None:
        addr = self._mapper.map_block(
            block_id=max(0, req.block_id),
            layer_id=req.layer_id,
            sequence_id=req.sequence_id,
            token_id=req.token_id,
        )
        nand_req = NANDRequest(
            req_id=req.req_id,
            op=req.op,
            addr=addr,
            size_bytes=req.size_bytes,
            issued_at=self._current_time_ns,
        )
        self._scheduler.submit(nand_req)
        self._pending_meta[req.req_id] = req

        if self._emit_trace:
            self._record_trace(req, addr)

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        completions = self._scheduler.advance(delta_ns)
        self._current_time_ns += delta_ns
        for c in completions:
            self._pending_meta.pop(c.req_id, None)
        return completions

    def drain(self) -> List[MemCompletion]:
        completions = self._scheduler.drain()
        self._current_time_ns = self._scheduler.current_time_ns()
        for c in completions:
            self._pending_meta.pop(c.req_id, None)
        return completions

    def get_read_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        """
        Single-request cold-read estimate (no queuing, I/O-gated):
          every page costs tR (I/O bottleneck dominates internal sense time).
        """
        page_bytes = self._cfg.subarray.page_size_bytes
        pages = max(1, math.ceil(size_bytes / page_bytes))
        return self._cfg.subarray.tR_ns * pages

    def get_write_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        page_bytes = self._cfg.subarray.page_size_bytes
        pages = max(1, math.ceil(size_bytes / page_bytes))
        return self._cfg.subarray.tPROG_ns * pages

    def peak_bandwidth_gbps(self) -> float:
        """
        Peak read bandwidth when all planes execute a multi-plane command in parallel.

        BW = total_planes x page_size_bytes / tR_ns   [bytes/ns = GB/s]

        This is the hardware ceiling; actual throughput is lower under mixed
        workloads or when not all planes have requests queued.
        """
        total_planes = self._mapper.total_planes
        page_bytes   = self._cfg.subarray.page_size_bytes
        return total_planes * page_bytes / self._cfg.subarray.tR_ns  # bytes/ns = GB/s

    def capacity_per_plane_bytes(self) -> int:
        """Capacity of one plane: blocks_per_plane x pages_per_block x page_size."""
        return self._mapper.capacity_per_plane_bytes()

    def total_capacity_bytes(self) -> int:
        """Total flash capacity across all planes."""
        return self._mapper.total_capacity_bytes()

    def plane_stats(self) -> PlaneStats:
        sched = self._scheduler
        occupancies = sched.plane_occupancies()
        stalls      = sched.plane_stall_counts()
        n = len(occupancies)
        active = sum(1 for o in occupancies if o > 0.0)
        return PlaneStats(
            num_planes=sched._total_planes,  # channels × dies × planes/die
            plane_occupancies=occupancies,
            plane_stall_counts=stalls,
            multi_plane_merge_count=sched.multi_plane_merge_count(),
            total_cmds_issued=sched.total_cmds_issued(),
            mean_batch_size=sched.mean_batch_size(),
            gini_coefficient=sched.gini(),
            active_plane_fraction=active / n if n > 0 else 0.0,
            cache_hits=sched.cache_hit_count(),
            cache_misses=sched.cache_miss_count(),
            suspend_events=sched.suspend_event_count(),
            current_time_ns=sched.current_time_ns(),
        )

    def hbm_stats(self) -> Optional[HBMStats]:
        return None  # HBF-only backend; see RamulatorBackend for HBM

    def reset_stats(self) -> None:
        self._scheduler = PlaneScheduler(
            die_cfg=self._cfg.nand_die,
            sa_cfg=self._cfg.subarray,
            stack_cfg=self._cfg.nand_stack,
        )
        self._trace_rows.clear()

    # ------------------------------------------------------------------
    # Convenience: batch submit for a full decode step
    # ------------------------------------------------------------------

    def submit_decode_step(
        self,
        sequence_ids: List[int],
        token_id:     int,
        num_layers:   int,
        kv_block_ids: Dict[int, List[int]],  # seq_id → [block_id per layer]
        kv_block_size_bytes: int = 16 * 1024,
        data_class: DataClass = DataClass.COLD_KV,
        hot_fraction: float = 0.0,  # accepted but unused; routing is CompositeBackend's job
    ) -> List[int]:
        """
        Submit all KV reads for one decode step across all sequences and layers.

        Returns the list of req_ids submitted.  Call drain() after to get
        latencies — the max latency across all completions is the decode memory
        stall for this step.

        The placement policy in the address mapper determines how effectively
        multi-plane commands are formed:
          STRIPE_ACROSS_PLANES: seqs on different planes → multi-plane merge per layer.
          PACK_BY_SEQUENCE:     each seq's layers form a cache-read chain (tR then tRCs).
        """
        req_ids = []
        for seq_id in sequence_ids:
            blocks = kv_block_ids.get(seq_id, [])
            for layer_id, block_id in enumerate(blocks[:num_layers]):
                rid = self._next_req_id()
                req = MemRequest(
                    req_id=rid,
                    op=OpType.READ,
                    size_bytes=kv_block_size_bytes,
                    data_class=data_class,
                    layer_id=layer_id,
                    sequence_id=seq_id,
                    token_id=token_id,
                    block_id=block_id,
                    timestamp_ns=self._current_time_ns,
                )
                self.submit(req)
                req_ids.append(rid)
        return req_ids

    def submit_plane_reads(self, plane_page_counts: dict, token_id: int = 0) -> None:
        """
        Submit one aggregate read request per plane.

        plane_page_counts: {plane_id: num_pages} — pre-computed per-plane load.
        token_id controls row_offset for multi-plane command merging.
        All planes with the same token_id and row_offset merge into one command.

        This is more efficient than submitting one request per KV block when the
        block count per plane is large (e.g. long-context decode).
        """
        spp            = self._mapper.subarrays_per_plane
        planes_per_die = self._mapper._planes_per_die
        planes_per_ch  = planes_per_die * self._cfg.nand_stack.num_dies_per_channel
        page_bytes     = self._cfg.subarray.page_size_bytes
        row_offset     = token_id % self._cfg.subarray.pages_per_block

        for plane_id, num_pages in plane_page_counts.items():
            if num_pages <= 0:
                continue
            channel_id    = plane_id // planes_per_ch
            die_in_ch     = (plane_id % planes_per_ch) // planes_per_die
            subarray_id   = plane_id * spp  # first subarray in the plane
            addr = PhysicalAddr(
                channel_id=channel_id,
                stack_id=0,
                die_id=die_in_ch,
                plane_id=plane_id,
                block_id=0,
                page_id=row_offset,
                subarray_id=subarray_id,
            )
            nand_req = NANDRequest(
                req_id=self._next_req_id(),
                op=OpType.READ,
                addr=addr,
                size_bytes=num_pages * page_bytes,
                issued_at=self._current_time_ns,
            )
            self._scheduler.submit(nand_req)

    def submit_prefetch(
        self,
        sequence_id: int,
        layer_id:    int,
        kv_size_bytes: int,
        block_id:    int = -1,
        token_id:    int = 0,
    ) -> int:
        """Submit a single-layer KV prefetch. Returns req_id."""
        rid = self._next_req_id()
        req = MemRequest(
            req_id=rid,
            op=OpType.PREFETCH,
            size_bytes=kv_size_bytes,
            data_class=DataClass.HOT_KV,
            layer_id=layer_id,
            sequence_id=sequence_id,
            token_id=token_id,
            block_id=block_id if block_id >= 0 else sequence_id * 32 + layer_id,
            timestamp_ns=self._current_time_ns,
        )
        self.submit(req)
        return rid

    # ------------------------------------------------------------------
    # HBFSim trace emission (for offline validation)
    # ------------------------------------------------------------------

    def emit_trace(self, path: Optional[str | Path] = None) -> Path:
        """
        Write a HBFSim-compatible trace file.

        Trace format (space-delimited):
          <timestamp_cycles> <type> <addr_hex> <size> <layer_id> <op_name>
          <data_class> <seq_id> 0 <token_id> prefill 0 0 0

        The file can be pointed to by [requester] trace_file in a HBFSim
        TOML config for full-fidelity C++ simulation.
        """
        out = Path(path) if path else self._trace_dir / "chiplastic.trace"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Trace uses cycle timestamps; assume 1 GHz reference (1 cycle = 1 ns)
        # so the C++ HBFSim simulator can be configured to match.
        with out.open("w") as f:
            for row in self._trace_rows:
                ts_cycles = int(row["timestamp_ns"])
                op_str = {
                    OpType.READ:     "R",
                    OpType.WRITE:    "W",
                    OpType.PREFETCH: "PREFETCH",
                }[row["op"]]
                dc_str = row["data_class"].value.upper()
                addr = row["logical_addr"]
                size = row["size_bytes"]
                layer = row["layer_id"]
                seq   = row["sequence_id"]
                token = row["token_id"]
                f.write(
                    f"{ts_cycles} {op_str} 0x{addr:016x} {size} "
                    f"{layer} kv_{layer} {dc_str} "
                    f"{seq} 0 {token} prefill 0 0 0\n"
                )
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _record_trace(self, req: MemRequest, addr: PhysicalAddr) -> None:
        # Logical address: encode (stack, die, plane, sa, block, page) compactly
        logical_addr = (
            (addr.stack_id    & 0xFF) << 48 |
            (addr.die_id      & 0xFF) << 40 |
            (addr.plane_id    & 0xFF) << 32 |
            (addr.subarray_id & 0xFF) << 24 |
            (addr.block_id    & 0xFFFF) << 8 |
            (addr.page_id     & 0xFF)
        )
        self._trace_rows.append({
            "timestamp_ns":  self._current_time_ns,
            "op":            req.op,
            "logical_addr":  logical_addr,
            "size_bytes":    req.size_bytes,
            "layer_id":      req.layer_id,
            "sequence_id":   req.sequence_id,
            "token_id":      req.token_id,
            "data_class":    req.data_class,
        })
