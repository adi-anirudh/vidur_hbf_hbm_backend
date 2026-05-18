"""
Ramulator HBM backend.

Models the HBM tier using timing parameters from the HBFSim TOML [hbm_media]
section (which itself uses HBM3E.yaml timing via Ramulator2 internally).

This backend handles weight and activation traffic — data that lives in HBM
rather than the HBF flash tier.  For full-stack experiments, combine with
HBFSimBackend: route KV cache through HBFSimBackend, weights/activations
through RamulatorBackend.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from vidur.memory_backends.base import (
    DataClass, HBMStats, MemCompletion, MemoryBackend, MemRequest, OpType, PlaneStats,
)
from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader
from vidur.memory_backends.ramulator.bank_scheduler import HBMBankScheduler, HBMRequest


_HBM_DATA_CLASSES = {DataClass.WEIGHT, DataClass.ACTIVATION}
_HBF_DATA_CLASSES = {DataClass.KV, DataClass.HOT_KV, DataClass.COLD_KV}


class RamulatorBackend(MemoryBackend):
    """
    HBM bank-level memory backend driven by HBFSim timing parameters.

    For experiments that need only the HBM tier (weights, activations) without
    HBF flash, or as the complementary backend alongside HBFSimBackend.

    Timing model (FRFCFS row-buffer policy):
      Row hit:      tCL            ≈ 14 ns @ HBM3E
      Row miss:     tRCD + tCL     ≈ 28 ns
      Row conflict: tRP+tRCD+tCL   ≈ 42 ns
    """

    def __init__(
        self,
        config_path: str | Path,
        num_channels:          Optional[int] = None,  # override TOML value
        num_banks_per_channel: int = 8,
        row_size_bytes:        int = 2048,
    ):
        cfg = HBFSimConfigLoader(config_path).load()
        hbm = cfg.hbm

        # HBFSim stores total banks; infer channel count
        total_banks = hbm.num_banks
        num_ch = num_channels or max(1, total_banks // num_banks_per_channel)

        self._sched = HBMBankScheduler(
            num_channels=num_ch,
            num_banks_per_channel=num_banks_per_channel,
            tCL_cycles=hbm.tCL_cycles,
            tRCD_cycles=hbm.tRCD_cycles,
            tRP_cycles=hbm.tRP_cycles,
            freq_mhz=hbm.freq_mhz,
            row_size_bytes=row_size_bytes,
        )
        self._bandwidth_gbps = hbm.bandwidth_gbps
        self._req_counter = 0
        self._current_time_ns: float = 0.0

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    def submit(self, req: MemRequest) -> None:
        # Derive a deterministic address from block_id / layer / sequence
        addr = self._logical_addr(req)
        hreq = HBMRequest(
            req_id=req.req_id,
            op=req.op,
            addr=addr,
            size_bytes=req.size_bytes,
            issued_at=self._current_time_ns,
        )
        self._sched.submit(hreq)

    def submit_at_addr(
        self, req_id: int, op: OpType, addr: int, size_bytes: int
    ) -> None:
        """Submit a request at an explicit physical HBM address (bypasses logical mapping)."""
        self._sched.submit(HBMRequest(
            req_id=req_id, op=op, addr=addr,
            size_bytes=size_bytes, issued_at=self._current_time_ns,
        ))

    def reset_step(self) -> None:
        """
        Reset the simulation clock and bank busy-until state for a new batch step.

        Vidur's event loop handles inter-batch timing; the bank scheduler only
        needs to model intra-batch conflicts.  Row-buffer open_row state is
        also cleared — cross-batch row-buffer warmth is not tracked at this level.
        """
        self._current_time_ns = 0.0
        self._sched._time_ns = 0.0
        for bank in self._sched._banks:
            bank.busy_until = 0.0
            bank.open_row = -1

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        completions = self._sched.advance(delta_ns)
        self._current_time_ns += delta_ns
        return completions

    def drain(self) -> List[MemCompletion]:
        completions = self._sched.drain()
        self._current_time_ns = self._sched.current_time_ns()
        return completions

    def get_read_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        cycle_ns = self._sched._cycle_ns
        # Pessimistic: row conflict
        base = (self._sched._tRP + self._sched._tRCD + self._sched._tCL)
        return base + size_bytes / (self._bandwidth_gbps * 1e9 / 8) * 1e9

    def get_write_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        return self.get_read_latency_ns(size_bytes, data_class)

    def plane_stats(self) -> Optional[PlaneStats]:
        return None  # HBM has no plane model

    def hbm_stats(self) -> HBMStats:
        return HBMStats(
            row_hits=self._sched.row_hits(),
            row_misses=self._sched.row_misses(),
            row_conflicts=self._sched.row_conflicts(),
            total_reqs=self._sched.total_reqs(),
        )

    def reset_stats(self) -> None:
        # Preserve bank state (row buffers) but reset counters
        self._sched._row_hits = 0
        self._sched._row_misses = 0
        self._sched._row_conflicts = 0
        self._sched._total_reqs = 0

    # ------------------------------------------------------------------
    # Address derivation
    # ------------------------------------------------------------------

    def _logical_addr(self, req: MemRequest) -> int:
        """
        Derive a deterministic byte address from request metadata.
        Weights: row-major layout by (layer_id, sequence_id).
        KV / other: block_id * page_size.
        """
        page = 64 * 1024  # 64 KB granularity
        if req.data_class in _HBM_DATA_CLASSES:
            # Weights are layer-major; different layers go to different rows
            return (req.layer_id * 1024 + req.sequence_id) * page
        else:
            bid = max(0, req.block_id)
            return bid * page
