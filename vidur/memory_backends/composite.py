"""
CompositeBackend: serialized two-tier memory.

Rule: at any simulated time only one tier is active.

  COLD_KV / KV         → HBFSimBackend   (NAND plane scheduler)
  HOT_KV / WEIGHT / ACT → RamulatorBackend (HBM bank scheduler)

Serialization is enforced in drain():
  1. Drain HBF (cold, slow).  HBM idles.
  2. Advance HBM clock to HBF end time.
  3. Issue deferred HBM requests (timestamp = HBF end).
  4. Drain HBM.
  5. Adjust HBM completion latencies so they include the HBF wait —
     max(all completion latencies) then equals the true total stall.

submit() defers all HBM-bound requests into _deferred_hbm so that nothing
enters HBM's scheduler queue until HBF has finished.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from vidur.memory_backends.base import (
    DataClass, HBMStats, MemCompletion, MemoryBackend, MemRequest, OpType, PlaneStats,
)

_HBM_CLASSES = {DataClass.HOT_KV, DataClass.WEIGHT, DataClass.ACTIVATION}
_HBF_CLASSES = {DataClass.COLD_KV, DataClass.KV}


class CompositeBackend(MemoryBackend):
    """
    Serialized HBF + HBM backend: only one tier active at a time.

    Args:
        hbf: HBFSimBackend (NAND plane scheduler owns cold KV).
        hbm: RamulatorBackend (HBM bank scheduler owns hot KV).
    """

    def __init__(self, hbf, hbm) -> None:
        self._hbf = hbf
        self._hbm = hbm
        self._deferred_hbm: List[MemRequest] = []
        self._req_counter = 0

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    def submit(self, req: MemRequest) -> None:
        if req.data_class in _HBM_CLASSES:
            # Defer: HBM must not start until HBF finishes.
            self._deferred_hbm.append(req)
        else:
            self._hbf.submit(req)

    def drain(self) -> List[MemCompletion]:
        """
        Serialized drain: HBF runs first, then HBM.

        HBM completion latencies are adjusted to include the HBF wait so that
        max(c.latency_ns for c in result) equals the true end-to-end stall.
        """
        hbm_t0_ns = self._hbm._current_time_ns

        # --- Phase 1: drain HBF (cold tier) ---
        hbf_completions = self._hbf.drain()
        hbf_end_ns = self._hbf._current_time_ns

        # --- Phase 2: advance HBM clock to HBF end (HBM was idle) ---
        gap_ns = hbf_end_ns - self._hbm._current_time_ns
        if gap_ns > 0:
            self._hbm.advance(gap_ns)

        # --- Phase 3: issue deferred HBM requests (now timestamped at hbf_end) ---
        for req in self._deferred_hbm:
            self._hbm.submit(req)
        self._deferred_hbm.clear()

        # --- Phase 4: drain HBM (hot tier) ---
        hbm_completions = self._hbm.drain()

        # --- Phase 5: adjust HBM latencies to include the HBF wait ---
        hbf_stall_ns = hbf_end_ns - hbm_t0_ns
        adjusted_hbm = [
            MemCompletion(
                req_id=c.req_id,
                latency_ns=c.latency_ns + hbf_stall_ns,
                plane_id=c.plane_id,
                die_id=c.die_id,
                was_cache_hit=c.was_cache_hit,
            )
            for c in hbm_completions
        ]

        return hbf_completions + adjusted_hbm

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        # Between-batch clock tick: advance both tiers together.
        # Deferred HBM requests are not dispatched here; they wait for drain().
        return self._hbf.advance(delta_ns) + self._hbm.advance(delta_ns)

    def get_read_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        if data_class in _HBM_CLASSES:
            return self._hbm.get_read_latency_ns(size_bytes, data_class)
        return self._hbf.get_read_latency_ns(size_bytes, data_class)

    def get_write_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        if data_class in _HBM_CLASSES:
            return self._hbm.get_write_latency_ns(size_bytes, data_class)
        return self._hbf.get_write_latency_ns(size_bytes, data_class)

    def plane_stats(self) -> Optional[PlaneStats]:
        return self._hbf.plane_stats()

    def hbm_stats(self) -> Optional[HBMStats]:
        return self._hbm.hbm_stats()

    def reset_stats(self) -> None:
        self._hbf.reset_stats()
        self._hbm.reset_stats()
        self._deferred_hbm.clear()

    # ------------------------------------------------------------------
    # Decode step helper
    # ------------------------------------------------------------------

    def submit_decode_step(
        self,
        sequence_ids:        List[int],
        token_id:            int,
        num_layers:          int,
        kv_block_ids:        Dict[int, List[int]],
        kv_block_size_bytes: int = 16 * 1024,
        hot_fraction:        float = 0.0,
        data_class:          DataClass = DataClass.COLD_KV,  # unused; split controls routing
    ) -> List[int]:
        """
        Submit one decode step split across HBM (hot) and HBF (cold).

        hot_fraction of sequence_ids → HOT_KV → deferred to HBM.
        Remainder → COLD_KV → HBF plane scheduler immediately.

        Call drain() after to get completions. max(latency_ns) is the total
        memory stall (HBF serial time + HBM serial time).
        """
        n_hot    = int(len(sequence_ids) * hot_fraction)
        hot_ids  = sequence_ids[:n_hot]
        cold_ids = sequence_ids[n_hot:]
        req_ids  = []

        # Hot → deferred HBM (submitted in drain() after HBF finishes)
        for seq_id in hot_ids:
            blocks = kv_block_ids.get(seq_id, [])
            for layer_id, block_id in enumerate(blocks[:num_layers]):
                rid = self._next_req_id()
                req = MemRequest(
                    req_id=rid,
                    op=OpType.READ,
                    size_bytes=kv_block_size_bytes,
                    data_class=DataClass.HOT_KV,
                    layer_id=layer_id,
                    sequence_id=seq_id,
                    token_id=token_id,
                    block_id=block_id,
                    timestamp_ns=0.0,  # overwritten at issue time in drain()
                )
                self._deferred_hbm.append(req)
                req_ids.append(rid)

        # Cold → HBF plane scheduler immediately
        if cold_ids:
            cold_block_ids = {s: kv_block_ids[s] for s in cold_ids if s in kv_block_ids}
            req_ids += self._hbf.submit_decode_step(
                sequence_ids=cold_ids,
                token_id=token_id,
                num_layers=num_layers,
                kv_block_ids=cold_block_ids,
                kv_block_size_bytes=kv_block_size_bytes,
                data_class=DataClass.COLD_KV,
            )

        return req_ids

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter
