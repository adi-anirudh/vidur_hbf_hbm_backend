"""
Analytical memory backend — wraps chiplastic's original bandwidth/latency formulas.

This is the default fallback and preserves identical behaviour to the pre-backend
code path. Switching to HBFSimBackend or RamulatorBackend is a drop-in replacement.
"""

from typing import List, Optional

from vidur.memory_backends.base import (
    DataClass, HBMStats, MemCompletion, MemoryBackend, MemRequest, OpType, PlaneStats,
)


class AnalyticalBackend(MemoryBackend):
    """
    Reconstructs latency from simple bandwidth formulas:
        latency = bytes / bandwidth

    Parameters mirror the NUMAConfig fields already present in chiplastic.
    """

    def __init__(
        self,
        local_bandwidth_gbps: float = 1600.0,   # HBM local die
        remote_bandwidth_gbps: float = 900.0,    # inter-die
        nvlink_bandwidth_gbps: float = 300.0,
        pcie_bandwidth_gbps: float = 64.0,
        local_latency_ns: float = 60.0,
        remote_latency_ns: float = 120.0,
    ):
        self._local_bw  = local_bandwidth_gbps  * 1e9 / 8   # bytes/s
        self._remote_bw = remote_bandwidth_gbps * 1e9 / 8
        self._nvlink_bw = nvlink_bandwidth_gbps * 1e9 / 8
        self._pcie_bw   = pcie_bandwidth_gbps   * 1e9 / 8
        self._local_lat_ns  = local_latency_ns
        self._remote_lat_ns = remote_latency_ns

        self._pending: List[MemRequest] = []
        self._req_counter = 0

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    def submit(self, req: MemRequest) -> None:
        self._pending.append(req)

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        completions = [self._complete(r) for r in self._pending]
        self._pending.clear()
        return completions

    def drain(self) -> List[MemCompletion]:
        completions = [self._complete(r) for r in self._pending]
        self._pending.clear()
        return completions

    def get_read_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        return self._remote_lat_ns + size_bytes / self._remote_bw * 1e9

    def get_write_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        return self._remote_lat_ns + size_bytes / self._remote_bw * 1e9

    def plane_stats(self) -> Optional[PlaneStats]:
        return None  # analytical backend has no plane model

    def hbm_stats(self) -> Optional[HBMStats]:
        return None

    def reset_stats(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _complete(self, req: MemRequest) -> MemCompletion:
        bw = self._local_bw if req.data_class in (DataClass.KV, DataClass.HOT_KV) \
             else self._remote_bw
        lat_ns = req.timestamp_ns + req.size_bytes / bw * 1e9
        return MemCompletion(req_id=req.req_id, latency_ns=lat_ns)
