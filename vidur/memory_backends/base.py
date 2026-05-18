"""
Abstract memory backend interface.

Backends implement this to replace chiplastic's analytical bandwidth/latency
formulas with cycle-accurate simulation (HBF plane scheduler, HBM bank
scheduler, or a combination of both).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DataClass(str, Enum):
    KV        = "kv"
    HOT_KV    = "hot_kv"
    COLD_KV   = "cold_kv"
    WEIGHT    = "weight"
    ACTIVATION = "activation"


class OpType(str, Enum):
    READ     = "read"
    WRITE    = "write"
    PREFETCH = "prefetch"


@dataclass
class MemRequest:
    req_id:       int
    op:           OpType
    size_bytes:   int
    data_class:   DataClass = DataClass.KV
    layer_id:     int = 0
    sequence_id:  int = 0
    token_id:     int = 0
    block_id:     int = -1   # logical KV block id; -1 = let backend assign
    timestamp_ns: float = 0.0


@dataclass
class MemCompletion:
    req_id:       int
    latency_ns:   float
    plane_id:     Optional[int] = None  # HBF: which plane served this
    die_id:       Optional[int] = None
    was_cache_hit: bool = False         # HBF: tRC (True) vs tR (False)


@dataclass
class PlaneStats:
    """Per-plane statistics from the HBF plane scheduler."""
    num_planes:              int
    plane_occupancies:       List[float]  # mean in-flight fraction per plane
    plane_stall_counts:      List[int]    # stall events per plane
    multi_plane_merge_count: int          # packets merged into multi-plane cmds
    total_cmds_issued:       int
    mean_batch_size:         float        # plane ops per cmd (ideal=num_planes)
    gini_coefficient:        float        # load imbalance across planes (0=perfect)
    active_plane_fraction:   float        # fraction of planes that saw traffic
    cache_hits:              int          # tRC accesses
    cache_misses:            int          # tR accesses
    suspend_events:          int          # program suspend/resume cycles
    current_time_ns:         float


@dataclass
class HBMStats:
    """Row-buffer statistics from the HBM bank scheduler."""
    row_hits:      int
    row_misses:    int
    row_conflicts: int
    total_reqs:    int

    @property
    def hit_rate(self) -> float:
        if self.total_reqs == 0:
            return 0.0
        return self.row_hits / self.total_reqs


class MemoryBackend(ABC):
    """
    Abstract base for pluggable memory backends.

    Usage pattern in the simulator:
        backend.submit(req)          # enqueue a request at current simulated time
        completions = backend.drain()  # run until all submitted requests finish
        latency_ns = max(c.latency_ns for c in completions)
    """

    @abstractmethod
    def submit(self, req: MemRequest) -> None:
        """Enqueue a memory request."""

    @abstractmethod
    def advance(self, delta_ns: float) -> List[MemCompletion]:
        """Advance the internal clock by delta_ns, return completions in that window."""

    @abstractmethod
    def drain(self) -> List[MemCompletion]:
        """Run until all submitted-but-not-yet-completed requests finish."""

    @abstractmethod
    def get_read_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        """Synchronous single-request latency estimate (no queuing)."""

    @abstractmethod
    def get_write_latency_ns(self, size_bytes: int, data_class: DataClass) -> float:
        """Synchronous single-request write latency estimate."""

    @abstractmethod
    def plane_stats(self) -> Optional[PlaneStats]:
        """Return plane-level stats, or None for backends without plane model."""

    @abstractmethod
    def hbm_stats(self) -> Optional[HBMStats]:
        """Return HBM bank stats, or None for backends without HBM model."""

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset accumulated statistics (call between epochs if desired)."""
