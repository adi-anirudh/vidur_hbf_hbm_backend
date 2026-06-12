"""
Cycle-accurate NAND plane scheduler.

Mirrors the logic of HBFSim's NANDDie + Subarray C++ classes:

  Subarray level:
  - Serialises one command at a time.
  - Cache-read optimisation: if the requested block is already in the page
    buffer, pay tRC (30 ns) instead of tR (50 μs) — a 1666× difference.
  - Program suspend/resume: a write in flight can be paused to service an
    incoming read (pay suspend_cycles + read + resume_cycles).

  Die / plane level:
  - Subarrays are partitioned into planes (plane_id = sa_id // sas_per_plane).
  - Per-plane in-flight counter enforces max_concurrent_per_plane.
  - Multi-plane command merging: requests that share the same (op_type, row_offset)
    and land on different planes are merged into one command that executes in
    parallel — all planes complete at max(individual latencies), but one die-level
    command slot is consumed rather than N.  This is the key mechanism that turns
    batch-decode reads (all seqs at the same token_id → same row_offset) into a
    single tR cost for the entire batch.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vidur.memory_backends.base import DataClass, MemCompletion, OpType
from vidur.memory_backends.hbf.address_mapper import PhysicalAddr
from vidur.memory_backends.hbf.config_loader import NANDDieConfig, NANDStackConfig, SubarrayConfig


# ---------------------------------------------------------------------------
# Internal request / state types
# ---------------------------------------------------------------------------

@dataclass
class NANDRequest:
    req_id:    int
    op:        OpType
    addr:      PhysicalAddr
    size_bytes: int
    issued_at: float  # simulation time (ns) when submit() was called
    # Number of DISTINCT NAND blocks this request's pages span. 0 = legacy
    # contiguous behaviour (pages laid out sequentially from addr.block_id).
    # >0 marks an aggregate per-plane read covering `num_blocks` independent KV
    # blocks, each of which incurs its own tR page-buffer miss — used to make
    # the analytical per-plane read match the per-KV-block trace-level sim.
    num_blocks: int = 0


@dataclass
class _SubarrayState:
    sa_id:          int
    busy_until:     float = 0.0          # ns; 0 = idle
    current_block:  int   = -1           # block in page buffer (-1 = empty)
    current_page:   int   = -1           # page offset within that block (-1 = empty)
    # Program suspend state
    suspended:      bool  = False
    suspended_req:  Optional[NANDRequest] = None
    suspended_remaining_ns: float = 0.0
    reads_since_suspend: int = 0


@dataclass
class _ActiveCmd:
    """Multi-plane command grouping up to max_plane_batch plane operations."""
    op_type:     OpType
    row_offset:  int
    plane_busy:  Dict[int, NANDRequest] = field(default_factory=dict)  # plane→req
    finish_at:   float = 0.0   # when the last plane op completes


# ---------------------------------------------------------------------------
# PlaneScheduler
# ---------------------------------------------------------------------------

class PlaneScheduler:
    """
    Python implementation of HBFSim's NANDDie + Subarray timing model.

    Operates in simulated nanoseconds. Call advance(delta_ns) to move time
    forward; it returns MemCompletion objects for every request that finishes
    within the window.
    """

    def __init__(
        self,
        die_cfg:   NANDDieConfig,
        sa_cfg:    SubarrayConfig,
        stack_cfg: NANDStackConfig,
    ):
        self._die   = die_cfg
        self._sa    = sa_cfg
        self._stack = stack_cfg

        # ── Geometry ──────────────────────────────────────────────────────
        # Subarrays are a die-level concept; planes partition a die's subarrays.
        self._subarrays_per_plane = die_cfg.num_subarrays // die_cfg.num_planes

        # Total exposed planes across the full system:
        #   channels × dies_per_channel × planes_per_die
        self._total_planes = (
            stack_cfg.num_channels
            * stack_cfg.num_dies_per_channel
            * die_cfg.num_planes
        )

        # Total subarrays in the system (global flat index)
        self._total_subarrays = self._total_planes * self._subarrays_per_plane

        # Per-plane concurrency cap (max simultaneous subarray ops within a plane)
        self._plane_cap = (
            die_cfg.max_concurrent_per_plane
            if die_cfg.max_concurrent_per_plane > 0
            else self._subarrays_per_plane
        )

        # ── Per-subarray state (one entry per global subarray) ─────────────
        self._subarrays: List[_SubarrayState] = [
            _SubarrayState(sa_id=i) for i in range(self._total_subarrays)
        ]

        # ── Per-plane state (one entry per global plane) ───────────────────
        self._plane_in_flight:    List[int]   = [0]   * self._total_planes
        self._plane_stall_counts: List[int]   = [0]   * self._total_planes
        self._plane_active_ns:    List[float] = [0.0] * self._total_planes

        # Active multi-plane commands
        self._active_cmds: List[_ActiveCmd] = []

        # Request queues
        self._pending:   deque[NANDRequest] = deque()
        self._completed: List[MemCompletion] = []

        # Current simulated time
        self._time_ns: float = 0.0

        # Statistics
        self._plane_last_sample:  float = 0.0
        self._multi_plane_merges: int = 0
        self._total_cmds:         int = 0
        self._cache_hits:         int = 0
        self._cache_misses:       int = 0
        self._suspend_events:     int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self, req: NANDRequest) -> None:
        """Enqueue a request for dispatch."""
        self._pending.append(req)

    def has_pending(self) -> bool:
        return bool(self._pending) or any(
            sa.busy_until > self._time_ns for sa in self._subarrays
        )

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        """
        Advance simulation by delta_ns nanoseconds.
        Returns all requests that completed within this window.
        """
        end_ns = self._time_ns + delta_ns
        completed: List[MemCompletion] = []

        while True:
            self._dispatch_pending()

            next_ev = self._next_event_ns()
            if next_ev is None or next_ev > end_ns:
                self._time_ns = end_ns
                break

            # Jump to next event
            self._time_ns = next_ev
            completed.extend(self._collect_completions())

        return completed

    def drain(self) -> List[MemCompletion]:
        """Process until all submitted requests complete."""
        completed: List[MemCompletion] = []
        max_iters = 10_000_000
        for _ in range(max_iters):
            self._dispatch_pending()
            next_ev = self._next_event_ns()
            if next_ev is None:
                break
            self._time_ns = next_ev
            completed.extend(self._collect_completions())
        return completed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def plane_stall_counts(self) -> List[int]:
        return list(self._plane_stall_counts)

    def plane_occupancies(self) -> List[float]:
        """Mean in-flight fraction per plane over simulation so far."""
        total_ns = self._time_ns
        if total_ns <= 0:
            return [0.0] * self._total_planes
        return [act / total_ns for act in self._plane_active_ns]

    def multi_plane_merge_count(self) -> int:
        return self._multi_plane_merges

    def total_cmds_issued(self) -> int:
        return self._total_cmds

    def mean_batch_size(self) -> float:
        if self._total_cmds == 0:
            return 0.0
        return (self._total_cmds + self._multi_plane_merges) / self._total_cmds

    def gini(self) -> float:
        """Gini coefficient of per-plane stall counts (0=perfect balance)."""
        counts = [float(c) for c in self._plane_stall_counts]
        total = sum(counts)
        if total == 0:
            return 0.0
        n = len(counts)
        counts.sort()
        cum = 0.0
        for i, c in enumerate(counts):
            cum += (2 * (i + 1) - n - 1) * c
        return cum / (n * total)

    def cache_hit_count(self) -> int:
        return self._cache_hits

    def cache_miss_count(self) -> int:
        return self._cache_misses

    def suspend_event_count(self) -> int:
        return self._suspend_events

    def current_time_ns(self) -> float:
        return self._time_ns

    # ------------------------------------------------------------------
    # Dispatch logic (mirrors NANDDie::tick / accept)
    # ------------------------------------------------------------------

    def _plane_of(self, sa_id: int) -> int:
        # global_subarray_id = plane_id × subarrays_per_plane + subarray_in_plane
        # so plane_id = global_subarray_id // subarrays_per_plane
        return sa_id // self._subarrays_per_plane

    def _dispatch_pending(self) -> None:
        """Try to dispatch as many pending requests as possible."""
        still_pending: deque[NANDRequest] = deque()

        while self._pending:
            req = self._pending.popleft()
            sa_id = req.addr.subarray_id  # global flat index across all channels/dies/planes
            plane = self._plane_of(sa_id)

            # Check plane capacity
            if self._plane_in_flight[plane] >= self._plane_cap:
                self._plane_stall_counts[plane] += 1
                still_pending.append(req)
                continue

            # Subarray busy — must wait for it to drain
            sa = self._subarrays[sa_id]
            if sa.busy_until > self._time_ns:
                still_pending.append(req)
                continue

            # Try multi-plane command merge
            if self._die.enable_multi_plane_cmds:
                cmd_idx = self._find_joinable_cmd(req.op, req.addr.row_offset, plane)
                if cmd_idx >= 0:
                    self._merge_into_cmd(cmd_idx, req, sa_id, plane)
                    self._multi_plane_merges += 1
                    continue

            # Dispatch to subarray
            self._issue_to_subarray(req, sa_id, plane)

        self._pending = still_pending

    def _find_joinable_cmd(self, op: OpType, row: int, plane: int) -> int:
        for i, cmd in enumerate(self._active_cmds):
            if cmd.op_type != op:
                continue
            if cmd.row_offset != row:
                continue
            if plane in cmd.plane_busy:
                continue
            if len(cmd.plane_busy) >= self._die.max_plane_batch:
                continue
            return i
        return -1

    def _merge_into_cmd(
        self, cmd_idx: int, req: NANDRequest, sa_id: int, plane: int
    ) -> None:
        sa = self._subarrays[sa_id]
        lat = self._subarray_latency_ns(req, sa)
        finish = self._time_ns + lat
        cmd = self._active_cmds[cmd_idx]
        cmd.plane_busy[plane] = req
        cmd.finish_at = max(cmd.finish_at, finish)
        sa.busy_until   = finish
        sa.current_page = req.addr.page_id
        self._plane_in_flight[plane] += 1
        self._plane_active_ns[plane] += lat

    def _issue_to_subarray(
        self, req: NANDRequest, sa_id: int, plane: int
    ) -> None:
        sa = self._subarrays[sa_id]

        # Program suspend: if write in flight and this is a read, suspend the write
        if sa.suspended and req.op == OpType.READ:
            if sa.reads_since_suspend < self._sa.max_reads_per_suspend:
                sa.reads_since_suspend += 1
                self._suspend_events += 1

        lat = self._subarray_latency_ns(req, sa)
        finish = self._time_ns + lat

        # Start a new single-plane command
        cmd = _ActiveCmd(
            op_type=req.op,
            row_offset=req.addr.row_offset,
            plane_busy={plane: req},
            finish_at=finish,
        )
        self._active_cmds.append(cmd)
        self._total_cmds += 1

        sa.busy_until   = finish
        sa.current_page = req.addr.page_id
        self._plane_in_flight[plane] += 1
        self._plane_active_ns[plane] += lat

    def _subarray_latency_ns(self, req: NANDRequest, sa: _SubarrayState) -> float:
        """
        Compute subarray latency for req, modelling the NAND page buffer.

        Page-buffer model (mirrors HBFSim):
          - First access to a block: pay tR (sense entire block into page buffer).
          - Subsequent accesses to the SAME block: pay tRC (page-buffer output only).
          - Access to a DIFFERENT block: pay tR again, displacing the current buffer.

        For multi-page requests the pages are read sequentially, crossing block
        boundaries when (page_id + i) >= pages_per_block.  Each new block costs
        one tR; pages within the already-loaded block cost tRC.
        """
        num_pages      = max(1, math.ceil(req.size_bytes / self._sa.page_size_bytes))
        pages_per_block = self._sa.pages_per_block

        if req.op in (OpType.READ, OpType.PREFETCH):
            # ── Aggregate read spanning N distinct NAND blocks ───────────────
            # Each KV block lives in its own NAND block, so each incurs a tR
            # page-buffer miss; only the pages within a block hit (tRC). This
            # makes one aggregate per-plane read produce the same latency as the
            # equivalent N per-KV-block reads of the trace-level sim.
            if req.num_blocks > 0:
                nb           = req.num_blocks
                pages_per_kv = max(1, num_pages // nb)
                misses_per_kv = max(1, math.ceil(pages_per_kv / pages_per_block))
                num_misses   = min(num_pages, nb * misses_per_kv)
                num_hits     = num_pages - num_misses
                total_ns     = num_hits * self._sa.tRC_ns + num_misses * self._sa.tR_ns
                # Distinct physical blocks: leave the page buffer cold so a
                # subsequent step does not score a spurious cache hit.
                sa.current_block = -1
                self._cache_hits   += num_hits
                self._cache_misses += num_misses
                return total_ns

            # ── Fast path: sequential read from block_id, page 0 ─────────────
            # submit_plane_reads always issues requests with block_id=0, page_id=0
            # (token_id=0 → row_offset=0).  For this fully sequential pattern the
            # hit/miss count is exact and O(1): miss at every block boundary plus
            # a possible miss on the very first page if the page buffer holds a
            # different block.  This produces the IDENTICAL result to the loop.
            if req.addr.page_id == 0:
                ppb        = pages_per_block
                base_block = req.addr.block_id
                full_blks  = num_pages // ppb
                partial    = num_pages % ppb
                is_first_hit = (sa.current_block == base_block)
                num_misses = (full_blks - int(is_first_hit)) + int(partial > 0)
                num_hits   = num_pages - num_misses
                total_ns   = num_hits * self._sa.tRC_ns + num_misses * self._sa.tR_ns
                sa.current_block = base_block + (num_pages - 1) // ppb
                self._cache_hits   += num_hits
                self._cache_misses += num_misses
                return total_ns

            # ── General path: vectorised page-by-page for non-zero page_id ──
            pages      = np.arange(num_pages, dtype=np.int64)
            blocks     = req.addr.block_id + (req.addr.page_id + pages) // pages_per_block
            prev_blks  = np.empty(num_pages, dtype=np.int64)
            prev_blks[0]  = sa.current_block
            prev_blks[1:] = blocks[:-1]
            is_miss        = blocks != prev_blks
            num_misses     = int(np.sum(is_miss))
            num_hits       = num_pages - num_misses
            total_ns       = num_hits * self._sa.tRC_ns + num_misses * self._sa.tR_ns
            sa.current_block = int(blocks[-1])
            self._cache_hits   += num_hits
            self._cache_misses += num_misses
            return total_ns

        elif req.op == OpType.WRITE:
            sa.current_block = req.addr.block_id
            return self._sa.tPROG_ns * num_pages

        return self._sa.tR_ns * num_pages

    # ------------------------------------------------------------------
    # Event collection
    # ------------------------------------------------------------------

    def _next_event_ns(self) -> Optional[float]:
        """Time of the next subarray completion."""
        events = [
            cmd.finish_at for cmd in self._active_cmds
            if cmd.finish_at > self._time_ns
        ]
        return min(events) if events else None

    def _collect_completions(self) -> List[MemCompletion]:
        """Collect all commands whose finish_at <= current time."""
        completed: List[MemCompletion] = []
        still_active: List[_ActiveCmd] = []

        for cmd in self._active_cmds:
            if cmd.finish_at <= self._time_ns:
                for plane, req in cmd.plane_busy.items():
                    sa_id = req.addr.subarray_id  # global flat index across all channels/dies/planes
                    latency_ns = cmd.finish_at - req.issued_at
                    was_hit = (
                        req.op in (OpType.READ, OpType.PREFETCH)
                        and latency_ns < self._sa.tR_ns * 0.5
                    )
                    completed.append(MemCompletion(
                        req_id=req.req_id,
                        latency_ns=latency_ns,
                        plane_id=plane,
                        die_id=req.addr.die_id,
                        was_cache_hit=was_hit,
                    ))
                    self._plane_in_flight[plane] = max(
                        0, self._plane_in_flight[plane] - 1
                    )
            else:
                still_active.append(cmd)

        self._active_cmds = still_active
        return completed
