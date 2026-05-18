"""
HBM bank-level row-buffer scheduler.

Models FRFCFS (First-Ready First-Come First-Serve) row buffer policy using
timing parameters from HBM3E.yaml / [hbm_media] in the HBFSim TOML config.

Three access latencies depending on row buffer state:
  Row hit:      tCL                        (~14 ns @ 1 GHz for HBM3E)
  Row miss:     tRCD + tCL                 (closed bank, no conflict)
  Row conflict: tRP  + tRCD + tCL          (different row already open)

Channels are independent; banks within a channel share a row buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from vidur.memory_backends.base import MemCompletion, OpType


@dataclass
class HBMRequest:
    req_id:    int
    op:        OpType
    addr:      int        # physical byte address
    size_bytes: int
    issued_at: float      # ns


@dataclass
class _BankState:
    open_row:   int   = -1   # -1 = no row open (closed-page policy default)
    busy_until: float = 0.0  # ns


class HBMBankScheduler:
    """
    Stateful HBM bank scheduler.

    Timing source: HBFSim [hbm_media] section (tCL_cycles, tRCD_cycles, tRP_cycles)
    or HBM3E.yaml (tCL, tRCDRD, tRP — converted from ns to cycles internally).

    Address mapping (matching HBFSim's RoBaRaCoCh scheme):
      bits [5:0]    — column offset (cache line = 64 B)
      bits [5+col_bits-1:6] — column
      bits [...]    — row
      bits [...]    — bank
      bits [...]    — channel
    """

    def __init__(
        self,
        num_channels:          int   = 16,
        num_banks_per_channel: int   = 8,    # HBM3E: 16 channels × 8 banks = 128 banks
        tCL_cycles:            int   = 14,
        tRCD_cycles:           int   = 14,
        tRP_cycles:            int   = 14,
        freq_mhz:              float = 1000.0,
        row_size_bytes:        int   = 2048,  # HBM typical row
    ):
        self._num_ch  = num_channels
        self._num_bk  = num_banks_per_channel
        self._cycle_ns = 1000.0 / freq_mhz
        self._tCL  = tCL_cycles  * self._cycle_ns
        self._tRCD = tRCD_cycles * self._cycle_ns
        self._tRP  = tRP_cycles  * self._cycle_ns
        self._row_bytes = row_size_bytes

        total_banks = num_channels * num_banks_per_channel
        self._banks: List[_BankState] = [_BankState() for _ in range(total_banks)]

        self._pending:   deque[HBMRequest] = deque()
        self._in_flight: List[tuple] = []   # (finish_ns, MemCompletion)

        self._time_ns: float = 0.0

        # Stats
        self._row_hits:      int = 0
        self._row_misses:    int = 0
        self._row_conflicts: int = 0
        self._total_reqs:    int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self, req: HBMRequest) -> None:
        self._pending.append(req)

    def has_pending(self) -> bool:
        return bool(self._pending) or bool(self._in_flight)

    def advance(self, delta_ns: float) -> List[MemCompletion]:
        end_ns = self._time_ns + delta_ns
        completed: List[MemCompletion] = []

        while True:
            self._dispatch_pending()
            next_ev = self._next_event_ns()
            if next_ev is None or next_ev > end_ns:
                self._time_ns = end_ns
                break
            self._time_ns = next_ev
            completed.extend(self._collect_completions())

        return completed

    def drain(self) -> List[MemCompletion]:
        completed: List[MemCompletion] = []
        for _ in range(10_000_000):
            self._dispatch_pending()
            next_ev = self._next_event_ns()
            if next_ev is None:
                break
            self._time_ns = next_ev
            completed.extend(self._collect_completions())
        return completed

    # Stats accessors
    def row_hits(self)      -> int: return self._row_hits
    def row_misses(self)    -> int: return self._row_misses
    def row_conflicts(self) -> int: return self._row_conflicts
    def total_reqs(self)    -> int: return self._total_reqs
    def current_time_ns(self) -> float: return self._time_ns

    # ------------------------------------------------------------------
    # Address → bank/row mapping (RoBaRaCoCh-style)
    # ------------------------------------------------------------------

    def _channel_of(self, addr: int) -> int:
        # Channel interleaved at cache-line granularity (64 B)
        return (addr // 64) % self._num_ch

    def _bank_of(self, addr: int) -> int:
        # Bank interleaved above channel
        return ((addr // 64) // self._num_ch) % self._num_bk

    def _row_of(self, addr: int) -> int:
        banks_per_row = self._row_bytes // 64
        return (addr // 64) // (self._num_ch * self._num_bk * banks_per_row)

    def _bank_idx(self, addr: int) -> int:
        ch = self._channel_of(addr)
        bk = self._bank_of(addr)
        return ch * self._num_bk + bk

    # ------------------------------------------------------------------
    # Dispatch / timing
    # ------------------------------------------------------------------

    def _dispatch_pending(self) -> None:
        still: deque[HBMRequest] = deque()
        while self._pending:
            req = self._pending.popleft()
            bank_idx = self._bank_idx(req.addr)
            bank = self._banks[bank_idx]

            if bank.busy_until > self._time_ns:
                still.append(req)
                continue

            row = self._row_of(req.addr)
            lat = self._access_latency_ns(bank, row)
            finish = self._time_ns + lat

            bank.open_row   = row
            bank.busy_until = finish
            self._total_reqs += 1

            self._in_flight.append((finish, MemCompletion(
                req_id=req.req_id,
                latency_ns=finish - req.issued_at,
            )))
        self._pending = still

    def _access_latency_ns(self, bank: _BankState, row: int) -> float:
        if bank.open_row == row:
            self._row_hits += 1
            return self._tCL
        elif bank.open_row == -1:
            self._row_misses += 1
            return self._tRCD + self._tCL
        else:
            self._row_conflicts += 1
            return self._tRP + self._tRCD + self._tCL

    def _next_event_ns(self) -> Optional[float]:
        if not self._in_flight:
            return None
        return min(finish for finish, _ in self._in_flight)

    def _collect_completions(self) -> List[MemCompletion]:
        done, still = [], []
        for finish, comp in self._in_flight:
            if finish <= self._time_ns:
                done.append(comp)
            else:
                still.append((finish, comp))
        self._in_flight = still
        return done
