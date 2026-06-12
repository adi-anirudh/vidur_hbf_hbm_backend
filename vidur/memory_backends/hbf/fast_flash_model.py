"""
Fast analytical flash-latency model — provably equivalent to the cycle-accurate
PlaneScheduler for the all-reads-available-at-t0 (decode-step drain) case.

Model
-----
Every read in a decode step is submitted at t=0 and the scheduler is drained.
Each subarray processes its assigned reads serially, in submission order, and
subarrays run concurrently. Multi-plane command merging only groups reads for
die-level command accounting; it never delays a subarray past its own serial
work, and it never extends the last completion beyond the busiest subarray's
serial finish. Therefore:

    wall_clock = max over subarrays of  sum(per-read page-buffer latency,
                                            in submission order)

The per-read latency is computed by the scheduler's OWN _subarray_latency_ns,
applied with a per-subarray _SubarrayState carried in submission order, so the
page-buffer (tR/tRC) accounting is identical to the cycle-accurate model by
construction. The single modelling claim — the max-over-subarrays reduction —
is checked exhaustively in verify_flash_equivalence.py.

This is O(num_reads) with no global event queue or merge search.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from vidur.memory_backends.hbf.address_mapper import PlaneAddressMapper
from vidur.memory_backends.hbf.config_loader import HBFSimConfig
from vidur.memory_backends.hbf.plane_scheduler import (
    NANDRequest, PlaneScheduler, _SubarrayState,
)
from vidur.memory_backends.base import OpType


# A read is (op, size_bytes, block_id, layer_id, sequence_id, token_id, num_blocks)
Read = Tuple


def fast_decode_latency_ns(
    reads: List[dict],
    cfg: HBFSimConfig,
    mapper: PlaneAddressMapper,
) -> float:
    """
    Return the decode-step wall-clock (ns) for `reads`, matching a fresh
    PlaneScheduler.drain() that received the same reads in the same order.

    reads: list of dicts with keys op, size_bytes, block_id, layer_id,
           sequence_id, token_id, and optional num_blocks.
    """
    # Reuse the scheduler's exact per-read latency function and state type.
    sched = PlaneScheduler(die_cfg=cfg.nand_die, sa_cfg=cfg.subarray,
                           stack_cfg=cfg.nand_stack)
    sa_state: Dict[int, _SubarrayState] = {}
    sa_busy:  Dict[int, float] = {}

    for r in reads:
        addr = mapper.map_block(
            block_id=max(0, r["block_id"]),
            layer_id=r.get("layer_id", 0),
            sequence_id=r.get("sequence_id", 0),
            token_id=r.get("token_id", 0),
        )
        nand = NANDRequest(
            req_id=0, op=r["op"], addr=addr,
            size_bytes=r["size_bytes"], issued_at=0.0,
            num_blocks=r.get("num_blocks", 0),
        )
        sa_id = addr.subarray_id
        sa = sa_state.get(sa_id)
        if sa is None:
            sa = _SubarrayState(sa_id=sa_id)
            sa_state[sa_id] = sa
        lat = sched._subarray_latency_ns(nand, sa)
        sa_busy[sa_id] = sa_busy.get(sa_id, 0.0) + lat

    return max(sa_busy.values()) if sa_busy else 0.0
