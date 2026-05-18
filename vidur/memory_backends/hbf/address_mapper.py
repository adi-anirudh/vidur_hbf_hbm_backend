"""
KV block -> physical NAND address mapper.

Host-visible address hierarchy:
    plane -> block -> page

    plane_id  in [0, total_planes)      — bandwidth unit; each plane has its own page buffer
    block_id  in [0, blocks_per_plane)  — capacity unit within a plane
    page_id   in [0, pages_per_block)   — smallest addressable unit

    total_planes = num_channels x num_dies_per_channel x num_planes_per_die

Peak bandwidth (all planes reading in parallel via multi-plane commands):
    BW = total_planes x page_size_bytes / tR_ns   [bytes/ns = GB/s]

Capacity per plane:
    blocks_per_plane x pages_per_block x page_size_bytes

Subarrays are an internal microarchitectural concept inside a plane (used
by the PlaneScheduler for program-suspend and concurrency modelling).
They are NOT part of the host address space.

Placement policies
------------------
STRIPE_ACROSS_PLANES   block_id % total_planes
    Consecutive block IDs round-robin across all planes.  In a batch decode
    where blocks are allocated sequentially, each sequence lands on a
    different plane -> maximum multi-plane parallelism.

PACK_BY_SEQUENCE       sequence_id % total_planes
    All KV blocks of sequence S land on the same plane.  Sequential layer
    reads within a decode step benefit from the tRC cache-read path
    (same block, same page stays in the plane's page buffer).

INTERLEAVE_BY_TOKEN    (token_id x num_layers + layer_id) % total_planes
    Distributes by (token, layer), balancing locality and cross-request
    parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vidur.memory_backends.hbf.config_loader import (
    NANDDieConfig, NANDStackConfig, SubarrayConfig,
)


class PlacementPolicy(str, Enum):
    STRIPE_ACROSS_PLANES  = "stripe"      # block_id % total_planes
    PACK_BY_SEQUENCE      = "pack_seq"    # sequence_id % total_planes
    INTERLEAVE_BY_TOKEN   = "interleave"  # (token_id × num_layers + layer_id) % total_planes


@dataclass
class PhysicalAddr:
    """
    Physical address in the host-visible NAND hierarchy: plane -> block -> page.

    plane_id  — GLOBAL flat index in [0, total_planes); bandwidth unit
    block_id  — block within the plane, in [0, blocks_per_plane)
    page_id   — page within the block; doubles as row_offset for multi-plane merging

    subarray_id — INTERNAL flat index used by PlaneScheduler only;
                  derived as plane_id * subarrays_per_plane + subarray_in_plane
    """
    channel_id:  int
    stack_id:    int
    die_id:      int   # die within channel
    plane_id:    int   # global plane index (flat: channel * dies_per_ch * planes/die + ...)
    block_id:    int   # block within the plane [0, blocks_per_plane)
    page_id:     int   # page within block; = row_offset for multi-plane merge
    subarray_id: int   # internal scheduler index (not part of host address)

    @property
    def row_offset(self) -> int:
        """
        Row offset used for multi-plane command merging.
        Two requests with the same row_offset on different planes can merge into
        one command that reads all planes in parallel at one tR cost.
        """
        return self.page_id


class PlaneAddressMapper:
    """
    Maps logical KV block IDs to physical NAND addresses.

    total_planes = num_channels × num_dies_per_channel × num_planes_per_die

    Each plane's page buffer holds exactly one (block, page) pair.
    A cache hit (tRC) requires the same (block_id, page_id) to be re-read
    on the same plane.
    """

    def __init__(
        self,
        sa_cfg:      SubarrayConfig,
        die_cfg:     NANDDieConfig,
        stack_cfg:   NANDStackConfig,
        policy:      PlacementPolicy = PlacementPolicy.STRIPE_ACROSS_PLANES,
        num_layers:  int = 32,
    ):
        self._sa         = sa_cfg
        self._die        = die_cfg
        self._stack      = stack_cfg
        self._policy     = policy
        self._num_layers = num_layers

        # ── Derived geometry ──────────────────────────────────────────────
        self._planes_per_die      = die_cfg.num_planes
        self._subarrays_per_plane = die_cfg.subarrays_per_plane
        # blocks_per_subarray is internal; derived from plane-level config
        self._blocks_per_subarray = die_cfg.blocks_per_subarray

        # Total parallel planes exposed to the host
        self._total_planes = (
            stack_cfg.num_channels
            * stack_cfg.num_dies_per_channel
            * die_cfg.num_planes
        )

        # Total subarrays in the system (internal flat index space)
        self._total_subarrays = self._total_planes * self._subarrays_per_plane

        # Capacity per plane (pages): plane -> block -> page
        self._pages_per_plane = (
            die_cfg.blocks_per_plane
            * sa_cfg.pages_per_block
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def total_planes(self) -> int:
        return self._total_planes

    @property
    def total_subarrays(self) -> int:
        return self._total_subarrays

    @property
    def subarrays_per_plane(self) -> int:
        return self._subarrays_per_plane

    @property
    def policy(self) -> PlacementPolicy:
        return self._policy

    @policy.setter
    def policy(self, p: PlacementPolicy) -> None:
        self._policy = p

    # ── Address mapping ───────────────────────────────────────────────────

    def map_block(
        self,
        block_id:    int,
        layer_id:    int = 0,
        sequence_id: int = 0,
        token_id:    int = 0,
    ) -> PhysicalAddr:
        """
        Map a logical KV block to a physical NAND address.

        plane_id and subarray_id in the returned PhysicalAddr are GLOBAL flat
        indices understood by PlaneScheduler.
        """
        total_planes = self._total_planes

        if self._policy == PlacementPolicy.STRIPE_ACROSS_PLANES:
            # Round-robin across all planes; batch-decode sequences land on
            # different planes → maximises multi-plane merge parallelism.
            plane_id    = block_id % total_planes
            local_block = block_id // total_planes

        elif self._policy == PlacementPolicy.PACK_BY_SEQUENCE:
            # All KV blocks of sequence S on the same plane; sequential layer
            # reads benefit from the tRC cache-read path.
            plane_id    = sequence_id % total_planes
            local_block = block_id * self._num_layers + layer_id

        elif self._policy == PlacementPolicy.INTERLEAVE_BY_TOKEN:
            # Distribute by (token, layer) — balances locality and parallelism.
            plane_id    = (token_id * self._num_layers + layer_id) % total_planes
            local_block = block_id

        else:
            raise ValueError(f"Unknown placement policy: {self._policy}")

        # ── Host-visible block within plane ──────────────────────────────
        block_in_plane = local_block % self._die.blocks_per_plane

        # ── Internal: map block_in_plane -> subarray (scheduler index) ───
        spp = self._subarrays_per_plane
        subarray_in_plane  = block_in_plane % spp

        # Global subarray index (internal scheduler index)
        global_subarray_id = plane_id * spp + subarray_in_plane

        # ── Physical decomposition of plane_id ────────────────────────────
        planes_per_ch  = self._planes_per_die * self._stack.num_dies_per_channel
        channel_id     = plane_id // planes_per_ch
        die_in_channel = (plane_id % planes_per_ch) // self._planes_per_die

        # ── Page offset = row_offset for multi-plane merge ─────────────────
        # All planes at the same token_id share row_offset -> multi-plane merge.
        row_offset = token_id % self._sa.pages_per_block

        return PhysicalAddr(
            channel_id=channel_id,
            stack_id=0,
            die_id=die_in_channel,
            plane_id=plane_id,
            block_id=block_in_plane,
            page_id=row_offset,
            subarray_id=global_subarray_id,
        )

    # ── Capacity helpers ──────────────────────────────────────────────────

    def capacity_per_plane_pages(self) -> int:
        return self._pages_per_plane

    def capacity_per_plane_bytes(self) -> int:
        return self._pages_per_plane * self._sa.page_size_bytes

    def total_capacity_bytes(self) -> int:
        return self._total_planes * self.capacity_per_plane_bytes()

    def blocks_in_plane(self) -> int:
        """Blocks addressable within a plane."""
        return self._die.blocks_per_plane
