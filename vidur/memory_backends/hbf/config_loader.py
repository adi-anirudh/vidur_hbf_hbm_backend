"""
Reads an HBFSim TOML config file and returns strongly-typed Python config objects
that drive the plane scheduler and address mapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config dataclasses (mirror HBFSim C++ structs)
# ---------------------------------------------------------------------------

@dataclass
class SubarrayConfig:
    """
      page_size = 4 KB, tR = 4096 ns (I/O-gated: 4 KB at 1 GB/s/plane).
    """
    page_size_bytes:        int   = 4_096        # 4 KB / page (fixed by NAND process)
    pages_per_block:        int   = 256          # 256 pages / block
    tR_ns:                  float = 4_096.0      # page read: I/O transfer of page_size at 1 GB/s/plane
    tRC_ns:                 float = 30.0         # cache read (internal sense shortcut; dominated by I/O)
    tPROG_ns:               float = 100_000.0    # page program: ~100 us SLC
    tBERS_ns:               float = 1_000_000.0  # block erase: ~1 ms SLC
    queue_depth:            int   = 8
    enable_program_suspend: bool  = True
    suspend_ns:             float = 50.0         # program-suspend overhead
    resume_ns:              float = 50.0
    max_reads_per_suspend:  int   = 1


@dataclass
class NANDDieConfig:
    """
    Die-level parameters.

    Host-visible address space per plane:
        blocks_per_plane x pages_per_block x page_size_bytes

    Peak bandwidth when all planes read in parallel (multi-plane command):
        total_planes x page_size_bytes / tR_ns   [bytes/ns = GB/s]

    Subarrays partition planes internally (scheduler uses them for
    concurrency and program-suspend modelling); the host does not
    address subarrays directly.  blocks_per_plane must be divisible
    by (num_subarrays // num_planes).

    Defaults match the HBF SLC die: 96 planes, 256 blocks/plane.
    With 8 dies/stack @ 1 GHz: BW = 768 planes x 4 KB / 4096 ns = 768 GB/s.
    """
    num_planes:               int   = 96     # planes per die (bandwidth knob)
    blocks_per_plane:         int   = 256    # capacity knob: plane x block x page x page_size
    num_subarrays:            int   = 96     # internal: 1 subarray per plane by default
    max_concurrent_per_plane: int   = 0      # 0 = subarrays_per_plane (full concurrency)
    enable_multi_plane_cmds:  bool  = True
    max_plane_batch:          int   = 96     # all planes in one die can merge
    max_commands_per_die:     int   = 0      # 0 = unlimited

    @property
    def subarrays_per_plane(self) -> int:
        return max(1, self.num_subarrays // self.num_planes)

    @property
    def blocks_per_subarray(self) -> int:
        """Internal value derived from the plane-level capacity config."""
        spp = self.subarrays_per_plane
        return max(1, self.blocks_per_plane // spp)


@dataclass
class NANDStackConfig:
    num_channels:         int = 1   # independent I/O channels
    num_dies_per_channel: int = 8   # 8 flash dies / stack (HBF spec)
    num_stacks:           int = 1   # physical HBF stacks

    @property
    def total_dies(self) -> int:
        return self.num_stacks * self.num_channels * self.num_dies_per_channel


@dataclass
class HBMConfig:
    num_banks:       int   = 128
    bandwidth_gbps:  float = 1024.0
    tCL_cycles:      int   = 14
    tRCD_cycles:     int   = 14
    tRP_cycles:      int   = 14
    freq_mhz:        float = 1000.0


@dataclass
class HBFSimConfig:
    subarray:    SubarrayConfig  = field(default_factory=SubarrayConfig)
    nand_die:    NANDDieConfig   = field(default_factory=NANDDieConfig)
    nand_stack:  NANDStackConfig = field(default_factory=NANDStackConfig)
    hbm:         HBMConfig       = field(default_factory=HBMConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_size(value: str | int | float) -> int:
    """Parse a size string like '32 GB', '512 KB', or a raw int."""
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip().replace("_", "")
    m = re.match(r"^([\d.]+)\s*([KMGT]?B?)$", value, re.IGNORECASE)
    if not m:
        return int(value)
    num, unit = float(m.group(1)), m.group(2).upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(num * multipliers.get(unit, 1))


def _int(d: dict, *keys: str, default: int = 0) -> int:
    for k in keys:
        if k in d:
            return int(str(d[k]).replace("_", ""))
    return default


def _float(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d:
            return float(str(d[k]).replace("_", ""))
    return default


def _bool(d: dict, key: str, default: bool = True) -> bool:
    if key in d:
        v = d[key]
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")
    return default


class HBFSimConfigLoader:
    """
    Minimal TOML parser for HBFSim config files (no external dependencies).
    Supports the subset of TOML used by HBFSim: sections, integers, floats,
    booleans, quoted strings, and size strings.
    """

    def __init__(self, config_path: str | Path):
        self._path = Path(config_path)

    def load(self) -> HBFSimConfig:
        sections = self._parse_toml(self._path)
        return self._build_config(sections)

    # ------------------------------------------------------------------
    # TOML parsing
    # ------------------------------------------------------------------

    def _parse_toml(self, path: Path) -> dict:
        """Parse TOML into a nested dict of {section: {key: value}}."""
        sections: dict = {"_root": {}}
        current = sections["_root"]
        current_key = "_root"

        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#")[0].strip()
            if not line:
                continue

            # Section header [foo] or [foo.bar]
            if line.startswith("[") and line.endswith("]"):
                current_key = line[1:-1].strip()
                if current_key not in sections:
                    sections[current_key] = {}
                current = sections[current_key]
                continue

            # Key = value
            if "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                current[k] = self._parse_value(v)

        return sections

    def _parse_value(self, v: str):
        v = v.strip()
        # Quoted string
        if (v.startswith('"') and v.endswith('"')) or \
           (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        # Boolean
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        # Strip underscores (TOML numeric separators)
        v_clean = v.replace("_", "")
        # Integer
        try:
            return int(v_clean, 0)
        except ValueError:
            pass
        # Float
        try:
            return float(v_clean)
        except ValueError:
            pass
        return v  # leave as string (e.g. size strings handled by caller)

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------

    def _build_config(self, s: dict) -> HBFSimConfig:
        nand = s.get("nand_media", {})
        hbm  = s.get("hbm_media", {})
        ld   = s.get("logic_die", {})
        am   = s.get("address_mapping", {})

        subarray = SubarrayConfig(
            page_size_bytes=_int(nand, "page_size_bytes", default=4_096),
            pages_per_block=_int(nand, "pages_per_block", default=256),
            tR_ns=_float(nand, "tR_ns", default=4_096.0),
            tRC_ns=_float(nand, "tRC_ns", default=30.0),
            tPROG_ns=_float(nand, "tPROG_ns", default=100_000.0),
            tBERS_ns=_float(nand, "tBERS_ns", default=1_000_000.0),
            queue_depth=_int(nand, "queue_depth", default=8),
        )

        num_planes = _int(nand, "num_planes_per_die", default=96)
        nand_die = NANDDieConfig(
            num_planes=num_planes,
            blocks_per_plane=_int(nand, "blocks_per_plane", default=256),
            num_subarrays=_int(
                nand, "num_subarrays_per_die",
                default=_int(ld, "num_subarrays_per_die", default=num_planes),
            ),
            max_concurrent_per_plane=_int(nand, "max_concurrent_per_plane", default=0),
            enable_multi_plane_cmds=_bool(nand, "enable_multi_plane_cmds", default=True),
            max_plane_batch=_int(nand, "max_plane_batch", default=num_planes),
        )

        # num_channels: logic_die section takes priority, then address_mapping, default=1
        num_channels = _int(ld, "num_channels",
                            default=_int(am, "num_channels", default=1))
        # dies per channel: nand_media.num_dies_per_stack is per-channel in HBFSim convention
        num_dies_per_channel = _int(nand, "num_dies_per_stack",
                                    default=_int(ld, "num_dies_per_stack", default=8))
        nand_stack = NANDStackConfig(
            num_channels=num_channels,
            num_dies_per_channel=num_dies_per_channel,
            num_stacks=_int(am, "num_stacks", default=1),
        )

        hbm_freq = _float(hbm, "freq_mhz", default=1000.0)
        hbm_cfg = HBMConfig(
            num_banks=_int(hbm, "num_banks", default=128),
            bandwidth_gbps=_float(hbm, "bandwidth_GBps", default=1024.0),
            tCL_cycles=_int(hbm, "tCL_cycles", default=14),
            tRCD_cycles=_int(hbm, "tRCD_cycles", default=14),
            tRP_cycles=_int(hbm, "tRP_cycles", default=14),
            freq_mhz=hbm_freq,
        )

        return HBFSimConfig(
            subarray=subarray,
            nand_die=nand_die,
            nand_stack=nand_stack,
            hbm=hbm_cfg,
        )
