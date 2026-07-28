#!/usr/bin/env python3
"""Write exact, technology-independent SPLASH hardware-storage counts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from evaluation_common import PLATFORM

SCORE_BYTES = 2
PAGE_ID_BYTES = 4
TOPK_ENTRY_BYTES = 8  # padded {FP16 score, 32-bit page id}
MAX_BATCH = 128
MAX_CONTEXT = 2_097_152

bridge_per_stack = (
    2 * PLATFORM.hbf_planes_per_stack * PLATFORM.hbf_page_bytes
)
logical_pages_per_request = math.ceil(
    MAX_CONTEXT / PLATFORM.tokens_per_selection_page
)
logical_pages_per_stack = math.ceil(
    logical_pages_per_request / PLATFORM.hbf_stacks_per_gpu
)
logical_pages_per_plane = math.ceil(
    logical_pages_per_stack / PLATFORM.hbf_planes_per_stack
)
quota_per_plane = math.ceil(
    logical_pages_per_plane * PLATFORM.sparse_kv_fraction
)
topk_state_per_stack = (
    PLATFORM.hbf_planes_per_stack
    * quota_per_plane
    * TOPK_ENTRY_BYTES
    * MAX_BATCH
)
page_map_bytes_per_stack = (
    PLATFORM.hbf_gb_per_gpu * 1e9
    / PLATFORM.hbf_stacks_per_gpu
    / PLATFORM.hbf_page_bytes
    * 4
)

payload = {
    "schema_version": 1,
    "fixed_architectural_dimensions": {
        "stacks_per_gpu": PLATFORM.hbf_stacks_per_gpu,
        "planes_per_stack": PLATFORM.hbf_planes_per_stack,
        "physical_page_bytes": PLATFORM.hbf_page_bytes,
        "bridge_windows_per_stack": 2,
        "bridge_sram_bytes_per_stack": bridge_per_stack,
        "bridge_sram_mib_per_stack": bridge_per_stack / 2**20,
        "bridge_sram_mib_per_gpu": (
            bridge_per_stack * PLATFORM.hbf_stacks_per_gpu / 2**20
        ),
        "nma_fp16_macs_per_stack": 512,
        "nma_input_bytes_per_cycle_per_stack": 1024,
        "nma_clock_ghz": 1.0,
        "centroid_metadata_fraction_of_k": (
            1.0 / PLATFORM.tokens_per_selection_page
        ),
        "centroid_metadata_fraction_of_kv": (
            1.0 / (2 * PLATFORM.tokens_per_selection_page)
        ),
    },
    "worst_evaluated_topk_state": {
        "context_tokens": MAX_CONTEXT,
        "batch": MAX_BATCH,
        "sparse_fraction": PLATFORM.sparse_kv_fraction,
        "logical_pages_per_request": logical_pages_per_request,
        "logical_pages_per_stack_per_request": logical_pages_per_stack,
        "logical_pages_per_plane_per_request": logical_pages_per_plane,
        "quota_entries_per_plane_per_request": quota_per_plane,
        "bytes_per_entry_assumption": TOPK_ENTRY_BYTES,
        "state_mib_per_stack": topk_state_per_stack / 2**20,
    },
    "ftl_comparison": {
        "conventional_four_byte_page_map_mib_per_stack": (
            page_map_bytes_per_stack / 2**20
        ),
        "splash_mapping": (
            "request extent descriptor plus deterministic stripe; "
            "bad-block exceptions are not yet sized"
        ),
    },
    "author_supplied_placeholders": {
        "area_mm2": "fill in paper",
        "dynamic_power_w": "fill in paper",
        "leakage_power_w": "fill in paper",
        "energy_per_operation": "fill in paper",
    },
}

out = Path("results/hardware_overhead_manifest.json")
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
