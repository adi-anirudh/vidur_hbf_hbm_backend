# Vidur HBF/HBM Memory Backend

A plane-accurate NAND flash (HBF) and HBM memory backend for [Vidur](https://github.com/microsoft/vidur), a simulator for LLM inference. Models a High Bandwidth Flash stack stacked on the compute die — the same physical form-factor as HBM, but using NAND flash cells for ~10× higher capacity at the cost of higher read latency.

The backend replaces Vidur's analytical bandwidth/latency formulas with a cycle-accurate plane scheduler that models multi-plane command merging, page-buffer cache-read shortcuts (tRC), program suspend/resume, and configurable KV placement policies. It is validated against the HBFSim C++ simulator.

---

## Motivation

Long-context LLM inference (100K–1M tokens) requires storing KV caches that far exceed GPU HBM capacity. HBF provides a memory tier between HBM and DRAM/SSD:

| Memory | BW | Capacity | Latency |
|---|---|---|---|
| HBM | 3 TB/s | ~80 GB | ~100 ns |
| **HBF (this work)** | **~768 GB/s** | **~200 GB** | **~4 µs/page** |
| DRAM | 50 GB/s | ~TB | ~100 ns |
| NVMe SSD | 7 GB/s | ~TB | ~100 µs |

With sparsity-aware scheduling (query-dependent top-K attention), only a fraction of KV blocks need to be fetched from flash per decode step, making HBF competitive with HBM for long-context decode throughput.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HBFLinearRegressionPredictor            │
│  (hbf_execution_time_predictor.py)                       │
│  • computes per-plane KV read counts analytically        │
│  • submits aggregate reads to HBFSimBackend              │
│  • models HBM hot-window + HBF cold-window pipeline      │
└───────────────────┬─────────────────────────────────────┘
                    │ submit_plane_reads / submit_decode_step
┌───────────────────▼─────────────────────────────────────┐
│                  HBFSimBackend  (backend.py)             │
│  • maps logical KV block IDs → physical NAND addresses  │
│  • wraps PlaneScheduler + PlaneAddressMapper             │
│  • optionally emits HBFSim-compatible trace files        │
└───────┬────────────────────────────┬────────────────────┘
        │                            │
┌───────▼──────────┐   ┌────────────▼────────────────────┐
│ PlaneAddressMapper│   │      PlaneScheduler              │
│ (address_mapper) │   │      (plane_scheduler.py)        │
│                  │   │                                  │
│ Placement policy:│   │ • serialises ops per subarray    │
│  STRIPE — best   │   │ • tRC page-buffer cache hits     │
│   for batch      │   │ • multi-plane command merging    │
│   decode (max    │   │ • program suspend/resume         │
│   multi-plane    │   │ • per-plane stall / occupancy    │
│   merges)        │   │   statistics                     │
│  PACK — best for │   └──────────────────────────────────┘
│   prefetch (tRC) │
│  INTERLEAVE —    │
│   balanced       │
└──────────────────┘
```

---

## Key Files

| File | Role |
|---|---|
| `vidur/memory_backends/hbf/plane_scheduler.py` | Cycle-accurate NAND plane/subarray scheduler |
| `vidur/memory_backends/hbf/address_mapper.py` | Logical KV block → physical NAND address mapper |
| `vidur/memory_backends/hbf/backend.py` | `MemoryBackend` implementation; trace emission |
| `vidur/memory_backends/hbf/config_loader.py` | TOML config parser; `HBFSimConfig` dataclasses |
| `vidur/execution_time_predictor/hbf_execution_time_predictor.py` | Vidur predictor; HBM+HBF pipeline model |
| `configs/hbf_default.toml` | Default 768-plane SLC HBF config (768 GB/s, 206 GB) |
| `run_hbf_simulation.py` | End-to-end simulation runner with sparsity support |
| `compare_hbfsim.py` | Micro-benchmark comparison against HBFSim C++ |

---

## Hardware Model

### NAND Timing

| Parameter | Default | Meaning |
|---|---|---|
| `tR_ns` | 4096 ns | Page read — I/O transfer of `page_size` at 1 GB/s/plane |
| `tRC_ns` | 30 ns | Cache read — page already in plane's page buffer |
| `tPROG_ns` | 100 µs | Page program (write) |
| `tBERS_ns` | 1 ms | Block erase |

**Bandwidth identity:** `BW = n_planes × page_size / tR`. This is preserved across configurations — 768 planes × 4 KB / 4096 ns = 64 planes × 8 MB / 699 µs = **768 GB/s**.

### Page Buffer and tRC

Each plane has a page buffer holding the last-loaded NAND block. When a subsequent read hits the same block, it pays `tRC = 30 ns` instead of `tR`. This matters for the `PACK_BY_SEQUENCE` placement policy where consecutive layer reads within a sequence stay on the same plane and same block.

For long-context sparse decode (`STRIPE_ACROSS_PLANES`), reads from different sequences go to different physical blocks — the page buffer is always cold and tRC never fires. In this regime `tR` dominates and the two models are equivalent.

### Multi-Plane Command Merging

Reads from different planes that share the same `row_offset` (= `page_id = token_id % pages_per_block`) are merged into a single die-level command. All planes execute in parallel and complete at one `tR` cost. This is the primary bandwidth amplification mechanism:

- Batch decode step: all sequences at the same `token_id` → same `row_offset` → up to `num_planes_per_die` reads merged into one command.
- Peak throughput: `total_planes × page_size / tR` when all planes are occupied.

### Placement Policies

| Policy | CLI | Best for |
|---|---|---|
| `STRIPE_ACROSS_PLANES` | `STRIPE` | Large-batch decode — consecutive block IDs round-robin across planes, maximising multi-plane merge parallelism |
| `PACK_BY_SEQUENCE` | `PACK` | Prefetch pipelines — all KV blocks of a sequence share one plane, enabling tRC cache-read chains within a sequence |
| `INTERLEAVE_BY_TOKEN` | `INTERLEAVE` | Mixed workloads — distributes by (token_id, layer_id) |

---

## Quick Start

### Install

```bash
git clone <this repo>
cd vidur_hbf_hbm_backend
pip install -e ".[dev]"   # or: uv sync
```

### Run a simulation

```bash
python3 run_hbf_simulation.py \
    --model meta-llama/Llama-2-70b-hf \
    --context_length 131072 \
    --batch_size 16 \
    --num_requests 64 \
    --sparsity_fraction 0.1 \
    --placement_policy STRIPE
```

Key arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--context_length` | 4096 | Max KV context length per request |
| `--batch_size` | 32 | Decode batch size cap |
| `--sparsity_fraction` | 1.0 | Fraction of KV blocks fetched per step (1.0 = dense) |
| `--hbm_kv_fraction` | 0.0 | Fraction of KV cache kept hot in HBM |
| `--placement_policy` | STRIPE | STRIPE / PACK / INTERLEAVE |

### Run the HBFSim comparison

Validates the Python scheduler against the HBFSim C++ offline model on three micro-benchmarks:

```bash
PYTHONPATH=. python3 compare_hbfsim.py
```

Case A — 4-plane parallel read (multi-plane merge): both models agree at 1×tR  
Case B — same-plane same-block sequential: both models agree at tR + (N−1)×tRC  
Case C — same-plane different-blocks (realistic KV): both agree at N×tR  
Case D — bandwidth invariance: 768×4 KB, 64×8 MB, 1024×16 KB all converge at 2.097 ms

---

## Configuration

Edit `configs/hbf_default.toml` to tune hardware parameters:

```toml
[nand_media]
# Bandwidth knobs — BW = num_planes x page_size_bytes / tR_ns
num_planes_per_die  = 96    # planes per die
num_dies_per_stack  = 8     # total_planes = planes/die x dies/stack x channels
page_size_bytes     = 4096  # bytes per NAND page
tR_ns               = 4096  # read latency (I/O-gated: page_size / BW_per_plane)

# Capacity knob
blocks_per_plane    = 256   # pages per plane = blocks x pages_per_block

[hbm_media]
bandwidth_GBps  = 1024.0    # HBM bandwidth (for hot-window pipeline)
```

To model HBFSim's `hbf_research_scaled` profile (tR=1 µs, 16 KB pages, 1024 planes):

```toml
[nand_media]
num_planes_per_die  = 4
num_dies_per_stack  = 16
num_subarrays_per_die = 4
page_size_bytes     = 16384
tR_ns               = 1000
tRC_ns              = 30
pages_per_block     = 512

[logic_die]
num_channels      = 16
num_dies_per_stack = 16
```

---

## Relation to Original Vidur

This repo is a fork of [Vidur](https://github.com/microsoft/vidur). All original Vidur schedulers, request generators, and metrics are preserved. The HBF/HBM backend adds:

- `vidur/memory_backends/` — pluggable memory backend interface + HBF and Ramulator backends
- `vidur/execution_time_predictor/hbf_execution_time_predictor.py` — HBF-aware decode time predictor
- Modified `vidur/config/config.py`, `vidur/types/` — registration of new predictor type

The backend is designed to be swappable: the `MemoryBackend` ABC in `vidur/memory_backends/base.py` can be implemented for any memory technology.
