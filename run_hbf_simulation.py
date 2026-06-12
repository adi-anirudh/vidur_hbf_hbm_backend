#!/usr/bin/env python3.10
"""
HBF simulation runner using cycle-accurate HBFSim overlay.

Runs a Vidur simulation with the HbfsimOverlayPredictor wired in.
The overlay submits KV-cache memory accesses to the cycle-accurate HBFSim
backend and adds the resulting flash stall to each decode step's execution time.

Usage:
    python3.10 run_hbf_simulation.py \\
        --model meta-llama/Meta-Llama-3-8B --device a100 \\
        --batch_size 32 --context_length 131072 \\
        --decode_only --qps 1000 --num_requests 64 \\
        --kv_read_fraction 0.1 --hbm_kv_fraction 0.05 \\
        --hbfsim_toml configs/hbf_paper.toml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

VIDUR_ROOT  = os.path.dirname(os.path.abspath(__file__))
HBFSIM_ROOT = "/home/adityaan/HBFSim"

for p in (VIDUR_ROOT, HBFSIM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Build Vidur SimulationConfig argv
# ---------------------------------------------------------------------------

def _build_vidur_argv(args: argparse.Namespace, output_dir: str) -> list:
    return [
        "run_hbf_simulation.py",
        "--replica_config_model_name",           args.model,
        "--replica_config_device",               args.device,
        "--replica_config_num_pipeline_stages",  "1",
        "--replica_config_tensor_parallel_size", str(args.tensor_parallel_size),
        # Use random forest (accurate compute timing), overlay adds flash stall.
        # prediction_max_tokens_per_request must cover context_length so the
        # prediction table has entries for all KV cache sizes we encounter.
        # kv_cache_prediction_granularity is raised for long contexts to keep
        # the table size manageable (≤ ~2048 KV-size entries).
        "--execution_time_predictor_config_type", "random_forrest",
        "--random_forrest_execution_time_predictor_config_num_training_job_threads", "1",
        "--random_forrest_execution_time_predictor_config_kv_cache_prediction_granularity",
        str(max(16, args.context_length // 2048)),
        # Max tokens must be >= ceil((ctx + decode_tokens) / granularity) * granularity
        # so that every decode-step lookup hits a populated table entry.
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(args.context_length + args.decode_tokens + max(16, args.context_length // 2048) + 16),
        # For TP>1, Sarathi initialises KV across workers with a prefill pass
        # even in decode-only mode, so the table must cover the chunk size.
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(min(args.context_length, 4096)),
        # Sarathi chunked prefill
        "--replica_scheduler_config_type",         "sarathi",
        "--sarathi_scheduler_config_batch_size_cap", str(args.batch_size),
        "--sarathi_scheduler_config_chunk_size",   str(min(args.context_length, 4096)),
        "--sarathi_scheduler_config_num_blocks",
        str(args.num_kv_blocks if args.num_kv_blocks > 0
            else math.ceil(args.batch_size * math.ceil(args.context_length / 16) / 0.99) + 64),
        # Requests
        "--request_generator_config_type",          "synthetic",
        "--length_generator_config_type",            "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(args.context_length),
        "--fixed_request_length_generator_config_decode_tokens",  str(args.decode_tokens),
        "--interval_generator_config_type",          "poisson",
        "--poisson_request_interval_generator_config_qps", str(args.qps),
        "--synthetic_request_generator_config_num_requests", str(args.num_requests),
        "--synthetic_request_generator_config_decode_only"
        if args.decode_only else
        "--no-synthetic_request_generator_config_decode_only",
        # Metrics
        "--metrics_config_output_dir",              output_dir,
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--log_level", "warning",
    ]


# ---------------------------------------------------------------------------
# Build and attach HBFSim overlay
# ---------------------------------------------------------------------------

def _build_overlay(sim, args):
    from integration.vidur import (
        AddressLayout, HbfsimAddrSpace, KvPoolConfig, ModelDims,
        build_default_overlay,
    )

    replica = list(sim._cluster.replicas.values())[0]
    cluster_cfg   = sim._config.cluster_config
    scheduler_cfg = cluster_cfg.replica_scheduler_config

    dims = ModelDims(
        num_layers=replica.num_layers,
        num_q_heads=replica.num_q_heads,
        num_kv_heads=replica.num_kv_heads,
        head_dim=replica.attention_head_dim,
        embedding_dim=replica.embedding_dim,
        mlp_hidden_dim=replica.mlp_hidden_dim,
        vocab_size=replica.vocab_size,
        use_gated_mlp=replica.use_gated_mlp,
        tensor_parallel_size=replica.num_tensor_parallel_workers,
    )
    num_blocks = scheduler_cfg.num_blocks if scheduler_cfg.num_blocks else (
        math.ceil(args.batch_size * math.ceil(args.context_length / 16) / 0.99) + 64
    )
    kv = KvPoolConfig(block_size=scheduler_cfg.block_size, num_blocks=num_blocks)

    # HBF address space: large enough for any (model, ctx, batch) combination.
    GiB = 1 << 30
    TiB = 1 << 40
    addr_space = HbfsimAddrSpace(
        hbm_base=0x0,          hbm_size=256 * GiB,
        hbf_base=0x4000000000, hbf_size=16 * TiB,
    )
    hbm_kv_blocks = int(num_blocks * args.hbm_kv_fraction)
    layout = AddressLayout(
        dims, kv, addr_space,
        tier_split_hbm_blocks=hbm_kv_blocks,
        weights_on_hbf_fraction=0.0,
    )

    # Auto-compute max_drain_cycles if not specified (3× expected flash time).
    max_dc = args.max_drain_cycles
    if max_dc <= 0:
        per_tok_kv = 2 * dims.num_kv_heads * dims.head_dim * 2
        blk_bytes = kv.block_size * per_tok_kv * dims.num_layers
        n_blocks_est = math.ceil(args.batch_size * math.ceil(args.context_length / 16))
        flash_bytes = n_blocks_est * blk_bytes * args.kv_read_fraction
        flash_cycles = int(flash_bytes / 768e9 * 1e9)  # at 1 GHz
        max_dc = max(1_000_000_000, flash_cycles * 3)

    overlay = build_default_overlay(
        hbfsim_config_path=args.hbfsim_toml,
        dims=dims,
        kv_pool=kv,
        layout=layout,
        base_predictor=None,
        num_pipeline_stages=replica.num_pipeline_stages,
        overlap_with_compute=True,
        emit_attn_weights=False,
        emit_mlp_weights=False,
        lazy_allocator=True,
        kv_read_fraction=args.kv_read_fraction,
        max_drain_cycles=max_dc,
    )
    return overlay, dims, kv


def _attach_overlay(sim, overlay):
    """Swap base predictor into overlay and install in every stage scheduler.

    Also hooks replica_scheduler.free() so overlay's LazyBlockAllocator
    mirrors Vidur's request lifecycle and returns blocks to the free pool.
    """
    for rs in sim._scheduler._replica_schedulers.values():
        orig_free = rs.free
        def _hooked_free(*request_ids, _orig=orig_free, _ov=overlay):
            _orig(*request_ids)
            for rid in request_ids:
                _ov.on_request_completed(rid)
        rs.free = _hooked_free

        for stage_sched in rs._replica_stage_schedulers.values():
            base = stage_sched._execution_time_predictor
            overlay._base = base
            stage_sched._execution_time_predictor = overlay


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace):
    import glob, csv as csvmod

    output_dir = tempfile.mkdtemp(prefix="hbf_sim_")
    saved_argv = sys.argv[:]
    sys.argv = _build_vidur_argv(args, output_dir)

    try:
        from vidur.config import SimulationConfig
        from vidur.simulator import Simulator
        from vidur.utils.random import set_seeds

        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        sim = Simulator(config)

        overlay, dims, kv = _build_overlay(sim, args)
        _attach_overlay(sim, overlay)

        sim.run()
        sim._write_output()
    finally:
        sys.argv = saved_argv

    # Print metrics
    req_csvs = glob.glob(f"{output_dir}/**/request_metrics.csv", recursive=True)
    if not req_csvs:
        print("ERROR: no request_metrics.csv", file=sys.stderr)
        return

    with open(req_csvs[0]) as f:
        rows = list(csvmod.DictReader(f))

    def pct(col, scale=1000):
        vals = sorted(float(r[col]) for r in rows if r.get(col, ""))
        if not vals:
            return None, None
        n = len(vals)
        return vals[n // 2] * scale, vals[min(n-1, int(n*0.99))] * scale

    ttft_p50, ttft_p99 = pct("prefill_e2e_time")
    tpot_p50, tpot_p99 = pct("decode_time_execution_plus_preemption_normalized")

    tput = None
    try:
        by_id = sorted(rows, key=lambda r: int(r["Request Id"]))
        t = 0.0; arrivals = []
        for r in by_id:
            t += float(r.get("request_inter_arrival_delay") or 0)
            arrivals.append(t)
        comps = [a + float(r["request_e2e_time"]) for a, r in zip(arrivals, by_id)]
        dur = max(comps) - min(arrivals)
        dtoks = sum(float(r.get("request_num_decode_tokens", 0)) for r in rows)
        tput = dtoks / dur if dur > 0 else None
    except Exception:
        pass

    stall_ms = overlay.total_stall_seconds * 1e3

    print(f"\n--- Request Metrics ---")
    for lbl, v in [("TTFT p50 (ms)", ttft_p50), ("TTFT p99 (ms)", ttft_p99),
                   ("TPOT p50 (ms)", tpot_p50), ("TPOT p99 (ms)", tpot_p99),
                   ("Throughput (tok/s)", tput),
                   ("HBF stall total (ms)", stall_ms)]:
        print(f"  {lbl:30s} {f'{v:.2f}' if v is not None else 'N/A'}")
    print(f"\n  Output: {output_dir}")
    print(f"  Requests completed: {len(rows)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="HBF end-to-end simulation (HBFSim overlay)")
    p.add_argument("--model",            default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--device",           default="a100")
    p.add_argument("--num_requests",     type=int,   default=64)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--context_length",   type=int,   default=4096)
    p.add_argument("--decode_tokens",    type=int,   default=128)
    p.add_argument("--qps",              type=float, default=1000.0)
    p.add_argument("--decode_only",      action="store_true", default=False)
    p.add_argument("--kv_read_fraction", type=float, default=1.0,
                   help="Fraction of KV blocks to read per (request, layer). "
                        "1.0=Dense, 0.1=Sparse TopK. NaiveSparse=1.0 (same as Dense).")
    p.add_argument("--hbm_kv_fraction",  type=float, default=0.0,
                   help="Fraction of KV blocks in HBM hot window. "
                        "0 = all-flash. Computed by run_paper_sweep.py.")
    p.add_argument("--num_kv_blocks",    type=int,   default=0,
                   help="Override KV cache block count (0 = auto).")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Tensor parallel degree (1 = single GPU).")
    p.add_argument("--max_drain_cycles", type=int, default=0,
                   help="HBFSim drain budget in cycles (0 = auto-compute from flash time).")
    p.add_argument("--hbfsim_toml",      default="configs/hbf_paper.toml",
                   help="Path to HBFSim TOML config")
    args = p.parse_args()

    # Resolve to absolute path NOW before Vidur can chdir during init.
    toml_path = os.path.abspath(args.hbfsim_toml)
    if not os.path.exists(toml_path):
        sys.exit(f"ERROR: HBFSim TOML not found: {toml_path}")
    args.hbfsim_toml = toml_path

    _run(args)


if __name__ == "__main__":
    main()
