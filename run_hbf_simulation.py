# -*- coding: utf-8 -*-
"""
End-to-end HBF simulation runner with sparsity support.

Runs a Vidur simulation using the HBF plane scheduler for cold KV reads
and reports both request-level metrics and memory backend plane statistics.

For long-context inference, use --context_length to set the KV context size
and --sparsity_fraction to enable query-dependent sparsity (e.g. 0.1 = 10%).

Usage:
    python3.9 run_hbf_simulation.py [--context_length C] [--decode_tokens D]
                                    [--batch_size B] [--num_requests N]
                                    [--sparsity_fraction S]
                                    [--placement_policy STRIPE|PACK|INTERLEAVE]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Ensure vidur package is importable from the repo root
sys.path.insert(0, os.path.dirname(__file__))


def _build_argv(args: argparse.Namespace, output_dir: str, toml_path: str) -> list:
    """Convert our simple args into the flat SimulationConfig CLI format."""
    policy_map = {
        "STRIPE":     "STRIPE_ACROSS_PLANES",
        "PACK":       "PACK_BY_SEQUENCE",
        "INTERLEAVE": "INTERLEAVE_BY_TOKEN",
    }
    policy = policy_map.get(args.placement_policy.upper(), args.placement_policy)

    return [
        "run_hbf_simulation.py",
        # Model + device
        "--replica_config_model_name",            args.model,
        "--replica_config_device",                args.device,
        "--replica_config_num_pipeline_stages",   "1",
        "--replica_config_tensor_parallel_size",  "1",
        # Predictor
        "--execution_time_predictor_config_type", "hbf_linear_regression",
        "--h_b_f_linear_regression_execution_time_predictor_config_hbfsim_config_path",
        toml_path,
        "--h_b_f_linear_regression_execution_time_predictor_config_placement_policy",
        policy,
        "--h_b_f_linear_regression_execution_time_predictor_config_sparsity_fraction",
        str(args.sparsity_fraction),
        "--h_b_f_linear_regression_execution_time_predictor_config_hbm_kv_fraction",
        str(args.hbm_kv_fraction),
        # Extend sklearn lookup table to cover the full context length
        "--h_b_f_linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(max(4096, args.context_length)),
        # Scheduler (Sarathi chunked prefill)
        "--replica_scheduler_config_type",        "sarathi",
        "--sarathi_scheduler_config_batch_size_cap", str(args.batch_size),
        "--sarathi_scheduler_config_chunk_size",  str(min(args.context_length, 4096)),
        "--sarathi_scheduler_config_num_blocks",
        str(args.num_kv_blocks if args.num_kv_blocks > 0
            else args.batch_size * max(1, -(-args.context_length // 16)) + 1024),
        # Requests: synthetic, fixed length, Poisson arrivals
        "--request_generator_config_type",        "synthetic",
        "--length_generator_config_type",         "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(args.context_length),
        "--fixed_request_length_generator_config_decode_tokens",  str(args.decode_tokens),
        "--interval_generator_config_type",       "poisson",
        "--poisson_request_interval_generator_config_qps", str(args.qps),
        "--synthetic_request_generator_config_num_requests", str(args.num_requests),
        "--synthetic_request_generator_config_decode_only"
        if args.decode_only else
        "--no-synthetic_request_generator_config_decode_only",
        # Metrics
        "--metrics_config_output_dir",            output_dir,
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--log_level",                            "warning",
    ]


def _run_simulation(args: argparse.Namespace, toml_path: str, sparsity: float):
    """Run one simulation and return (simulator, output_dir)."""
    import copy
    a = copy.copy(args)
    a.sparsity_fraction = sparsity

    output_dir = tempfile.mkdtemp(prefix="hbf_sim_")
    sys.argv = _build_argv(a, output_dir, toml_path)

    from vidur.config import SimulationConfig
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()
    simulator._write_output()
    return simulator, output_dir


def _print_request_metrics(output_dir: str):
    try:
        import glob, csv as csvmod
        req_csvs = glob.glob(f"{output_dir}/**/request_metrics.csv", recursive=True)
        if not req_csvs:
            print("  No request_metrics.csv found.")
            return
        with open(req_csvs[0]) as f:
            rows = list(csvmod.DictReader(f))

        def _pct(label, col, scale=1000, unit="ms"):
            vals = sorted(float(r[col]) for r in rows if r.get(col, ""))
            if not vals:
                return
            n = len(vals)
            p50 = vals[n // 2]
            p90 = vals[min(n - 1, int(n * 0.90))]
            p99 = vals[min(n - 1, int(n * 0.99))]
            print(f"  {label:42s}  "
                  f"p50={p50*scale:8.1f}{unit}  "
                  f"p90={p90*scale:8.1f}{unit}  "
                  f"p99={p99*scale:8.1f}{unit}")

        cols = [
            ("E2E latency",                   "request_e2e_time"),
            ("Prefill TTFT",                  "prefill_e2e_time"),
            ("Decode time / token",           "decode_time_execution_plus_preemption_normalized"),
            ("Scheduling delay",              "request_scheduling_delay"),
            ("Prefill time (exec+preemption)","prefill_time_execution_plus_preemption"),
        ]
        for lbl, col in cols:
            try:
                _pct(lbl, col)
            except Exception:
                pass

        # Throughput: reconstruct arrival times from cumulative inter-arrival delays
        try:
            by_id = sorted(rows, key=lambda r: int(r["Request Id"]))
            t = 0.0
            arrivals = []
            for r in by_id:
                t += float(r.get("request_inter_arrival_delay") or 0)
                arrivals.append(t)
            completions = [arr + float(r["request_e2e_time"])
                           for arr, r in zip(arrivals, by_id)]
            sim_duration = max(completions) - min(arrivals) if arrivals else 0
            total_decode_tok = sum(float(r.get("request_num_decode_tokens", 0)) for r in rows)
            total_all_tok    = sum(float(r.get("request_num_tokens", 0)) for r in rows)
            if sim_duration > 0:
                print(f"  {'Throughput':42s}  "
                      f"{len(rows)/sim_duration:.3f} req/s  |  "
                      f"{total_decode_tok/sim_duration:.1f} decode tok/s  |  "
                      f"{total_all_tok/sim_duration:.1f} total tok/s")
        except Exception:
            pass
        print(f"  {'Requests completed':42s}  {len(rows)}")
    except Exception as e:
        print(f"  (request metrics not available: {e})")


def _get_plane_stats(simulator):
    """Extract HBF plane stats from the simulator."""
    try:
        from vidur.execution_time_predictor.hbf_execution_time_predictor import (
            HBFLinearRegressionExecutionTimePredictor,
        )
        for rs in simulator._scheduler._replica_schedulers.values():
            stage_sched = rs._replica_stage_schedulers.get(0)
            if stage_sched and isinstance(
                stage_sched._execution_time_predictor,
                HBFLinearRegressionExecutionTimePredictor,
            ):
                return stage_sched._execution_time_predictor._hbf.plane_stats()
    except Exception:
        pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="HBF end-to-end simulation with sparsity")
    p.add_argument("--model",            default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--device",           default="a100")
    p.add_argument("--num_requests",     type=int,   default=32)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--context_length",   type=int,   default=4096,
                   help="Prefill (KV context) length in tokens")
    p.add_argument("--decode_tokens",    type=int,   default=128)
    p.add_argument("--qps",              type=float, default=1.0)
    p.add_argument("--sparsity_fraction", type=float, default=0.1,
                   help="Fraction of KV blocks to read per (seq, layer). "
                        "1.0=dense, 0.1=10%% sparsity. Use 0 to run dense+sparse comparison.")
    p.add_argument("--hbm_kv_fraction", type=float, default=0.0,
                   help="Fraction of each sequence's most-recent KV blocks in HBM hot window. "
                        "0 = all-flash baseline. 1/11 ≈ 0.0909 for 1:10 HBM:HBF split. "
                        "When > 0: pipeline model where HBM dense attention and HBF sparse "
                        "reads run concurrently; stall = max(hbm_time, flash_time).")
    p.add_argument("--placement_policy", default="STRIPE",
                   choices=["STRIPE", "PACK", "INTERLEAVE"])
    p.add_argument("--decode_only", action="store_true",
                   help="Start all requests with prefill already complete. "
                        "Bypasses the prefill queue so the decode batch fills immediately. "
                        "Use this to test steady-state decode throughput at large batch sizes.")
    p.add_argument("--num_kv_blocks", type=int, default=0,
                   help="Override KV cache block count (default: auto = batch × blocks_per_seq). "
                        "Set >0 to bypass the GPU-memory-based limit. Required for "
                        "decode_only runs with batch > ~300 on Llama-7B/A100.")
    p.add_argument("--toml", default="configs/hbf_default.toml",
                   help="Path to HBFSim TOML config")
    p.add_argument("--compare", action="store_true",
                   help="Run dense (1.0) and sparse (--sparsity_fraction) back-to-back for comparison")
    args = p.parse_args()

    toml_path = os.path.abspath(args.toml)
    if not os.path.exists(toml_path):
        print(f"ERROR: TOML config not found: {toml_path}")
        sys.exit(1)

    # Show hardware config
    from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader
    cfg = HBFSimConfigLoader(toml_path).load()
    total_planes = (cfg.nand_stack.num_channels
                    * cfg.nand_stack.num_dies_per_channel
                    * cfg.nand_die.num_planes)
    tR_ns    = cfg.subarray.tR_ns
    peak_bw  = total_planes * cfg.subarray.page_size_bytes / tR_ns
    cap_gb   = (total_planes * cfg.nand_die.blocks_per_plane
                * cfg.subarray.pages_per_block * cfg.subarray.page_size_bytes) / 1e9
    page_kb  = cfg.subarray.page_size_bytes / 1024

    print(f"\n=== HBF Long-Context Sparsity Simulation ===")
    print(f"Model:          {args.model}")
    print(f"Context length: {args.context_length} tokens  (decode: +{args.decode_tokens})")
    print(f"Batch cap:      {args.batch_size}  |  QPS: {args.qps}  |  Requests: {args.num_requests}")
    print(f"Policy:         {args.placement_policy}")
    if args.hbm_kv_fraction > 0:
        hbf_frac = 1.0 - args.hbm_kv_fraction
        ratio = round(hbf_frac / args.hbm_kv_fraction)
        print(f"KV split:       HBM {args.hbm_kv_fraction:.1%} (hot) + HBF {hbf_frac:.1%} (cold)  "
              f"[1:{ratio} HBM:HBF]  →  pipeline model")
        print(f"HBF sparsity:   {args.sparsity_fraction:.0%} of cold KV blocks read (staged addresses)")
    else:
        print(f"Sparsity:       {args.sparsity_fraction:.0%} of KV blocks read per (seq, layer)  "
              f"[all-flash baseline]")
    print()
    print(f"HBF hardware:")
    print(f"  planes = {total_planes}  ({cfg.nand_stack.num_channels}ch x "
          f"{cfg.nand_stack.num_dies_per_channel}dies x {cfg.nand_die.num_planes}pl/die)")
    print(f"  page   = {page_kb:.0f} KB  |  tR = {tR_ns:.0f} ns  |  peak BW = {peak_bw:.0f} GB/s")
    print(f"  cap    = {cap_gb:.1f} GB")

    # Analytical KV data volume estimate
    try:
        from vidur.config import SimulationConfig  # noqa: F401
        # quick block-size lookup: default sarathi block_size=16
        block_size = 16
        Nkv, D = 32, 128  # Llama-7B defaults (close enough for estimate)
        kv_block_bytes = 2 * block_size * Nkv * D * 2
        pages_per_kv_block = max(1, kv_block_bytes // cfg.subarray.page_size_bytes)
        num_kv_blocks = max(1, -(-args.context_length // block_size))  # ceiling div
        active_blocks = max(1, -(-int(num_kv_blocks * args.sparsity_fraction) // 1))
        num_layers = 32  # Llama-7B

        dense_data_gb = (args.batch_size * num_layers * num_kv_blocks
                         * kv_block_bytes) / 1e9
        sparse_data_gb = (args.batch_size * num_layers * active_blocks
                          * kv_block_bytes) / 1e9

        # Blocks per plane (STRIPE)
        total_blocks_dense  = args.batch_size * num_layers * num_kv_blocks
        total_blocks_sparse = args.batch_size * num_layers * active_blocks
        bpp_dense  = -(-total_blocks_dense  // total_planes)  # ceiling
        bpp_sparse = -(-total_blocks_sparse // total_planes)

        dense_latency_ms  = bpp_dense  * pages_per_kv_block * tR_ns / 1e6
        sparse_latency_ms = bpp_sparse * pages_per_kv_block * tR_ns / 1e6

        hbm_bw_gbps = cfg.hbm.bandwidth_gbps  # GBps = bytes/ns

        print()
        print(f"Analytical estimate (batch={args.batch_size}, L={num_layers}, block_size={block_size}):")
        print(f"  KV blocks per (seq, layer): {num_kv_blocks} dense  →  {active_blocks} sparse "
              f"({args.sparsity_fraction:.0%})")
        print(f"  Data per decode step:       {dense_data_gb*1000:.0f} MB dense  "
              f"→  {sparse_data_gb*1000:.0f} MB sparse")
        print(f"  Blocks per plane:           {bpp_dense} dense  →  {bpp_sparse} sparse")
        print(f"  Flash stall (per step):     {dense_latency_ms:.1f} ms dense  "
              f"→  {sparse_latency_ms:.1f} ms sparse  "
              f"({dense_latency_ms/sparse_latency_ms:.1f}x speedup)")

        if args.hbm_kv_fraction > 0:
            # Pipeline model breakdown
            hbm_kv_frac    = args.hbm_kv_fraction
            hbm_blocks_seq = max(1, round(num_kv_blocks * hbm_kv_frac))
            hbf_blocks_seq = max(0, num_kv_blocks - hbm_blocks_seq)

            # HBM: read all seqs × all layers' hot window sequentially
            hbm_kv_total  = args.batch_size * num_layers * hbm_blocks_seq * kv_block_bytes
            hbm_time_ms   = hbm_kv_total / (hbm_bw_gbps * 1e6)

            # HBF sparse: cold window only
            total_hbf     = args.batch_size * num_layers * hbf_blocks_seq
            bpp_hbf_dense = -(-total_hbf // total_planes)
            bpp_hbf_sparse = max(1, -(-int(bpp_hbf_dense * args.sparsity_fraction) // 1))
            flash_time_ms  = bpp_hbf_sparse * pages_per_kv_block * tR_ns / 1e6

            effective_ms  = max(hbm_time_ms, flash_time_ms)
            bound         = "HBM-bound ✓ flash free" if flash_time_ms <= hbm_time_ms else "flash-bound"

            # Breakeven: largest sparsity where flash ≤ hbm_time
            bpp_hbf_full  = bpp_hbf_dense * pages_per_kv_block  # pages/plane at 100%
            breakeven     = hbm_time_ms / (bpp_hbf_full * tR_ns / 1e6) if bpp_hbf_full > 0 else 1.0

            print()
            print(f"  Pipeline model (HBM:HBF = 1:{round((1-hbm_kv_frac)/hbm_kv_frac)}):")
            print(f"    HBM hot window:  {hbm_blocks_seq*block_size:6d} tokens  "
                  f"({hbm_blocks_seq} blocks/seq)  "
                  f"→  HBM read: {hbm_time_ms:.1f} ms  (all {num_layers}L × {args.batch_size}B @ {hbm_bw_gbps:.0f} GB/s)")
            print(f"    HBF cold window: {hbf_blocks_seq*block_size:6d} tokens  "
                  f"({hbf_blocks_seq} blocks/seq)  "
                  f"→  flash:    {flash_time_ms:.1f} ms  ({args.sparsity_fraction:.0%} sparse, {bpp_hbf_sparse} pp)")
            print(f"    Effective stall: {effective_ms:.1f} ms  [{bound}]")
            print(f"    Breakeven sparsity (flash ≤ HBM): ≤ {breakeven:.1%}")
            print(f"    vs dense all-flash:  {dense_latency_ms:.1f} ms  →  {effective_ms:.1f} ms  "
                  f"({dense_latency_ms/effective_ms:.1f}× speedup)")
    except Exception:
        pass

    if args.compare:
        # Run dense and sparse scenarios as separate subprocesses to avoid
        # global config state leaking between SimulationConfig instantiations.
        import subprocess
        base_cmd = [
            sys.executable, __file__,
            "--model",          args.model,
            "--device",         args.device,
            "--num_requests",   str(args.num_requests),
            "--batch_size",     str(args.batch_size),
            "--context_length", str(args.context_length),
            "--decode_tokens",  str(args.decode_tokens),
            "--qps",            str(args.qps),
            "--placement_policy", args.placement_policy,
            "--toml",           args.toml,
        ]
        dense_cmd  = base_cmd + ["--sparsity_fraction", "1.0"]
        sparse_cmd = base_cmd + ["--sparsity_fraction", str(args.sparsity_fraction)]

        print(f"\n{'='*60}")
        print("Running: Dense (100%)")
        print(f"{'='*60}")
        subprocess.run(dense_cmd, check=False)

        print(f"\n{'='*60}")
        print(f"Running: Sparse ({args.sparsity_fraction:.0%})")
        print(f"{'='*60}")
        subprocess.run(sparse_cmd, check=False)
        return

    # Single-scenario run
    print(f"\n{'='*60}")
    print(f"Running: Sparsity {args.sparsity_fraction:.0%}")
    print(f"{'='*60}")
    try:
        simulator, output_dir = _run_simulation(args, toml_path, args.sparsity_fraction)
        print(f"\n--- Request Metrics ---")
        _print_request_metrics(output_dir)

        ps = _get_plane_stats(simulator)
        if ps:
            # Total data transferred = pages read × page_size
            total_data_gb = ps.cache_misses * cfg.subarray.page_size_bytes / 1e9
            # Sustained BW = data / flash-active time (all planes busy → ≈ peak)
            sustained_bw  = (total_data_gb * 1e9 / ps.current_time_ns
                             if ps.current_time_ns > 0 else 0.0)
            # Decode steps ≈ total plane ops / total_planes
            decode_steps  = (ps.total_cmds_issued + ps.multi_plane_merge_count) // total_planes
            print(f"\n--- HBF Plane Stats ---")
            print(f"  planes active fraction = {ps.active_plane_fraction:.3f}  "
                  f"(Gini={ps.gini_coefficient:.3f})")
            print(f"  multi-plane merges     = {ps.multi_plane_merge_count:,}")
            print(f"  mean cmd batch size    = {ps.mean_batch_size:.0f} planes")
            print(f"  flash sim time         = {ps.current_time_ns/1e6:,.1f} ms")
            print(f"  decode steps (est.)    = {decode_steps:,}")
            print(f"  total data transferred = {total_data_gb:,.1f} GB")
            print(f"  sustained flash BW     = {sustained_bw:.0f} GB/s  "
                  f"(peak={peak_bw:.0f} GB/s)")
        print(f"\n  Output: {output_dir}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
