#!/usr/bin/env python3
"""
Single-point runner using the IN-REPO HBF_LINEAR_REGRESSION predictor
(no external HBFSim overlay). Drives a Vidur decode-only simulation and prints
TPOT / throughput. Used by run_full_sweep.py.

Usage:
    python run_point_hbf.py --model meta-llama/Meta-Llama-3-8B --device a100 \
        --batch_size 8 --context_length 4096 --tensor_parallel_size 1 \
        --sparsity_fraction 1.0
"""
from __future__ import annotations

import argparse
import glob
import csv as csvmod
import math
import os
import sys
import tempfile

VIDUR_ROOT = os.path.dirname(os.path.abspath(__file__))
if VIDUR_ROOT not in sys.path:
    sys.path.insert(0, VIDUR_ROOT)

PRED = "hbf_linear_regression"
# flat-dataclass splits the class name per capital letter: HBF -> h_b_f
PFX = "--h_b_f_linear_regression_execution_time_predictor_config_"


def build_argv(args, output_dir):
    # kv-grid resolution for the prediction table. //256 samples the (memory-bound,
    # ~linear) decode-attention time at 256 kv points -> <0.5% interpolation error at
    # long context but ~8x smaller cached tables than //2048 (keeps pred_cache bounded).
    gran = max(16, args.context_length // 256)
    chunk = min(args.context_length, 4096)
    num_blocks = (args.num_kv_blocks if args.num_kv_blocks > 0 else
                  math.ceil(args.batch_size * math.ceil(args.context_length / 16) / 0.99) + 64)
    argv = [
        "run_point_hbf.py",
        "--replica_config_model_name",           args.model,
        "--replica_config_device",               args.device,
        "--replica_config_num_pipeline_stages",  "1",
        "--replica_config_tensor_parallel_size", str(args.tensor_parallel_size),
        # Explicit eight-GPU Blackwell NVLink profile. The CSV is generated from
        # the documented 1.8 TB/s/GPU link rate and an exposed latency parameter.
        "--replica_config_network_device",       args.network_device,
        # In-repo HBF predictor (LinearRegression base + NAND plane model)
        "--execution_time_predictor_config_type", PRED,
        PFX + "kv_cache_prediction_granularity", str(gran),
        PFX + "prediction_max_tokens_per_request",
        str(args.context_length + args.decode_tokens + gran + 16),
        PFX + "prediction_max_prefill_chunk_size", str(chunk),
        PFX + "hbfsim_config_path",  args.hbfsim_toml,
        PFX + "placement_policy",    "STRIPE_ACROSS_PLANES",
        PFX + "sparsity_fraction",   str(args.sparsity_fraction),
        PFX + "hbm_kv_fraction",     str(args.hbm_kv_fraction),
        # Sarathi chunked prefill scheduler
        "--replica_scheduler_config_type",           "sarathi",
        "--sarathi_scheduler_config_batch_size_cap", str(args.batch_size),
        "--sarathi_scheduler_config_chunk_size",     str(chunk),
        "--sarathi_scheduler_config_num_blocks",     str(num_blocks),
        # Requests: synthetic, fixed length, decode-only
        "--request_generator_config_type",            "synthetic",
        "--length_generator_config_type",             "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(args.context_length),
        "--fixed_request_length_generator_config_decode_tokens",  str(args.decode_tokens),
        "--interval_generator_config_type",           "poisson",
        "--poisson_request_interval_generator_config_qps", str(args.qps),
        "--synthetic_request_generator_config_num_requests", str(args.num_requests),
        "--synthetic_request_generator_config_decode_only",
        # Metrics
        "--metrics_config_output_dir",  output_dir,
        # Per-process cache dir: the sklearn predictor caches trained models to
        # disk keyed by a hash; parallel sweep workers writing a SHARED cache
        # race and truncate each other's files (empty-pickle EOFError). A unique
        # cache dir per process is collision-free.
        "--metrics_config_cache_dir",   args.cache_dir,
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--log_level", "warning",
    ]
    if args.all_reduce_input_file:
        argv += [
            PFX + "all_reduce_input_file",
            os.path.abspath(args.all_reduce_input_file),
        ]
    if args.hbf_sra:   # store_true bool flag (default False -> omit)
        argv.append(PFX + "hbf_sra")
    if args.naive_sparse:
        argv.append(PFX + "naive_sparse")
    # Ablation load multipliers (default 1.0 -> no effect on Dense/SPLASH)
    argv += [PFX + "sparse_read_amplification", str(args.sparse_read_amplification),
             PFX + "plane_imbalance_factor",    str(args.plane_imbalance_factor),
             PFX + "selection_page_tokens",     str(args.selection_page_tokens),
             PFX + "backing_bw_gbps",           str(args.backing_bw_gbps)]
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--context_length", type=int, default=4096)
    p.add_argument("--decode_tokens", type=int, default=4)
    p.add_argument("--num_requests", type=int, default=8)
    p.add_argument("--qps", type=float, default=1000.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--network_device", default="blackwell_nvl8")
    p.add_argument(
        "--all_reduce_input_file", default="",
        help="Optional generated all-reduce CSV for communication sensitivity.",
    )
    p.add_argument("--sparsity_fraction", type=float, default=1.0)
    p.add_argument("--hbm_kv_fraction", type=float, default=0.0)
    p.add_argument("--hbf_sra", action="store_true",
                   help="HBF-SRA baseline: in-flash NMP scoring of all cold K, "
                        "then GPU sparse read.")
    p.add_argument("--naive_sparse", action="store_true",
                   help="Token-granular (random-position) sparse baseline: "
                        "sequential read until the last useful block.")
    p.add_argument("--sparse_read_amplification", type=float, default=1.0,
                   help="Page-read amplification for token-granular retrieval "
                        "(ablation; 1.0 = page-granular SPLASH).")
    p.add_argument(
        "--selection_page_tokens", type=int, default=16,
        choices=(16, 32, 64, 128),
        help=(
            "KV tokens/head per physical HBF page: with separate FP16 K/V "
            "head streams at dimension 128, 16/32/64/128 correspond to "
            "4/8/16/32 KB for either stream."
        ),
    )
    p.add_argument("--plane_imbalance_factor", type=float, default=1.0,
                   help="Busiest-plane load multiplier for global (non-plane-"
                        "balanced) top-k (ablation; 1.0 = plane-balanced SPLASH).")
    p.add_argument("--backing_bw_gbps", type=float, default=0.0,
                   help="Flat backing-store bandwidth GB/s (DRAM/SSD); 0 = HBF plane model.")
    p.add_argument("--num_kv_blocks", type=int, default=0)
    p.add_argument("--hbfsim_toml", default="configs/hbf_paper.toml")
    p.add_argument("--cache_dir", default="",
                   help="Predictor cache dir (default: a unique temp dir, "
                        "collision-free for parallel sweeps).")
    args = p.parse_args()
    args.hbfsim_toml = os.path.abspath(args.hbfsim_toml)
    _own_cache = not args.cache_dir
    if _own_cache:
        args.cache_dir = tempfile.mkdtemp(prefix="hbf_cache_")

    output_dir = tempfile.mkdtemp(prefix="hbf_pt_")
    saved = sys.argv[:]
    sys.argv = build_argv(args, output_dir)
    try:
        from vidur.config import SimulationConfig
        from vidur.simulator import Simulator
        from vidur.utils.random import set_seeds
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
        sim._write_output()
    finally:
        sys.argv = saved

    req_csvs = glob.glob(f"{output_dir}/**/request_metrics.csv", recursive=True)
    if not req_csvs:
        print("RESULT status=FAIL reason=no_csv")
        return
    with open(req_csvs[0]) as f:
        rows = list(csvmod.DictReader(f))

    def stats(col, scale=1000.0):
        vals = sorted(float(r[col]) for r in rows if r.get(col, ""))
        if not vals:
            return None
        n = len(vals)
        return dict(mean=sum(vals) / n * scale, p50=vals[n // 2] * scale,
                    p95=vals[min(n - 1, int(n * 0.95))] * scale,
                    p99=vals[min(n - 1, int(n * 0.99))] * scale)

    s = stats("decode_time_execution_plus_preemption_normalized")
    if s is None:
        print("RESULT status=FAIL reason=no_decode")
        return
    print(f"RESULT status=OK tpot_p50_ms={s['p50']} tpot_p99_ms={s['p99']} "
          f"tpot_mean_ms={s['mean']} tpot_p95_ms={s['p95']} n_requests={len(rows)}")

    # tidy temp dirs so a large sweep doesn't fill /tmp
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    if _own_cache:
        shutil.rmtree(args.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
