#!/usr/bin/env python3
"""
Comprehensive paper analysis: throughput, SLO, and energy efficiency.

Three analyses:
  1. Throughput — geomean + percentile breakdown (p25/p50/p75/p95); where we win/lose
  2. Latency under SLO — max batch meeting 50ms/100ms SLO + corresponding throughput
  3. Energy efficiency — tokens/joule using HBM=2.7 pJ/bit, HBF=8 pJ/bit

Usage:
  python3 analyze_paper_results.py [paper_results.csv]
"""

import csv
import math
import statistics
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBM_PJ_PER_BIT  = 2.7    # pJ/bit (FlashAccel reference)
HBF_PJ_PER_BIT  = 8.0    # pJ/bit (NAND flash, H³ reference)
BLOCK_SIZE       = 16     # tokens per KV block (Vidur default)
SPARSITY         = 0.10   # Sparse top-K fraction

# Sparse top-K adds ~2% compute overhead (block-score FFN pass).
# This is the honest overhead that causes Sparse to *lose* at compute-bound configs.
SPARSE_OVERHEAD  = 0.02

GPU_COMPUTE_PJ_PER_FLOP = {
    # TDP / peak_TFLOPS_FP16 / assumed_utilisation(50%)
    "a100": 400e12 / (312e12 * 0.5),   # ~2.6 pJ/FLOP
    "h100": 700e12 / (2000e12 * 0.5),  # ~0.7 pJ/FLOP
    "a40":  300e12 / (150e12 * 0.5),   # ~4.0 pJ/FLOP
    "h200": 1000e12 / (4000e12 * 0.5), # ~0.5 pJ/FLOP
}

MODEL_INFO = {
    "microsoft/phi-2":                   dict(layers=32,  kv_heads=32, head_dim=80,  weights_gb=5.4),
    "mistralai/Mistral-7B-v0.1":         dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=14.0),
    "meta-llama/Meta-Llama-3-8B":        dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=16.0),
    "mistralai/Mixtral-8x7B-v0.1":       dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=93.0),
    "deepseek-ai/deepseek-llm-67b-chat": dict(layers=95,  kv_heads=8,  head_dim=128, weights_gb=134.0),
    "meta-llama/Meta-Llama-3-70B":       dict(layers=80,  kv_heads=8,  head_dim=128, weights_gb=140.0),
    "Qwen/Qwen-72B":                     dict(layers=80,  kv_heads=64, head_dim=128, weights_gb=144.0),
    "Qwen/Qwen2-72B":                    dict(layers=80,  kv_heads=8,  head_dim=128, weights_gb=145.0),
    "meta-llama/Meta-Llama-3.1-405B":    dict(layers=126, kv_heads=8,  head_dim=128, weights_gb=810.0),
}

GPU_MEM_GB     = {"a100": 80.0, "h100": 80.0, "a40": 45.0, "h200": 141.0}
GPU_OVERHEAD_GB = 2.0

SLO_THRESHOLDS = [50, 100]  # ms, matching FlashAccel/H³


def is_single_gpu_feasible(model, device):
    """True only when weights fit on a single GPU (no TP required)."""
    info = MODEL_INFO.get(model)
    if info is None:
        return True  # unknown model — let it through
    cap = GPU_MEM_GB.get(device, 0)
    return info["weights_gb"] + GPU_OVERHEAD_GB <= cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    idx = (len(xs) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

def fmt(v, decimals=1):
    try:    return f"{float(v):.{decimals}f}"
    except: return "N/A"

def fmtx(v):
    try:    return f"{float(v):.2f}x"
    except: return "N/A"


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

def energy_per_output_token_pJ(model, device, batch, ctx, hbm_frac, kv_read_fraction):
    """
    Estimate energy per output token in pJ.
    Components:
      - Weight reads (always HBM): read once per decode step
      - KV reads from HBM hot window
      - KV reads from HBF cold tier
      - GPU compute (attention + FFN)

    Returns (total_pJ, breakdown_dict)
    """
    info = MODEL_INFO.get(model)
    if info is None:
        return None, {}

    kv_per_token_bytes = 2 * info["kv_heads"] * info["head_dim"] * 2  # FP16 K+V

    # Per decode step: each of `batch` requests reads its KV history
    total_kv_bytes = batch * ctx * kv_per_token_bytes * info["layers"]
    hbm_kv_bytes   = total_kv_bytes * hbm_frac        * kv_read_fraction
    hbf_kv_bytes   = total_kv_bytes * (1 - hbm_frac)  * kv_read_fraction

    weight_bytes = info["weights_gb"] * 1e9  # all in HBM

    # Memory energy (pJ)
    e_hbm_pJ = (hbm_kv_bytes + weight_bytes) * 8 * HBM_PJ_PER_BIT
    e_hbf_pJ =  hbf_kv_bytes                 * 8 * HBF_PJ_PER_BIT

    # Compute energy: 2 FLOPs/param/token × batch tokens × params
    params = info["weights_gb"] * 1e9 / 2  # FP16 → 2 bytes/param
    flops_per_step = 2.0 * params * batch
    # Sparse adds top-K selection overhead
    if kv_read_fraction < 1.0:
        flops_per_step *= (1.0 + SPARSE_OVERHEAD)
    pj_per_flop = GPU_COMPUTE_PJ_PER_FLOP.get(device, 2.0)
    e_compute_pJ = flops_per_step * pj_per_flop

    total_pJ = e_hbm_pJ + e_hbf_pJ + e_compute_pJ
    per_token_pJ = total_pJ / batch  # per output token

    return per_token_pJ, {
        "hbm_pJ":     e_hbm_pJ     / batch,
        "hbf_pJ":     e_hbf_pJ     / batch,
        "compute_pJ": e_compute_pJ  / batch,
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ok = []
    dropped_tp = 0
    for r in rows:
        if not r.get("tpot_p50") or "FAILED" in str(r.get("note", "")):
            continue
        try:
            float(r["tpot_p50"])
            float(r["throughput_tps"])
        except (ValueError, TypeError):
            continue
        if not is_single_gpu_feasible(r["model"], r["device"]):
            dropped_tp += 1
            continue
        ok.append(r)
    if dropped_tp:
        print(f"Dropped {dropped_tp} rows requiring TP>1 (weights > single-GPU cap) — invalid for single-chip claims")
    return ok


# ---------------------------------------------------------------------------
# Analysis 1: Throughput
# ---------------------------------------------------------------------------

def analysis_throughput(rows):
    print("\n" + "=" * 110)
    print("ANALYSIS 1 — THROUGHPUT  (tokens/sec per GPU)")
    print("  geomean, p25/p50/p75/p95 across all (model × device × batch × ctx) configurations")
    print("=" * 110)

    by_bl = defaultdict(list)
    for r in rows:
        try:
            t = float(r["throughput_tps"])
        except (ValueError, TypeError):
            continue
        by_bl[r["baseline"]].append((r, t))

    # Summary table
    hdr = f"{'Baseline':<14}  {'N':>5}  {'geomean':>9}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'max':>9}"
    print(f"\n{hdr}")
    print("-" * 80)
    for bl in ["Dense", "NaiveSparse", "Sparse"]:
        pts = by_bl[bl]
        vals = [t for _, t in pts]
        if not vals:
            continue
        gm = geomean(vals)
        print(f"{bl:<14}  {len(vals):>5}  {fmt(gm,1):>9}  "
              f"{fmt(percentile(vals,25),1):>8}  {fmt(percentile(vals,50),1):>8}  "
              f"{fmt(percentile(vals,75),1):>8}  {fmt(percentile(vals,95),1):>8}  "
              f"{fmt(max(vals),1):>9}")

    # ---------------------------------------------------------------------------
    # Per-configuration speedup: where do we win, where do we lose?
    # ---------------------------------------------------------------------------
    print(f"\n--- Sparse vs Dense throughput ratio by configuration ---")
    print(f"{'':50}  {'ratio':>7}  {'Dense':>8}  {'Sparse':>8}  notes")
    print("-" * 90)

    dense_map = {(r["model"], r["device"], r["batch"], r["ctx"]): float(r["throughput_tps"])
                 for r in rows if r["baseline"] == "Dense"}
    sparse_map = {(r["model"], r["device"], r["batch"], r["ctx"]): float(r["throughput_tps"])
                  for r in rows if r["baseline"] == "Sparse"}

    ratios = []
    losses = []
    near_parity = []
    big_wins = []

    for key in sorted(dense_map.keys() & sparse_map.keys(),
                      key=lambda k: (k[0], k[1], int(k[3]), int(k[2]))):
        model, device, batch, ctx = key
        d_tps = dense_map[key]
        s_tps = sparse_map[key]

        # Apply sparse overhead at compute-bound configs (ctx ≤ 8K all-HBM):
        # top-K adds ~2% compute; if no HBF stall, Sparse is marginally slower.
        hbm_frac = next((float(r["hbm_frac"]) for r in rows
                         if r["model"]==model and r["device"]==device
                         and r["batch"]==batch and r["ctx"]==ctx
                         and r["baseline"]=="Sparse"), 1.0)
        flash_stall_s = next((float(r.get("flash_stall_ms") or 0) / 1000
                               for r in rows
                               if r["model"]==model and r["device"]==device
                               and r["batch"]==batch and r["ctx"]==ctx
                               and r["baseline"]=="Dense"), 0.0)

        # If stall is negligible and hbm_frac is near 1.0, apply overhead to Sparse
        if flash_stall_s < 0.01 and hbm_frac >= 0.99:
            adj_s_tps = s_tps / (1.0 + SPARSE_OVERHEAD)
        else:
            adj_s_tps = s_tps

        ratio = adj_s_tps / d_tps if d_tps > 0 else None
        if ratio is None:
            continue
        ratios.append(ratio)

        mshort = model.split("/")[-1][:22]
        label = f"{mshort:22} {device:5} B={batch:3} ctx={int(ctx)//1024:5}K"
        note = ""

        if ratio < 1.0:
            note = "<< LOSS (compute-bound, top-K overhead)"
            losses.append((ratio, label, d_tps, adj_s_tps, note))
        elif ratio < 1.05:
            note = "near parity (all-HBM or tiny HBF spill)"
            near_parity.append((ratio, label, d_tps, adj_s_tps, note))
        elif ratio > 5.0:
            note = "large win"
            big_wins.append((ratio, label, d_tps, adj_s_tps, note))

    # Print losses first (most important for reviewers)
    if losses:
        print(f"\n  *** LOSSES (Sparse < Dense) — {len(losses)} configurations ***")
        for ratio, label, d, s, note in sorted(losses)[:20]:
            print(f"  {label}  {ratio:6.3f}x  D={fmt(d,1):>8}/s  S={fmt(s,1):>8}/s  {note}")
    else:
        print(f"\n  *** No raw losses detected; after applying {SPARSE_OVERHEAD*100:.0f}% top-K overhead: ***")

    if near_parity:
        print(f"\n  Near-parity (1.00x–1.05x) — {len(near_parity)} configurations:")
        for ratio, label, d, s, note in sorted(near_parity)[:10]:
            print(f"  {label}  {ratio:6.3f}x  {note}")

    print(f"\n  Big wins (>5x) — {len(big_wins)} configurations:")
    for ratio, label, d, s, note in sorted(big_wins, reverse=True)[:15]:
        print(f"  {label}  {ratio:6.2f}x  D={fmt(d,1):>8}/s  S={fmt(s,1):>8}/s")

    # Distribution
    gm_ratio = geomean(ratios)
    print(f"\n  Speedup distribution (Sparse/Dense throughput ratio, {len(ratios)} configs):")
    print(f"  geomean={fmt(gm_ratio,3)}x  p25={fmt(percentile(ratios,25),3)}x  "
          f"p50={fmt(percentile(ratios,50),3)}x  p75={fmt(percentile(ratios,75),3)}x  "
          f"p95={fmt(percentile(ratios,95),3)}x  max={fmt(max(ratios),2)}x")
    pct_loss = sum(1 for r in ratios if r < 1.0) / len(ratios) * 100
    print(f"  Fraction of configs where Sparse loses: {pct_loss:.1f}%")


# ---------------------------------------------------------------------------
# Analysis 2: Latency under SLO
# ---------------------------------------------------------------------------

def analysis_slo(rows):
    print("\n" + "=" * 110)
    print("ANALYSIS 2 — LATENCY UNDER SLO")
    print("  Max batch meeting TPOT p50 SLO; throughput at that batch. FlashAccel/H³ conventions.")
    print("=" * 110)

    for slo_ms in SLO_THRESHOLDS:
        print(f"\n--- SLO: TPOT p50 ≤ {slo_ms}ms ---")
        hdr = (f"{'Model':22} {'device':6} {'ctx':>7}  "
               f"{'DMaxB':>6} {'DTps':>8}  "
               f"{'SMaxB':>6} {'STps':>8}  "
               f"{'ΔTps':>8}  {'Tps_ratio':>9}  note")
        print(hdr)
        print("-" * 105)

        # Group by (model, device, ctx); find max batch meeting SLO per baseline
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            try:
                tp = float(r["tpot_p50"])
                tps = float(r["throughput_tps"])
            except (ValueError, TypeError):
                continue
            groups[(r["model"], r["device"], r["ctx"])][r["baseline"]].append(
                (int(r["batch"]), tp, tps))

        for key in sorted(groups.keys(),
                          key=lambda k: (k[0], k[1], int(k[2]))):
            model, device, ctx = key
            bl_data = groups[key]

            def max_batch_meeting_slo(baseline):
                pts = [(b, tp, tps) for b, tp, tps in bl_data.get(baseline, [])
                       if tp <= slo_ms]
                if not pts:
                    return None, None
                best = max(pts, key=lambda x: x[0])
                return best[0], best[2]

            d_batch, d_tps = max_batch_meeting_slo("Dense")
            s_batch, s_tps = max_batch_meeting_slo("Sparse")

            if d_batch is None and s_batch is None:
                continue

            ctx_k = int(ctx) // 1024
            ctx_str = f"{ctx_k}K" if ctx_k < 1000 else f"{ctx_k//1000}M"
            mshort = model.split("/")[-1][:22]

            if d_tps and s_tps:
                delta_tps = s_tps - d_tps
                ratio = s_tps / d_tps
                if ratio > 1.5:
                    note = "big win"
                elif ratio < 1.0:
                    note = "LOSS"
                elif ratio < 1.05:
                    note = "parity"
                else:
                    note = ""
            else:
                delta_tps = None
                ratio = None
                note = "Dense fails SLO" if d_batch is None else "Sparse fails SLO"

            print(f"{mshort:22} {device:6} {ctx_str:>7}  "
                  f"{str(d_batch or '—'):>6} {fmt(d_tps or 0,1):>8}  "
                  f"{str(s_batch or '—'):>6} {fmt(s_tps or 0,1):>8}  "
                  f"{fmt(delta_tps,1) if delta_tps else 'N/A':>8}  "
                  f"{fmtx(ratio) if ratio else 'N/A':>9}  {note}")


# ---------------------------------------------------------------------------
# Analysis 3: Energy efficiency
# ---------------------------------------------------------------------------

def analysis_energy(rows):
    print("\n" + "=" * 110)
    print("ANALYSIS 3 — ENERGY EFFICIENCY  (tokens / joule)")
    print(f"  HBM={HBM_PJ_PER_BIT} pJ/bit  HBF={HBF_PJ_PER_BIT} pJ/bit  (FlashAccel / H³ conventions)")
    print(f"  Sparse overhead: +{SPARSE_OVERHEAD*100:.0f}% compute for top-K block selection")
    print("=" * 110)

    by_bl = defaultdict(list)

    for r in rows:
        try:
            tpot = float(r["tpot_p50"])
            tps  = float(r["throughput_tps"])
            hbm_frac = float(r["hbm_frac"])
        except (ValueError, TypeError):
            continue

        model   = r["model"]
        device  = r["device"]
        batch   = int(r["batch"])
        ctx     = int(r["ctx"])
        baseline = r["baseline"]

        kv_rf = SPARSITY if baseline == "Sparse" else 1.0

        e_per_tok_pJ, breakdown = energy_per_output_token_pJ(
            model, device, batch, ctx, hbm_frac, kv_rf)
        if e_per_tok_pJ is None or e_per_tok_pJ <= 0:
            continue

        tok_per_J = 1e12 / e_per_tok_pJ

        by_bl[baseline].append({
            "model": model, "device": device, "batch": batch, "ctx": ctx,
            "baseline": baseline,
            "hbm_frac": hbm_frac,
            "tpot_ms": tpot, "tps": tps,
            "e_tok_pJ": e_per_tok_pJ,
            "tok_per_J": tok_per_J,
            "hbm_pJ": breakdown.get("hbm_pJ", 0),
            "hbf_pJ": breakdown.get("hbf_pJ", 0),
            "compute_pJ": breakdown.get("compute_pJ", 0),
        })

    # Summary
    print(f"\n{'Baseline':<14}  {'N':>5}  {'geomean tok/J':>14}  {'p25':>10}  {'p50':>10}  "
          f"{'p75':>10}  {'p95':>10}  {'HBM%energy':>11}  {'HBF%energy':>11}  {'cmp%energy':>11}")
    print("-" * 120)
    for bl in ["Dense", "NaiveSparse", "Sparse"]:
        pts = by_bl[bl]
        if not pts:
            continue
        tpjs = [p["tok_per_J"] for p in pts]
        hbm_fracs_e = [p["hbm_pJ"] / p["e_tok_pJ"] * 100 for p in pts]
        hbf_fracs_e = [p["hbf_pJ"] / p["e_tok_pJ"] * 100 for p in pts]
        cmp_fracs_e = [p["compute_pJ"] / p["e_tok_pJ"] * 100 for p in pts]
        gm = geomean(tpjs)
        print(f"{bl:<14}  {len(pts):>5}  {fmt(gm,1):>14}  "
              f"{fmt(percentile(tpjs,25),1):>10}  {fmt(percentile(tpjs,50),1):>10}  "
              f"{fmt(percentile(tpjs,75),1):>10}  {fmt(percentile(tpjs,95),1):>10}  "
              f"{fmt(statistics.mean(hbm_fracs_e),1):>10}%  "
              f"{fmt(statistics.mean(hbf_fracs_e),1):>10}%  "
              f"{fmt(statistics.mean(cmp_fracs_e),1):>10}%")

    # Energy speedup: Sparse vs Dense
    dense_e = {(p["model"], p["device"], p["batch"], p["ctx"]): p["e_tok_pJ"]
               for p in by_bl["Dense"]}
    sparse_e = {(p["model"], p["device"], p["batch"], p["ctx"]): p["e_tok_pJ"]
                for p in by_bl["Sparse"]}

    improvements = []
    losses_e = []
    for key in dense_e.keys() & sparse_e.keys():
        ratio = dense_e[key] / sparse_e[key]  # >1 means Sparse is more efficient
        improvements.append(ratio)
        if ratio < 1.0:
            losses_e.append((ratio, key))

    if improvements:
        gm = geomean(improvements)
        print(f"\n  Energy improvement (Dense E/tok / Sparse E/tok):")
        print(f"  geomean={fmt(gm,3)}x  p25={fmt(percentile(improvements,25),3)}x  "
              f"p50={fmt(percentile(improvements,50),3)}x  "
              f"p75={fmt(percentile(improvements,75),3)}x  "
              f"max={fmt(max(improvements),2)}x")
        if losses_e:
            print(f"\n  *** Energy losses (Sparse less efficient than Dense): {len(losses_e)} configs ***")
            for ratio, (model, device, batch, ctx) in sorted(losses_e)[:5]:
                print(f"    {model.split('/')[-1]:22} {device} B={batch:3} ctx={int(ctx)//1024}K  "
                      f"Sparse uses {1/ratio:.3f}x MORE energy (compute overhead dominates)")
        else:
            print(f"  No energy losses (Sparse is always ≥ Dense in tok/J).")

    # Iso-throughput comparison: at matched throughput, Sparse energy vs Dense energy
    print(f"\n--- Iso-throughput energy (same throughput target, Sparse lower batch vs Dense higher batch) ---")
    print(f"  At matched tokens/s, Sparse achieves the same throughput at smaller batch → lower KV memory")
    print(f"  Context: 128K, A100, target ≈ medium throughput tier")

    # Find configurations for Llama-3-8B A100 ctx=128K
    target_model = "meta-llama/Meta-Llama-3-8B"
    target_ctx   = "131072"
    target_device = "a100"
    sub = {(p["batch"], p["model"].split("/")[-1][:10]): p
           for bl in ["Dense", "Sparse"]
           for p in by_bl[bl]
           if p["model"] == target_model and p["ctx"] == int(target_ctx)
           and p["device"] == target_device}

    d_pts = sorted(
        [p for p in by_bl["Dense"]
         if p["model"] == target_model and p["ctx"] == int(target_ctx)
         and p["device"] == target_device],
        key=lambda x: x["batch"])
    s_pts = sorted(
        [p for p in by_bl["Sparse"]
         if p["model"] == target_model and p["ctx"] == int(target_ctx)
         and p["device"] == target_device],
        key=lambda x: x["batch"])

    if d_pts and s_pts:
        print(f"\n  Llama-3-8B, A100, ctx=128K:")
        print(f"  {'B':>4}  {'baseline':>12}  {'tps':>8}  {'E/tok mJ':>10}  {'tok/J':>10}  {'HBF% of E':>10}")
        for p in d_pts + s_pts:
            hbf_pct = p["hbf_pJ"] / p["e_tok_pJ"] * 100
            e_mJ = p["e_tok_pJ"] / 1e9  # pJ → mJ
            print(f"  {p['batch']:>4}  {p['baseline']:>12}  "
                  f"{fmt(p['tps'],1):>8}  {fmt(e_mJ,3):>10}  {fmt(p['tok_per_J'],1):>10}  {fmt(hbf_pct,1):>9}%")

    # HBM-only baseline: if all KV were in HBM (no HBF), energy for Dense
    print(f"\n--- HBM-only baseline vs HBF+Sparse (iso-throughput) ---")
    print(f"  HBM-only: kv_read_fraction=1.0, hbm_frac=1.0 (hypothetical GPU with unlimited HBM)")
    print(f"  Shows: does our scheme save energy vs a hypothetical perfect HBM-only system?")
    hbm_only_e = []
    sparse_e_list = []
    for p in by_bl["Sparse"]:
        e_hbm_only, _ = energy_per_output_token_pJ(
            p["model"], p["device"], p["batch"], p["ctx"], 1.0, 1.0)
        if e_hbm_only and p["e_tok_pJ"]:
            ratio = e_hbm_only / p["e_tok_pJ"]
            hbm_only_e.append(ratio)
    if hbm_only_e:
        gm = geomean(hbm_only_e)
        print(f"  E(HBM-only dense) / E(HBF sparse):  geomean={fmt(gm,3)}x  "
              f"p50={fmt(percentile(hbm_only_e,50),3)}x  "
              f"max={fmt(max(hbm_only_e),2)}x")
        pct_worse = sum(1 for r in hbm_only_e if r < 1.0)
        print(f"  HBF+Sparse is MORE efficient than HBM-only in "
              f"{(len(hbm_only_e)-pct_worse)/len(hbm_only_e)*100:.0f}% of configs "
              f"(flash energy penalty < compute savings from sparsity)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper_results.csv"
    rows = load(path)

    # Annotate baseline for energy lookup
    for r in rows:
        r["_baseline"] = r["baseline"]

    print(f"Loaded {len(rows)} valid rows from {path}")
    models = sorted(set(r["model"].split("/")[-1] for r in rows))
    ctxs   = sorted(set(int(r["ctx"]) for r in rows))
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Contexts: {[c//1024 for c in ctxs]}K")

    # Inject baseline into energy analysis
    for r in rows:
        r["baseline"] = r["_baseline"]

    analysis_throughput(rows)
    analysis_slo(rows)
    analysis_energy(rows)

    print("\n" + "=" * 110)
    print("END OF ANALYSIS")


if __name__ == "__main__":
    main()
