#!/usr/bin/env python3
"""
Paper figure/table package from paper_results.csv, organized as main-paper vs appendix.

Label mapping (CSV -> paper):  Sparse->SpaKV (ours), Dense->H3 (baseline, HBM+HBF hybrid).
Only baseline present in the data is H3 (NaiveSparse is numerically identical; no HBM-only sim).
Main representative device = H100. Operating point = largest batch meeting the TPOT-p50 SLO.

MAIN FIGURES (H100, 4 representative models, bars by context at SLO-max batch):
  fig1_throughput.png     throughput/GPU            (item 1)
  fig2_tpot_p50.png       TPOT p50                  (item 2)
  fig3_tpot_p99.png       TPOT p99                  (item 3)
  fig4_energy.png         energy efficiency tok/J   (item 7)
  latency_throughput.png  TPOT p50+p99 vs throughput, batch swept
  fig_speedup_cdf.png     CDF of SpaKV/H3 speedup across all feasible configs (item 6)

TABLES:
  tab_slo50.tex / tab_slo100.tex   per (model x device) max-batch/tput/speedup, N/A marks (items 4-5)
  tab_speedup_summary.tex          geomean/median/p25/p75/peak, throughput & energy (item 6)
  tab_appendix_full.csv            full (model x device x context) sweep for verification
  headline_stats.txt               text summary

Run:  python3.9 make_paper_figures.py [paper_results.csv]
"""
import csv, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze_paper_results import (
    energy_per_output_token_pJ, geomean, percentile, SPARSITY,
    HBM_PJ_PER_BIT, HBF_PJ_PER_BIT,
)

CSV    = sys.argv[1] if len(sys.argv) > 1 else "paper_results.csv"
OUTDIR = "figures"
DPI    = 300
DEVICE = "h100"                 # main representative device
SLOS   = [50, 100]
CTX_MAIN = [16384, 65536, 131072, 262144, 524288]   # contexts shown in main bar figures
TAB_CTX  = 131072               # representative context for the SLO tables (128K)
FEATURE = [
    ("microsoft/phi-2",             "Phi-2"),
    ("mistralai/Mistral-7B-v0.1",   "Mistral-7B"),
    ("meta-llama/Meta-Llama-3-8B",  "Llama-3-8B"),
    ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B"),
]
ALL_MODELS = [
    ("microsoft/phi-2", "Phi-2"), ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
    ("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"), ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B"),
    ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"), ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"),
    ("Qwen/Qwen-72B", "Qwen-72B"), ("Qwen/Qwen2-72B", "Qwen2-72B"),
    ("meta-llama/Meta-Llama-3.1-405B", "Llama-3.1-405B"),
]
ALL_DEVICES = ["a40", "a100", "h100", "h200"]
C_H3, C_SPA = "#9aa7b4", "#c0392b"
# Per-figure accent colours (baseline stays grey so H3-vs-ours reads clearly)
C_TPUT   = "#c0392b"   # crimson  — throughput
C_LAT    = "#1b9e77"   # green    — latency (lower = better)
C_ENERGY = "#e67e22"   # orange   — energy
C_BATCH  = "#2c6fbb"   # blue     — batch headroom

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "legend.fontsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight", "axes.axisbelow": True,
})

# --------------------------------------------------------------------------
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

ROWS = []
for r in csv.DictReader(open(CSV)):
    if "FAILED" in str(r.get("note", "")) or not r.get("tpot_p50"):
        continue
    if r["baseline"] not in ("Sparse", "Dense"):
        continue
    tp50, tp99, tps = fnum(r["tpot_p50"]), fnum(r["tpot_p99"]), fnum(r["throughput_tps"])
    if tp50 is None or tps is None:
        continue
    ROWS.append(dict(model=r["model"], device=r["device"], batch=int(r["batch"]),
                     ctx=int(r["ctx"]), bl=("SpaKV" if r["baseline"] == "Sparse" else "H3"),
                     tp50=tp50, tp99=tp99 if tp99 is not None else tp50,
                     tps=tps, hbm_frac=fnum(r["hbm_frac"]) or 1.0))
os.makedirs(OUTDIR, exist_ok=True)

def op_point(model, dev, ctx, bl, slo):
    """Row at the largest batch meeting the TPOT-p50 SLO, or None."""
    pts = [x for x in ROWS if x["model"] == model and x["device"] == dev
           and x["ctx"] == ctx and x["bl"] == bl and x["tp50"] <= slo]
    return max(pts, key=lambda z: z["batch"]) if pts else None

def tok_per_J(row):
    kv_rf = SPARSITY if row["bl"] == "SpaKV" else 1.0
    e, _ = energy_per_output_token_pJ(row["model"], row["device"], row["batch"],
                                      row["ctx"], row["hbm_frac"], kv_rf)
    return (1e12 / e) if e and e > 0 else None

def ctx_lbl(c): return f"{c//1024}K" if c < 1024*1024 else f"{c//1024//1024}M"

def grouped_positions(model_per_bar):
    xs, pos = [], 0.0
    for i, m in enumerate(model_per_bar):
        if i > 0: pos += 1.7 if m != model_per_bar[i - 1] else 1.0
        xs.append(pos)
    return np.array(xs)

def two_level_xaxis(ax, x, model_per_bar, ctx_per_bar, fs=8.5):
    ax.set_xticks(x); ax.set_xticklabels(ctx_per_bar, fontsize=fs)
    ax.tick_params(axis="x", length=0)
    i = 0
    while i < len(model_per_bar):
        j = i
        while j + 1 < len(model_per_bar) and model_per_bar[j + 1] == model_per_bar[i]: j += 1
        ax.text((x[i] + x[j]) / 2, -0.14, model_per_bar[i], ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9.5, fontweight="bold")
        if j + 1 < len(model_per_bar):
            ax.axvline((x[j] + x[j + 1]) / 2, color="0.85", lw=0.8)
        i = j + 1
    ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)

def save(fig, name):
    p = os.path.join(OUTDIR, name); fig.savefig(p); plt.close(fig); print("  wrote", p)

# --------------------------------------------------------------------------
# Items 1-3,7: absolute metric by context, H100, at SLO-max batch (100ms)
# --------------------------------------------------------------------------
def abs_by_ctx(metric, ylab, fname, logy=False, slo=100, ratio_label=False, accent=C_SPA, only_better=False, title=None,
               figsize=(13, 4.8), fs=8.5, suptitle_fs=12):
    print(f"[fig] {fname}")
    mods, ctxs, h3v, spav, ratios = [], [], [], [], []
    for model, short in FEATURE:
        for c in CTX_MAIN:
            oh, os_ = op_point(model, DEVICE, c, "H3", slo), op_point(model, DEVICE, c, "SpaKV", slo)
            if not (oh and os_): continue
            if metric == "energy":
                hv, sv = tok_per_J(oh), tok_per_J(os_)
                if not (hv and sv): continue
            else:
                hv, sv = oh[metric], os_[metric]
            if only_better and not (hv and sv / hv > 1.05):   # drop configs where SpaKV ties H3
                continue
            mods.append(short); ctxs.append(ctx_lbl(c)); h3v.append(hv); spav.append(sv)
            ratios.append(sv / hv if hv else 0)
    fig, ax = plt.subplots(figsize=figsize)
    x = grouped_positions(mods); w = 0.4
    ax.bar(x - w/2, h3v, w, label="H3 (baseline)", color=C_H3, edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w/2, spav, w, label="SpaKV (ours)", color=accent, edgecolor="black", lw=0.5)
    if ratio_label:
        ax.bar_label(b2, labels=[f"{r:.1f}×" if r > 1.05 else "" for r in ratios], padding=2, fontsize=fs-1)
    if logy: ax.set_yscale("log")
    if metric in ("tp50", "tp99"):
        ax.axhline(slo, color="0.4", ls="-.", lw=0.9)
        ax.text(0.995, slo, f"{slo} ms SLO", transform=ax.get_yaxis_transform(),
                fontsize=7.5, ha="right", va="bottom", color="0.3")
    two_level_xaxis(ax, x, mods, ctxs, fs=fs)
    ax.set_ylabel(ylab)
    ax.margins(y=0.12)
    ax.tick_params(axis="y", labelsize=fs)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=fs)
    fig.suptitle(title or f"{ylab.split('(')[0].strip()} by context — {DEVICE.upper()}, at SLO-max batch (TPOT-p50 ≤ {slo} ms)",
                 y=1.11, fontsize=suptitle_fs)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, fname)

# --------------------------------------------------------------------------
# Items 2-3: latency under SLO — TPOT p50 & p99, H3 vs SpaKV at MATCHED batch
# (same work -> SpaKV reads 10% of KV -> always lower TPOT). Lower is better.
# --------------------------------------------------------------------------
def matched_point(model, dev, ctx, bl, batch):
    for x in ROWS:
        if x["model"]==model and x["device"]==dev and x["ctx"]==ctx and x["bl"]==bl and x["batch"]==batch:
            return x
    return None

def matched_batch(model, dev, ctx):
    bh = {x["batch"] for x in ROWS if x["model"]==model and x["device"]==dev and x["ctx"]==ctx and x["bl"]=="H3"}
    bs = {x["batch"] for x in ROWS if x["model"]==model and x["device"]==dev and x["ctx"]==ctx and x["bl"]=="SpaKV"}
    common = bh & bs
    return max(common) if common else None

def latency_slo_fig():
    print("[fig] latency_slo.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharex=True)
    fig.subplots_adjust(wspace=0.22)
    for pi, (ax, pct) in enumerate(zip(axes, ["tp50", "tp99"])):
        mods, ctxs, h3v, spav = [], [], [], []
        for model, short in FEATURE:
            for c in CTX_MAIN:
                b = matched_batch(model, DEVICE, c)
                if b is None: continue
                oh, os_ = matched_point(model, DEVICE, c, "H3", b), matched_point(model, DEVICE, c, "SpaKV", b)
                if not (oh and os_): continue
                mods.append(short); ctxs.append(ctx_lbl(c)); h3v.append(oh[pct]); spav.append(os_[pct])
        x = grouped_positions(mods); w = 0.4
        ax.bar(x - w/2, h3v, w, label="H3 (baseline)", color=C_H3, edgecolor="black", lw=0.5)
        ax.bar(x + w/2, spav, w, label="SpaKV (ours)", color=C_LAT, edgecolor="black", lw=0.5)
        ax.set_yscale("log")
        for s in SLOS:
            ax.axhline(s, color="0.4", ls="-.", lw=0.8)
        if pi == 1:   # label SLO lines once, in the right margin (free space)
            for s in SLOS:
                ax.text(1.012, s, f"{s} ms", transform=ax.get_yaxis_transform(),
                        fontsize=8, ha="left", va="center", color="0.3", clip_on=False)
        two_level_xaxis(ax, x, mods, ctxs, fs=7.5)
        ax.set_ylabel(f"TPOT {'p50' if pct=='tp50' else 'p99'} (ms, log)")
        ax.set_title(f"TPOT {'p50' if pct=='tp50' else 'p99'}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Decode latency at matched batch — {DEVICE.upper()}", y=1.11, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, "latency_slo.png")

# --------------------------------------------------------------------------
# Item 6: speedup population (all feasible op-points), CDF + summary table
# --------------------------------------------------------------------------
def speedup_population():
    """All (model x device x ctx x SLO) where both H3 & SpaKV meet the SLO."""
    tps, en = [], []
    for model, _ in ALL_MODELS:
        for dev in ALL_DEVICES:
            for c in sorted({x["ctx"] for x in ROWS if x["model"] == model and x["device"] == dev}):
                for slo in SLOS:
                    oh, os_ = op_point(model, dev, c, "H3", slo), op_point(model, dev, c, "SpaKV", slo)
                    if not (oh and os_ and oh["tps"] > 0): continue
                    tps.append(os_["tps"] / oh["tps"])
                    eh, es = tok_per_J(oh), tok_per_J(os_)
                    if eh and es: en.append(es / eh)
    return tps, en

def cdf_fig(tps, en):
    print("[fig] fig_speedup_cdf.png")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for vals, col, lab in [(sorted(tps), C_TPUT, "Throughput"), (sorted(en), C_ENERGY, "Energy efficiency")]:
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=col, lw=2, label=f"{lab} (median {percentile(vals,50):.2f}×)")
    ax.axvline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Speedup vs H3 (SpaKV / H3)"); ax.set_ylabel("Cumulative fraction of configs")
    ax.set_title(f"Speedup CDF across all SLO-feasible configs (N={len(tps)})")
    ax.legend(frameon=False, loc="lower right"); ax.set_ylim(0, 1)
    fig.tight_layout(); save(fig, "fig_speedup_cdf.png")

def batch_headroom_fig(slo=100):
    """Single-column horizontal bars: max batch under SLO, only where SpaKV is larger."""
    print("[fig] fig_maxbatch.png")
    labs, h3b, spab = [], [], []
    for model, short in FEATURE:
        for c in CTX_MAIN:
            oh, os_ = op_point(model, DEVICE, c, "H3", slo), op_point(model, DEVICE, c, "SpaKV", slo)
            if not (oh and os_) or os_["batch"] <= oh["batch"]:
                continue
            labs.append(f"{short}, {ctx_lbl(c)}"); h3b.append(oh["batch"]); spab.append(os_["batch"])
    y = np.arange(len(labs))[::-1]; h = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    ax.barh(y + h/2, h3b, h, color=C_H3, edgecolor="black", lw=0.4, label="H3 (baseline)")
    ax.barh(y - h/2, spab, h, color=C_BATCH, edgecolor="black", lw=0.4, label="SpaKV (ours)")
    for yi, hb, sb in zip(y, h3b, spab):
        ax.text(sb + 0.15, yi - h/2, f"{sb//hb}×", va="center", ha="left", fontsize=7, color=C_BATCH)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel("Max batch under SLO (TPOT-p50 ≤ 100 ms)", fontsize=8.5)
    ax.set_xlim(0, max(spab) * 1.15)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.3, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "fig_maxbatch.png")

def energy_compact_fig(slo=100):
    """Single-column horizontal bars: tok/J (H3 vs SpaKV) over the same flash-bound configs as the batch figure."""
    print("[fig] fig4_energy.png")
    labs, h3e, spae, fac = [], [], [], []
    for model, short in FEATURE:
        for c in CTX_MAIN:
            oh, os_ = op_point(model, DEVICE, c, "H3", slo), op_point(model, DEVICE, c, "SpaKV", slo)
            if not (oh and os_) or os_["batch"] <= oh["batch"]:
                continue
            eh, es = tok_per_J(oh), tok_per_J(os_)
            if not (eh and es):
                continue
            labs.append(f"{short}, {ctx_lbl(c)}"); h3e.append(eh); spae.append(es); fac.append(es / eh)
    y = np.arange(len(labs))[::-1]; h = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    ax.barh(y + h/2, h3e, h, color=C_H3, edgecolor="black", lw=0.4, label="H3 (baseline)")
    ax.barh(y - h/2, spae, h, color=C_ENERGY, edgecolor="black", lw=0.4, label="SpaKV (ours)")
    for yi, sv, r in zip(y, spae, fac):
        ax.text(sv + max(spae)*0.01, yi - h/2, f"{r:.1f}×", va="center", ha="left", fontsize=7, color="#a85a1c")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel("Energy efficiency (tok/J)", fontsize=8.5)
    ax.set_xlim(0, max(spae) * 1.16)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.3, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "fig4_energy.png")

def per_hardware_gains_fig():
    """Single-column: geomean throughput & energy gain per GPU over flash-bound configs."""
    print("[fig] fig_hw_gains.png")
    gpus = ALL_DEVICES
    tps_g, en_g = [], []
    for dev in gpus:
        tr, er = [], []
        for model, _ in ALL_MODELS:
            for c in sorted({x["ctx"] for x in ROWS if x["model"] == model and x["device"] == dev}):
                for slo in SLOS:
                    oh, os_ = op_point(model, dev, c, "H3", slo), op_point(model, dev, c, "SpaKV", slo)
                    if not (oh and os_ and oh["tps"] > 0): continue
                    r = os_["tps"] / oh["tps"]
                    if r > 1.01:                       # flash-bound (gain) regime
                        tr.append(r)
                        eh, es = tok_per_J(oh), tok_per_J(os_)
                        if eh and es: er.append(es / eh)
        tps_g.append(geomean(tr) if tr else 0.0)
        en_g.append(geomean(er) if er else 0.0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(gpus)); w = 0.38
    b1 = ax.bar(x - w/2, tps_g, w, label="Throughput", color=C_TPUT, edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w/2, en_g,  w, label="Energy eff.", color=C_ENERGY, edgecolor="black", lw=0.5)
    ax.bar_label(b1, fmt="%.1f×", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f×", padding=2, fontsize=8)
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g.upper() for g in gpus])
    ax.set_ylabel("Geomean gain vs H3")
    ax.set_ylim(0, max(tps_g + en_g) * 1.18)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); save(fig, "fig_hw_gains.png")

def speedup_summary_table(tps, en):
    def row(name, v):
        return (f"{name} & {geomean(v):.2f} & {percentile(v,50):.2f} & {percentile(v,25):.2f} "
                f"& {percentile(v,75):.2f} & {max(v):.2f}\\\\\n")
    with open(os.path.join(OUTDIR, "tab_speedup_summary.tex"), "w") as f:
        f.write("% SpaKV / H3 speedup over all SLO-feasible configs (all models x devices x ctx x SLO)\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Metric & Geomean & Median & p25 & p75 & Peak\\\\\n\\midrule\n")
        f.write(row("Throughput/GPU ($\\times$)", tps))
        f.write(row("Energy eff. ($\\times$)", en))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote", os.path.join(OUTDIR, "tab_speedup_summary.tex"))

# --------------------------------------------------------------------------
# Items 4-5: SLO tables (model x device) at representative context, N/A marks
# --------------------------------------------------------------------------
def slo_table(slo):
    name = f"tab_slo{slo}.tex"
    print(f"[tab] {name}")
    with open(os.path.join(OUTDIR, name), "w") as f:
        f.write(f"% Max batch / throughput at TPOT-p50<= {slo} ms, context={ctx_lbl(TAB_CTX)}. "
                f"'--' = no batch meets the SLO (OOM/latency-infeasible).\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("Model & Dev & H3 b & H3 tok/s & SpaKV b & SpaKV tok/s & Speedup\\\\\n\\midrule\n")
        for model, short in ALL_MODELS:
            for dev in ALL_DEVICES:
                oh, os_ = op_point(model, dev, TAB_CTX, "H3", slo), op_point(model, dev, TAB_CTX, "SpaKV", slo)
                if oh is None and os_ is None:
                    # only print a row if the context exists for this model/device at all
                    if not any(x["model"]==model and x["device"]==dev and x["ctx"]==TAB_CTX for x in ROWS):
                        continue
                hb = str(oh["batch"]) if oh else "--"
                ht = f"{oh['tps']:.0f}" if oh else "--"
                sb = str(os_["batch"]) if os_ else "--"
                st = f"{os_['tps']:.0f}" if os_ else "--"
                sp = f"{os_['tps']/oh['tps']:.2f}$\\times$" if (oh and os_) else "--"
                f.write(f"{short} & {dev} & {hb} & {ht} & {sb} & {st} & {sp}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote", os.path.join(OUTDIR, name))

# --------------------------------------------------------------------------
# Appendix: full sweep CSV for verification
# --------------------------------------------------------------------------
def appendix_csv(slo=100):
    path = os.path.join(OUTDIR, "tab_appendix_full.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "device", "ctx", f"H3_maxb@{slo}ms", "H3_tps", "H3_tpot_p50", "H3_tpot_p99",
                    "SpaKV_maxb", "SpaKV_tps", "SpaKV_tpot_p50", "SpaKV_tpot_p99", "tps_speedup", "energy_speedup"])
        for model, short in ALL_MODELS:
            for dev in ALL_DEVICES:
                for c in sorted({x["ctx"] for x in ROWS if x["model"] == model and x["device"] == dev}):
                    oh, os_ = op_point(model, dev, c, "H3", slo), op_point(model, dev, c, "SpaKV", slo)
                    if oh is None and os_ is None: continue
                    def g(o, k): return f"{o[k]:.2f}" if o else ""
                    sp = f"{os_['tps']/oh['tps']:.3f}" if (oh and os_ and oh['tps']>0) else ""
                    eh, es = (tok_per_J(oh) if oh else None), (tok_per_J(os_) if os_ else None)
                    esp = f"{es/eh:.3f}" if (eh and es) else ""
                    w.writerow([short, dev, c, oh["batch"] if oh else "", g(oh,"tps"), g(oh,"tp50"), g(oh,"tp99"),
                                os_["batch"] if os_ else "", g(os_,"tps"), g(os_,"tp50"), g(os_,"tp99"), sp, esp])
    print("  wrote", path)

# --------------------------------------------------------------------------
def main():
    print(f"Loaded {len(ROWS)} rows (SpaKV+H3). Main device={DEVICE}")
    abs_by_ctx("tps",  "Throughput / GPU (tok/s)",     "fig1_throughput.png", logy=False, ratio_label=True, accent=C_TPUT, only_better=True)
    batch_headroom_fig()
    latency_slo_fig()                                                          # items 2-3 (p50 & p99)
    energy_compact_fig()
    tps, en = speedup_population()
    cdf_fig(tps, en)
    per_hardware_gains_fig()
    speedup_summary_table(tps, en)
    for slo in SLOS: slo_table(slo)
    appendix_csv(100)
    with open(os.path.join(OUTDIR, "headline_stats.txt"), "w") as f:
        f.write("SpaKV vs H3 (only baseline in data). Operating point = SLO-max batch.\n")
        f.write(f"Energy model: HBM={HBM_PJ_PER_BIT} pJ/bit, HBF={HBF_PJ_PER_BIT} pJ/bit; top-K={SPARSITY}\n\n")
        f.write(f"Throughput speedup (all SLO-feasible configs, N={len(tps)}): "
                f"geomean={geomean(tps):.2f}x median={percentile(tps,50):.2f}x "
                f"p25={percentile(tps,25):.2f}x p75={percentile(tps,75):.2f}x peak={max(tps):.1f}x\n")
        f.write(f"Energy speedup (N={len(en)}): geomean={geomean(en):.2f}x median={percentile(en,50):.2f}x "
                f"p25={percentile(en,25):.2f}x p75={percentile(en,75):.2f}x peak={max(en):.1f}x\n\n")
        f.write("Gaps: Llama-3.1-405B has no usable rows; 67B-72B models are SLO-infeasible at long\n"
                "context (shown as '--' in tables). NaiveSparse identical to H3; no HBM-only baseline.\n")
    print("  wrote", os.path.join(OUTDIR, "headline_stats.txt"))
    print(f"Done. throughput geomean={geomean(tps):.2f}x (median {percentile(tps,50):.2f}x), "
          f"energy geomean={geomean(en):.2f}x")

if __name__ == "__main__":
    main()
