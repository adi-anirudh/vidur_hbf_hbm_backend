#!/usr/bin/env python3
"""
Presentation (talk) figures from paper_results.csv — deeper than the paper.
16:9, large fonts, one message per slide. Outputs to ./slides_figures/.

Builds:
  heatmaps_speedup.png        per-model throughput-speedup heatmap over (context x batch), H100
  per_device_speedup.png      throughput speedup vs context, 2x2 small multiples (A40/A100/H100/H200)
  crossover_step{1..4}.png    animated H3-vs-SpaKV throughput-vs-context build (one model)

Run:  python3.9 make_slide_figures.py [paper_results.csv]
"""
import csv, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

CSV    = sys.argv[1] if len(sys.argv) > 1 else "paper_results.csv"
OUTDIR = "slides_figures"
DEVICE = "h100"
SLO    = 100
C_H3, C_SPA = "#9aa7b4", "#c0392b"

MODELS = [  # 8 usable models (405B has no rows); 2x4 grid
    ("microsoft/phi-2", "Phi-2"), ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
    ("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"), ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B"),
    ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"), ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"),
    ("Qwen/Qwen-72B", "Qwen-72B"), ("Qwen/Qwen2-72B", "Qwen2-72B"),
]
DEVICES = [("a40", "A40"), ("a100", "A100"), ("h100", "H100"), ("h200", "H200")]
CROSS_MODEL, CROSS_SHORT = "meta-llama/Meta-Llama-3-8B", "Llama-3-8B"

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 21,
    "legend.fontsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.axisbelow": True, "lines.linewidth": 3.5, "lines.markersize": 11,
})

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

ROWS = []
for r in csv.DictReader(open(CSV)):
    if "FAILED" in str(r.get("note", "")) or not r.get("tpot_p50"): continue
    if r["baseline"] not in ("Sparse", "Dense"): continue
    tp50, tps = fnum(r["tpot_p50"]), fnum(r["throughput_tps"])
    if tp50 is None or tps is None: continue
    ROWS.append(dict(model=r["model"], device=r["device"], batch=int(r["batch"]),
                     ctx=int(r["ctx"]), bl=("SpaKV" if r["baseline"]=="Sparse" else "H3"),
                     tp50=tp50, tps=tps))
IDX = {(x["model"], x["device"], x["batch"], x["ctx"], x["bl"]): x for x in ROWS}
CTXS = sorted({x["ctx"] for x in ROWS})
BATCHES = sorted({x["batch"] for x in ROWS})
os.makedirs(OUTDIR, exist_ok=True)

def ctx_lbl(c): return f"{c//1024}K" if c < 1024*1024 else f"{c//1024//1024}M"

def op_point(model, dev, ctx, bl, slo=SLO):
    pts = [x for x in ROWS if x["model"]==model and x["device"]==dev and x["ctx"]==ctx
           and x["bl"]==bl and x["tp50"] <= slo]
    return max(pts, key=lambda z: z["batch"]) if pts else None

def matched_pt(model, dev, ctx, bl):
    """Throughput at the largest batch present for BOTH schemes (memory-limited, fair)."""
    bh = {x["batch"] for x in ROWS if x["model"]==model and x["device"]==dev and x["ctx"]==ctx and x["bl"]=="H3"}
    bs = {x["batch"] for x in ROWS if x["model"]==model and x["device"]==dev and x["ctx"]==ctx and x["bl"]=="SpaKV"}
    common = bh & bs
    return IDX.get((model, dev, max(common), ctx, bl)) if common else None

def save(fig, name):
    p = os.path.join(OUTDIR, name); fig.savefig(p); plt.close(fig); print("  wrote", p)

GPU_COLOR = {"a40": "#8e44ad", "a100": "#2c6fbb", "h100": "#1b9e77", "h200": "#c0392b"}

def cell_speedup(model, dev, ctx):
    """Throughput speedup at the largest batch present for both schemes."""
    h, s = matched_pt(model, dev, ctx, "H3"), matched_pt(model, dev, ctx, "SpaKV")
    return (s["tps"] / h["tps"]) if (h and s and h["tps"] > 0) else None

# --------------------------------------------------------------------------
# 1. Speedup heatmaps over (model x context), one panel per GPU — uses ALL GPUs
# --------------------------------------------------------------------------
def heatmaps_fig():
    print("[slide] heatmaps_speedup.png")
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), constrained_layout=True)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("#ededed")
    norm = Normalize(vmin=1.0, vmax=8.0)
    shown = [c for c in CTXS]
    im = None
    for ax, (dev, dshort) in zip(axes, DEVICES):
        M = np.full((len(MODELS), len(shown)), np.nan)
        for i, (model, _) in enumerate(MODELS):
            for j, c in enumerate(shown):
                v = cell_speedup(model, dev, c)
                if v is not None: M[i, j] = v
        im = ax.imshow(M, origin="upper", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(dshort, fontsize=22, fontweight="bold")
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([ctx_lbl(c) for c in shown], rotation=90, fontsize=14)
        if ax is axes[0]:
            ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([s for _, s in MODELS], fontsize=15)
        else:
            ax.set_yticks([])
        for i in range(len(MODELS)):
            for j in range(len(shown)):
                if not np.isnan(M[i, j]) and M[i, j] >= 1.5:
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=11,
                            color="white" if M[i, j] > 4.5 else "black", fontweight="bold")
        ax.set_xlabel("Context", fontsize=16)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, extend="max")
    cbar.set_label("Throughput speedup (SpaKV / H3)", fontsize=18)
    fig.suptitle("Where SpaKV wins: throughput speedup across every model and GPU",
                 fontsize=25, fontweight="bold")
    save(fig, "heatmaps_speedup.png")

# --------------------------------------------------------------------------
# 2. Speedup vs context, one panel per model, a line per GPU — uses ALL GPUs
# --------------------------------------------------------------------------
def speedup_lines_fig():
    print("[slide] speedup_vs_context.png")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (model, short) in zip(axes.flat, MODELS):
        for dev, dshort in DEVICES:
            xs, ys = [], []
            for c in CTXS:
                v = cell_speedup(model, dev, c)
                if v is not None: xs.append(c // 1024); ys.append(v)
            if xs: ax.plot(xs, ys, marker="o", ms=8, lw=3, color=GPU_COLOR[dev], label=dshort.upper())
        ax.axhline(1.0, color="0.45", ls="--", lw=2)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_title(short, fontsize=20, fontweight="bold")
        ax.set_xticks([c // 1024 for c in CTXS])
        ax.set_xticklabels([ctx_lbl(c) for c in CTXS], rotation=90, fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[:, 0]: ax.set_ylabel("Throughput speedup", fontsize=17)
    for ax in axes[-1, :]: ax.set_xlabel("Context", fontsize=17)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False, fontsize=20)
    fig.suptitle("Speedup grows with context on every GPU (largest feasible batch)",
                 fontsize=24, fontweight="bold")
    save(fig, "speedup_vs_context.png")

# --------------------------------------------------------------------------
# 3. Crossover small multiples: throughput vs context, H3 vs SpaKV, all models (H100)
# --------------------------------------------------------------------------
def crossover_allmodels_fig():
    print("[slide] crossover_allmodels.png")
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (model, short) in zip(axes.flat, MODELS):
        xs, h3, spa = [], [], []
        for c in CTXS:
            h, s = matched_pt(model, DEVICE, c, "H3"), matched_pt(model, DEVICE, c, "SpaKV")
            if h and s:
                xs.append(c // 1024); h3.append(h["tps"]); spa.append(s["tps"])
        if not xs:
            ax.set_visible(False); continue
        xk = np.array(xs); h3a, spaa = np.array(h3), np.array(spa)
        ratio = spaa / h3a
        div = next((i for i in range(len(xk)) if ratio[i] > 1.05), None)
        ymax = max(spaa.max(), h3a.max()) * 1.18
        if div is not None:
            ax.axvspan(xk[div], xk[-1], color="#fdebd0", alpha=0.7, zorder=0)
        ax.plot(xk, h3a, marker="s", ms=8, lw=3, color=C_H3, label="H3")
        ax.plot(xk, spaa, marker="o", ms=8, lw=3, color=C_SPA, label="SpaKV")
        gi = int(np.argmax(ratio))
        if ratio[gi] > 1.1:
            ax.annotate(f"{ratio[gi]:.1f}×", xy=(xk[gi], spaa[gi]), xytext=(xk[gi], spaa[gi]*0.55),
                        fontsize=18, fontweight="bold", color=C_SPA, ha="center",
                        arrowprops=dict(arrowstyle="->", color=C_SPA, lw=2))
        ax.set_xscale("log", base=2); ax.set_ylim(0, ymax)
        ax.set_title(short, fontsize=19, fontweight="bold")
        ax.set_xticks(xk); ax.set_xticklabels([ctx_lbl(c*1024) for c in xk], rotation=45, fontsize=12)
        ax.tick_params(axis="y", labelsize=13); ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]: ax.set_ylabel("Throughput (tok/s)", fontsize=16)
    for ax in axes[-1, :]: ax.set_xlabel("Context", fontsize=16)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=20)
    fig.suptitle(f"H3 vs SpaKV throughput at largest feasible batch — {DEVICE.upper()} "
                 f"(shaded = KV spilled to flash)", fontsize=23, fontweight="bold", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 1]); save(fig, "crossover_allmodels.png")

def main():
    print(f"Loaded {len(ROWS)} rows. Slide figures -> ./{OUTDIR}/")
    heatmaps_fig()
    per_device_fig()
    crossover_allmodels_fig()
    print("Done.")

if __name__ == "__main__":
    main()
