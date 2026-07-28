#!/usr/bin/env python3
"""TPOT at SPLASH's throughput operating point, as TWO separate plots (p50, p99).
For each (model, context) we take the (batch, tp) at which SPLASH maximises per-GPU
goodput under the SLO (weights must fit in HBM), then read every baseline's TPOT at
that SAME operating point from the sweep. Shows that at the load SPLASH sustains, the
baselines miss the SLO by 2-4x. Two-level grouped-bar layout like the goodput plots.
HBM-only dropped (always OOM at long context). No new sweep -> results/sweep_b200.csv."""
import csv
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from sweep_capacity import weight_bytes, HBM_GB

HBM = HBM_GB["blackwell"] * 1e9
SLO = 100.0
ROWS = [r for r in csv.DictReader(open("results/sweep_b200_weight_valid.csv"))
        if r["status"] == "OK"]
MODELS = [("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
          ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8×7B"),
          ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
          ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"),
          ("Qwen/Qwen3-235B-A22B", "Qwen3-235B")]
CTXS = [524288, 786432, 1048576, 2097152]
CTXLAB = {524288: "512K", 786432: "768K", 1048576: "1M", 2097152: "2M"}
SYS = [("Dense", "H3"), ("Naive", "Naive Sparse"), ("Sparse", "SPLASH (Ours)")]
COL = {"Dense": "#e6913a", "Naive": "#eed6a0", "Sparse": "#8279bd"}
EDGE = {"Dense": "#2b2b2b", "Naive": "#2b2b2b", "Sparse": "#3d3570"}
HATCH = {"Sparse": "///"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 0, "ytick.major.size": 2.5, "hatch.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

idx = {}
for r in ROWS:
    idx[(r["model"], int(r["context_length"]), r["baseline"], int(r["batch"]), int(r["tp"]))] = \
        (float(r["tpot_p50_ms"]), float(r["tpot_p99_ms"]))

def wfits(mk, tp):
    return weight_bytes(mk) / tp <= HBM

# SPLASH throughput operating point per (model, ctx): argmax goodput under SLO
work = []
for mk, md in MODELS:
    for c in CTXS:
        op = None
        for r in ROWS:
            if r["model"] == mk and int(r["context_length"]) == c and r["baseline"] == "Sparse":
                p50, tp = float(r["tpot_p50_ms"]), int(r["tp"])
                if p50 <= SLO and wfits(mk, tp):
                    g = int(r["batch"]) * 1000.0 / p50 / tp
                    if op is None or g > op[0]:
                        op = (g, int(r["batch"]), tp)
        cell = {}
        if op:
            _, b, tp = op
            for sk, _ in SYS:
                cell[sk] = idx.get((mk, c, sk, b, tp))
        else:
            cell = {sk: None for sk, _ in SYS}
        work.append(dict(model=md, ctx=CTXLAB[c], vals=cell))

NC = len(CTXS)
centers = []; x = 0.0
for m in range(len(MODELS)):
    for _ in range(NC):
        centers.append(x); x += 1.0
    x += 1.15
centers = np.array(centers)


def render(pi, plabel, fname):
    bw = 0.235
    YMAX = 400
    fig, ax = plt.subplots(figsize=(7.0, 1.2))
    for i, (sk, _) in enumerate(SYS):
        xs = centers + (i - 1) * bw
        for w, xx in zip(work, xs):
            v = w["vals"][sk]
            if v is None:
                continue
            ax.bar(xx, min(v[pi], YMAX), bw, color=COL[sk], edgecolor=EDGE[sk], linewidth=0.5,
                   hatch=HATCH.get(sk), zorder=3)

    ax.axhline(SLO, ls=(0, (5, 2)), lw=1.0, color="#d21f1f", zorder=4)
    ax.set_ylim(0, YMAX); ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_ylabel("TPOT (ms)", fontsize=8.5)
    ax.grid(True, axis="y", ls=(0, (4, 3)), lw=0.5, color="#cfcfcf", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for m in range(1, len(MODELS)):
        xsep = (centers[m * NC - 1] + centers[m * NC]) / 2
        ax.axvline(xsep, color="#dddddd", lw=0.6, zorder=1)

    ax.set_xticks(centers); ax.set_xticklabels([w["ctx"] for w in work], fontsize=5.6)
    ax.tick_params(axis="x", pad=1.2)
    for m, (_, name) in enumerate(MODELS):
        xc = (centers[m * NC] + centers[m * NC + NC - 1]) / 2
        ax.text(xc, -0.20, name, ha="center", va="top", fontsize=7.2,
                transform=ax.get_xaxis_transform())

    ax.text(0.005, 0.95, f"{plabel} TPOT at SPLASH's throughput operating point   ·   "
            f"SLO = {SLO:.0f} ms (dashed)", transform=ax.transAxes, fontsize=6.6,
            style="italic", color="#555", ha="left", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[s], ec=EDGE[s], lw=0.5, hatch=HATCH.get(s))
               for s, _ in SYS]
    ax.legend(handles, [d for _, d in SYS], loc="lower center", bbox_to_anchor=(0.5, 0.995),
              ncol=3, frameon=False, fontsize=7.6, handlelength=1.25, handleheight=1.05,
              handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.997, top=0.80, bottom=0.30)
    for e in ("png", "pdf"):
        fig.savefig(f"results/plots/{fname}.{e}")
    plt.close(fig)
    print(f"wrote {fname}")


render(0, "p50", "result_tpot_p50")
render(1, "p99", "result_tpot_p99")


def render_singlecol(fname):
    """Compact single-column (IEEE 3.4in) TPOT: 5 model panels stacked, 4 long
    contexts; bar = p50, whisker = p99, dashed SLO line. All datapoints."""
    cen = np.arange(NC)
    bw = 0.26
    YMAX = 400
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(3.4, 3.05),
                             sharex=True, sharey=True)
    for ax, (mk, md) in zip(axes, MODELS):
        mw = [w for w in work if w["model"] == md]
        for i, (sk, _) in enumerate(SYS):
            xs = cen + (i - 1) * bw
            for w, xx in zip(mw, xs):
                v = w["vals"][sk]
                if v is None:
                    ax.plot(xx, 10, marker="x", color="#d21f1f", ms=3, mew=1.0, zorder=6)
                    continue
                p50 = min(v[0], YMAX); p99 = min(v[1], YMAX)
                ax.bar(xx, p50, bw, color=COL[sk], edgecolor=EDGE[sk], linewidth=0.4,
                       hatch=HATCH.get(sk), zorder=3)
                if p99 > p50:
                    ax.plot([xx, xx], [p50, p99], color="#333", lw=0.5, zorder=5)
        ax.axhline(SLO, ls=(0, (4, 2)), lw=0.8, color="#d21f1f", zorder=4)
        ax.set_ylim(0, YMAX); ax.set_yticks([0, 200, 400])
        ax.grid(True, axis="y", ls=(0, (4, 3)), lw=0.4, color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, NC - 0.4)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.012, 0.9, md, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.0, fontweight="bold")
        ax.tick_params(labelsize=5.8)
    axes[-1].set_xticks(cen)
    axes[-1].set_xticklabels([CTXLAB[c] for c in CTXS], rotation=45, fontsize=5.8)
    fig.supylabel("TPOT (ms)", fontsize=7.0, x=0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[s], ec=EDGE[s], lw=0.4,
                             hatch=HATCH.get(s)) for s, _ in SYS]
    handles.append(plt.Line2D([], [], ls=(0, (4, 2)), color="#d21f1f", lw=0.8))
    labs = [d for _, d in SYS] + ["SLO 100 ms"]
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.004),
               ncol=4, frameon=False, fontsize=6.0, handlelength=1.1,
               handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.92, bottom=0.115, hspace=0.14)
    for e in ("png", "pdf"):
        fig.savefig(f"results/plots/{fname}.{e}")
    plt.close(fig)
    print("wrote", fname)


render_singlecol("result_tpot_singlecol")
