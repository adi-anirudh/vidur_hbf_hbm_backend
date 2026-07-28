#!/usr/bin/env python3
"""Normalized decode-throughput result plots (MICRO/ISCA grouped-bar style, 2-col wide,
minimal height). Per-GPU throughput at the TPOT SLO, normalized to SPLASH (=1.0).
Split into two figures by the user's operating-point plan:
  (a) SLO = 50 ms  over SHORT contexts  {128K,192K,256K,384K}
  (b) SLO = 100 ms over LONG  contexts  {512K,768K,1M,2M}
Each: 6-model logical sweep (Phi-2 -> Llama-3.1-405B) x 4 contexts.
Style: brown->orange->tan->purple(hatched, ours), red x for infeasible, top legend."""
import csv
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROWS = [r for r in csv.DictReader(open("results/sweep_b200.csv")) if r["status"] == "OK"]
MODELS = [("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
          ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8×7B"), ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
          ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"), ("Qwen/Qwen3-235B-A22B", "Qwen3-235B")]
SYS = [("HBM-only", "HBM-only"), ("Dense", "H3"), ("Naive", "Naive"), ("Sparse", "SPLASH")]
DISP = {"HBM-only": "HBM-only", "H3": "H3", "Naive": "Naive Sparse", "SPLASH": "SPLASH (Ours)"}
COL   = {"HBM-only": "#9e5e34", "H3": "#e6913a", "Naive": "#eed6a0", "SPLASH": "#8279bd"}
EDGE  = {"HBM-only": "#2b2b2b", "H3": "#2b2b2b", "Naive": "#2b2b2b", "SPLASH": "#3d3570"}
HATCH = {"SPLASH": "///"}
CTXLAB = {131072: "128K", 196608: "192K", 262144: "256K", 393216: "384K",
          524288: "512K", 786432: "768K", 1048576: "1M", 2097152: "2M"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 0, "ytick.major.size": 2.5, "hatch.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


def compute(slo, ctxs):
    best = defaultdict(lambda: None)
    for r in ROWS:
        t = float(r["tpot_p50_ms"])
        if t > slo:
            continue
        thr = int(r["batch"]) * 1000.0 / t / int(r["tp"])
        k = (r["model"], int(r["context_length"]), r["baseline"])
        if best[k] is None or thr > best[k]:
            best[k] = thr
    work = []
    for mk, md in MODELS:
        for c in ctxs:
            sp = best[(mk, c, "Sparse")]
            vals = {sd: (best[(mk, c, sk)] / sp if (best[(mk, c, sk)] is not None and sp) else None)
                    for sk, sd in SYS}
            work.append(dict(model=md, ctx=CTXLAB[c], vals=vals))
    return work


def render(work, slo, fname):
    NC = len(work) // len(MODELS)                     # contexts per model
    centers = []; x = 0.0
    for m in range(len(MODELS)):
        for c in range(NC):
            centers.append(x); x += 1.0
        x += 1.15                                     # inter-model gap
    centers = np.array(centers)

    bw = 0.205
    fig, ax = plt.subplots(figsize=(3.4, 1.2))
    for i, (_, s) in enumerate(SYS):
        xs = centers + (i - 1.5) * bw
        ys = [(w["vals"][s] if w["vals"][s] is not None else 0.0) for w in work]
        ax.bar(xs, ys, bw, color=COL[s], edgecolor=EDGE[s], linewidth=0.5,
               hatch=HATCH.get(s), zorder=3, label=DISP[s])
        for w, xx in zip(work, xs):
            if w["vals"][s] is None:
                ax.plot(xx, 0.05, marker="x", color="#d21f1f", ms=4.2, mew=1.25, zorder=6)

    ax.set_ylim(0, 1.16); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Normalized\nGoodput", fontsize=8.5, linespacing=0.95)
    ax.grid(True, axis="y", which="major", ls=(0, (4, 3)), lw=0.5, color="#cfcfcf", zorder=0)
    ax.set_yticks([0.25, 0.75], minor=True)
    ax.grid(True, axis="y", which="minor", ls=(0, (1, 4)), lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    for m in range(1, len(MODELS)):                   # faint block separators
        xs = (centers[m * NC - 1] + centers[m * NC]) / 2
        ax.axvline(xs, color="#dddddd", lw=0.6, zorder=1)

    ax.set_xticks(centers)
    ax.set_xticklabels([w["ctx"] for w in work], fontsize=3.6)
    ax.tick_params(axis="x", pad=1.2)
    for m, (_, name) in enumerate(MODELS):
        xc = (centers[m * NC] + centers[m * NC + NC - 1]) / 2
        ax.text(xc, -0.235, name, ha="center", va="top", fontsize=5.4,
                transform=ax.get_xaxis_transform())

    # SLO tag (top-left, inside)
    ax.text(0.005, 1.11, f"SLO = {slo} ms", transform=ax.transAxes, fontsize=7.5,
            style="italic", color="#444", ha="left", va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False,
              fontsize=5.2, handlelength=0.8, handleheight=1.0, handletextpad=0.3, columnspacing=0.7)

    fig.subplots_adjust(left=0.093, right=0.997, top=0.80, bottom=0.30)
    for e in ("png", "pdf"):
        fig.savefig(f"results/plots/{fname}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname} (SLO={slo}, {len(work)} workloads)")


render(compute(50, [131072, 196608, 262144, 393216]), 50, "result_goodput_slo50")
render(compute(100, [524288, 786432, 1048576, 2097152]), 100, "result_goodput_slo100")


def _abs_goodput(slo, ctxs):
    """Best per-GPU goodput (tok/s) per (model, ctx, baseline) under the SLO."""
    from collections import defaultdict
    b = defaultdict(lambda: None)
    for r in ROWS:
        t = float(r["tpot_p50_ms"])
        if t > slo:
            continue
        thr = int(r["batch"]) * 1000.0 / t / int(r["tp"])
        k = (r["model"], int(r["context_length"]), r["baseline"])
        if b[k] is None or thr > b[k]:
            b[k] = thr
    return b


def render_combined(fname):
    """Two panels, SLO 50 ms (top) + SLO 100 ms (bottom). Absolute per-GPU goodput
    on a LOG y-axis (keeps SPLASH bars, which vary rather than pinning at 1.0). All
    four systems; x marks an SLO-infeasible point."""
    short_ctx = [131072, 196608, 262144, 393216]
    long_ctx = [524288, 786432, 1048576, 2097152]
    panels = [(_abs_goodput(50, short_ctx), short_ctx, 50),
              (_abs_goodput(100, long_ctx), long_ctx, 100)]
    NCc = 4
    cen = []; x = 0.0
    for m in range(len(MODELS)):
        for _ in range(NCc):
            cen.append(x); x += 1.0
        x += 1.15
    cen = np.array(cen)
    ymax = max(v for bd, _, _ in panels for v in bd.values() if v is not None)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7))
    for ax, (bd, ctxs, slo) in zip(axes, panels):
        for i, (sk, sd) in enumerate(SYS):
            xs = cen + (i - 1.5) * bw_g()
            gi = 0
            for mk, _ in MODELS:
                for c in ctxs:
                    v = bd[(mk, c, sk)]; xx = xs[gi]; gi += 1
                    if v is None:
                        ax.plot(xx, 1.4, marker="x", color="#d21f1f", ms=2.3, mew=0.8, zorder=6)
                    else:
                        ax.bar(xx, v, bw_g(), color=COL[sd], edgecolor=EDGE[sd],
                               linewidth=0.4, hatch=HATCH.get(sd), zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1, ymax * 1.6); ax.set_yticks([1, 10, 100, 1000])
        ax.tick_params(labelsize=6.5)
        ax.set_ylabel("Goodput\n(tok/s/GPU)", fontsize=7.0, linespacing=0.9)
        ax.grid(True, axis="y", which="major", ls=(0, (4, 3)), lw=0.4, color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True); ax.set_xlim(cen[0] - 0.65, cen[-1] + 0.65)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for m in range(1, len(MODELS)):
            ax.axvline((cen[m * NCc - 1] + cen[m * NCc]) / 2, color="#dddddd", lw=0.5, zorder=1)
        ax.set_xticks(cen)
        ax.set_xticklabels([CTXLAB[c] for _ in MODELS for c in ctxs], fontsize=5.6, rotation=90)
        ax.tick_params(axis="x", pad=1.0)
        ax.text(0.01, 0.96, f"SLO {slo} ms", transform=ax.transAxes, fontsize=6.5,
                style="italic", color="#555", ha="left", va="top")
    for m, (_, name) in enumerate(MODELS):
        xc = (cen[m * NCc] + cen[m * NCc + NCc - 1]) / 2
        axes[1].text(xc, -0.72, name, ha="center", va="top", fontsize=6.8,
                     transform=axes[1].get_xaxis_transform())
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[sd], ec=EDGE[sd], lw=0.4,
                             hatch=HATCH.get(sd)) for _, sd in SYS]
    fig.legend(handles, [DISP[sd] for _, sd in SYS], loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=6.2,
               handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
    fig.subplots_adjust(left=0.155, right=0.99, top=0.9, bottom=0.2, hspace=0.5)
    for e in ("png", "pdf"):
        fig.savefig(f"results/plots/{fname}.{e}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def bw_g():
    return 0.205


render_combined("result_goodput_combined")
