#!/usr/bin/env python3
"""Per-GPU decode throughput snapshot at the same operating point as the TPOT figure
(1M ctx, batch-32) for the selected models x all baselines. Bar = throughput at mean
TPOT, downward whisker to throughput at p99 TPOT. Higher is better -> SPLASH tallest,
HBM-only infeasible (red x). Style consistent with result_tpot / goodput plots."""
import csv
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

R = {(r["model"], r["system"]): r for r in csv.DictReader(open("results/snapshot.csv"))}
MODELS = ["Llama-3-8B", "Mixtral-8x7B", "DeepSeek-67B", "Llama-3-70B", "Qwen3-235B"]
MDISP = {"Mixtral-8x7B": "Mixtral-8×7B"}
SYS = ["HBM-only", "Dense", "Naive", "Sparse"]
DISP = {"HBM-only": "HBM-only", "Dense": "H3", "Naive": "Naive", "Sparse": "SPLASH (Ours)"}
COL = {"HBM-only": "#9e5e34", "Dense": "#e6913a", "Naive": "#c9a24b", "Sparse": "#8279bd"}
EDGE = {"HBM-only": "#2b2b2b", "Dense": "#2b2b2b", "Naive": "#2b2b2b", "Sparse": "#3d3570"}
HATCH = {"Sparse": "///"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 0, "ytick.major.size": 2.5, "hatch.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

centers = np.arange(len(MODELS), dtype=float)
bw = 0.20
fig, ax = plt.subplots(figsize=(7.0, 2.1))
YMAX = max(float(R[(m, "Sparse")]["thr_mean"]) for m in MODELS if R.get((m, "Sparse"), {}).get("thr_mean"))
YMAX = np.ceil(YMAX / 20) * 20 + 10
for i, s in enumerate(SYS):
    xs = centers + (i - 1.5) * bw
    for x, md in zip(xs, MODELS):
        r = R.get((md, s))
        feas = r and r["feasible"] == "1" and r["thr_mean"]
        if not feas:
            ax.plot(x, YMAX * 0.03, marker="x", color="#d21f1f", ms=4.4, mew=1.3, zorder=6)
            continue
        tm, t99 = float(r["thr_mean"]), float(r["thr_p99"])
        ax.bar(x, tm, bw, color=COL[s], edgecolor=EDGE[s], linewidth=0.5,
               hatch=HATCH.get(s), zorder=3)
        if tm - t99 > 0.5:                              # p99-latency throughput (lower)
            ax.errorbar(x, tm, yerr=[[tm - t99], [0]], ecolor="#333",
                        elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)

ax.set_ylim(0, YMAX)
ax.set_ylabel("Decode throughput\n(tokens/s/GPU)", fontsize=8.5, linespacing=0.95)
ax.set_xticks(centers); ax.set_xticklabels([MDISP.get(m, m) for m in MODELS], fontsize=7.6)
ax.set_xlim(centers[0] - 0.6, centers[-1] + 0.6)
ax.tick_params(axis="x", pad=2)
ax.grid(True, axis="y", ls=(0, (4, 3)), lw=0.5, color="#d7d7d7", zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[s], ec=EDGE[s], lw=0.5, hatch=HATCH.get(s))
           for s in SYS]
handles.append(plt.Line2D([0], [0], marker="x", color="#d21f1f", ls="none", mew=1.3, ms=5))
labs = [DISP[s] for s in SYS] + ["infeasible (OOM)"]
ax.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=False,
          fontsize=7.6, handlelength=1.3, handleheight=1.05, handletextpad=0.4, columnspacing=1.3)
fig.subplots_adjust(left=0.092, right=0.995, top=0.88, bottom=0.10)
for e in ("png", "pdf"):
    fig.savefig(f"results/plots/result_throughput_snapshot.{e}")
print("wrote result_throughput_snapshot")
