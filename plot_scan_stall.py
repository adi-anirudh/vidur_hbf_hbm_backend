#!/usr/bin/env python3
"""Grouped bars: hot-window attention vs selection scan, per context.
Styled to match the HBF BW-vs-sparsity motivation figure (plot_bw_sparsity.py):
serif fonts, earthy palette, dashed grid, boxed top legend.

Message: the near-memory full-K scoring scan grows with context and overtakes the
capacity-capped hot-window attention -> it cannot hide behind it -> the GPU stalls,
and the stall grows with context. Numbers from the HBF predictor's flat SRA branch
(verified byte-faithful). Llama-3.1-8B, one B200 (HBM 192 GB / HBF 3072 GB, both
8 TB/s), TP1, batch 96, contexts within the model's real 128K window."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import _scan_explore as E
import motif_style as ms

MODEL, BATCH, TP = "meta-llama/Meta-Llama-3-8B", 96, 1
CTXS = [16384, 32768, 65536, 131072]
rows = [E.times(MODEL, BATCH, C, TP) for C in CTXS]
t_hot = np.array([r["t_hbm"] for r in rows])
t_scan = np.array([r["t_scan"] for r in rows])
idle = np.array([r["idle"] for r in rows])
HOT = float(np.median(t_hot[t_scan > 0.5]))            # capped hot-window level (~21.5 ms)

# palette uniform with the other motivation figures (motif_style.py)
C_HOT, C_SCAN, C_MARK, C_IDLE, C_INK, C_MEAS = ms.PAL["gold"], ms.PAL["primary"], "#9a9a9a", ms.PAL["rust"], "#333333", "#5a5a5a"
ms.apply()

fig, ax = plt.subplots(figsize=(ms.W_COL, ms.HEIGHT))
ax.set_axisbelow(True)
ax.grid(axis="y", zorder=0, **ms.GRID)

x = np.arange(len(CTXS)); w = 0.4
ax.bar(x - w/2, t_hot, w, color=C_HOT, edgecolor="black", lw=0.5, zorder=3)
ax.bar(x + w/2, t_scan, w, color=C_SCAN, edgecolor="black", lw=0.5, zorder=3)
ax.axhline(HOT, color=C_MARK, ls=(0, (4, 2)), lw=0.9, zorder=4)

YMAX = t_scan.max() * 1.24
for i in range(len(CTXS)):
    ax.text(x[i]-w/2, t_hot[i]+YMAX*0.02, f"{t_hot[i]:.0f}", ha="center", va="bottom",
            fontsize=ms.FS_ANNOT, color=C_INK)
    if idle[i] > 0.5:
        xa = x[i] + w/2 + w*0.58                       # dimension line just right of the scan bar
        ax.annotate("", xy=(xa, t_scan[i]), xytext=(xa, HOT),
                    arrowprops=dict(arrowstyle="|-|", color=C_MEAS, lw=0.8, mutation_scale=3.2))
        # on the tall bars the idle callout REPLACES the value label (avoids overlap)
        ax.text(x[i]+w/2, t_scan[i]+YMAX*0.03, f"+{idle[i]:.0f} ms\nGPU idle", ha="center",
                va="bottom", fontsize=ms.FS_ANNOT, color=C_IDLE, linespacing=0.95)
    else:
        ax.text(x[i]+w/2, t_scan[i]+YMAX*0.02, f"{t_scan[i]:.0f}", ha="center", va="bottom",
                fontsize=ms.FS_ANNOT, color=C_INK)

ax.set_xticks(x); ax.set_xticklabels([f"{c//1024}K" for c in CTXS])
ax.set_xlabel("Context length (tokens per request)")
ax.set_ylabel("Attention time (ms)")
ax.set_ylim(0, YMAX); ax.set_xlim(-0.55, len(CTXS)-0.35)
ax.set_yticks([0, 20, 40, 60, 80])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

handles = [Patch(fc=C_HOT, ec="black", lw=0.5), Patch(fc=C_SCAN, ec="black", lw=0.5)]
labels = ["Hot-window attention (HBM)", "Selection scan (HBF)"]
fig.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=(0.135, 1.0),
           frameon=True, edgecolor="#bfbfbf", framealpha=1.0, handlelength=1.2,
           borderpad=0.5, labelspacing=0.35)

for ext in ("png", "pdf"):
    fig.savefig(f"figures/scan_stall.{ext}", bbox_inches="tight", facecolor="white")
print("wrote figures/scan_stall.png/.pdf")
for c, r in zip(CTXS, rows):
    print(f"  {c//1024:>4}K  hot={r['t_hbm']:5.1f}  scan={r['t_scan']:6.1f}  idle={r['idle']:6.1f} ms")
