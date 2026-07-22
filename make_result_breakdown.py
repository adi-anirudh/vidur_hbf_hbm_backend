#!/usr/bin/env python3
"""Normalized GPU decode-step breakdown for ALL models at one operating point
(1M ctx, batch 16). Single-column, thin bars, compact. Per model: 3 stacked bars
(H3/Naive/SPLASH) normalized to that model's SPLASH = 1.0. Only the KV-read (blue)
segment changes; the fixed compute floor is identical, so SPLASH's shorter step comes
from collapsing KV read. Reads results/breakdown_all.csv."""
import csv
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

R = {(r["model"], r["system"]): r for r in csv.DictReader(open("results/breakdown_all.csv"))}
MODELS = ["Llama-3-8B", "Mixtral-8×7B", "DeepSeek-67B", "Llama-3-70B", "Qwen3-235B"]
SYSU = ["H3", "Naive", "SPLASH"]
SINIT = {"H3": "H3", "Naive": "NS", "SPLASH": "SP"}   # NS = Naive Sparse, SP = SPLASH
SEG = [("Attention", "#4c72b0", lambda d: d["kv_read"] + d["attn_proj"]),
       ("FFN", "#55a868", lambda d: d["mlp"]),
       ("Communication", "#c44e52", lambda d: d["comm"]),
       ("Norm + resid.", "#dd8452", lambda d: d["norms"] + d["kv_write"])]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 0, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

def val(md, sy, fn):
    return float(fn({k: float(R[(md, sy)][k]) for k in
                     ("kv_read", "mlp", "comm", "attn_proj", "norms", "kv_write")}))

# layout: 5 model groups, 3 thin bars each
centers, x = [], 0.0
for _ in MODELS:
    for _ in SYSU:
        centers.append(x); x += 1.0
    x += 0.7
centers = np.array(centers)
BW = 0.72

fig, ax = plt.subplots(figsize=(3.4, 1.62))
ymax = 0.0
for gi, md in enumerate(MODELS):
    nrm = float(R[(md, "SPLASH")]["model_time_ms"])
    for si, sy in enumerate(SYSU):
        xi = centers[gi * 3 + si]
        bottom = 0.0
        for _, col, fn in SEG:
            h = val(md, sy, fn) / nrm
            ax.bar(xi, h, BW, bottom=bottom, color=col, linewidth=0, zorder=3)
            bottom += h
        ymax = max(ymax, bottom)

ax.axhline(1.0, ls="-", lw=0.6, color="#8279bd", zorder=1)
ax.text(0.015, 0.95, "1M ctx  ·  batch 16   (NS = Naive Sparse, SP = SPLASH)",
        transform=ax.transAxes, fontsize=5.6, style="italic", color="#555", ha="left", va="top")
ax.set_ylim(0, ymax * 1.06); ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel("Norm. decode-step", fontsize=7.6)
ax.grid(True, axis="y", ls=(0, (4, 3)), lw=0.5, color="#d7d7d7", zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(centers[0] - 0.7, centers[-1] + 0.7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax.set_xticks(centers)
ax.set_xticklabels([SINIT[s] for _ in MODELS for s in SYSU], fontsize=5.0)
ax.tick_params(axis="x", pad=1.0)
for gi, md in enumerate(MODELS):
    xc = centers[gi * 3 + 1]
    ax.text(xc, -0.235, md, ha="center", va="top", fontsize=6.0,
            transform=ax.get_xaxis_transform())

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in SEG]
ax.legend(handles, [l for l, _, _ in SEG], loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, frameon=False, fontsize=5.9, handlelength=0.9, handleheight=0.9,
          handletextpad=0.3, columnspacing=0.7, labelspacing=0.25)
fig.subplots_adjust(left=0.135, right=0.99, top=0.79, bottom=0.20)
for e in ("png", "pdf"):
    fig.savefig(f"results/plots/result_breakdown.{e}")
print("wrote result_breakdown (all models)")
