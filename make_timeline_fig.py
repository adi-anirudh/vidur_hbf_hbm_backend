#!/usr/bin/env python3
"""Fig 6: decode-step timing comparison of KV-serving paths (schematic Gantt).
Output: fig6_timeline.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({
    "font.size": 9, "savefig.dpi": 300, "figure.dpi": 300, "savefig.bbox": "tight",
})

# block colors
C_FFN = "#8a8a8a"; C_LOAD = "#f2f2f2"; C_ATTN = "#f3d79b"; C_SEL = "#e07b2a"
LANE_H = 0.62

# Each scheme: (tag, name, note, gpu_blocks, mem_label, mem_blocks, crit)
# block = (start, width, color); times are schematic units.
SCHEMES = [
    ("(a)", "Data-offloading (PCIe)", "host streams all KV over slow PCIe — GPU compute gated by the full transfer",
     [(0,16,C_ATTN),(16,7,C_FFN)], "host", [(0,93,C_LOAD)], 93),
    ("(b)", "CPU-based sparse attn.", "selection + attention on CPU; GPU FFN waits until they complete",
     [(0,7,C_FFN),(58,9,C_ATTN)], "CPU", [(0,10,C_SEL),(10,46,C_ATTN)], 67),
    ("(c)", "PNM-based full attn.", "full attention on PNM serializes ahead of the GPU",
     [(46,12,C_ATTN)], "PNM", [(0,44,C_ATTN)], 58),
    ("(d)", "Baseline HBF (dense)", "dense read of the full KV from flash stalls the GPU each step",
     [(0,20,C_ATTN),(20,7,C_FFN)], "HBF", [(0,30,C_LOAD)], 30),
    ("(e)", "SpaKV (ours)", "NMP selects top-k: less KV loaded AND less attention; flash + scoring hidden",
     [(0,16,C_ATTN),(16,7,C_FFN)], "NMP+HBF", [(0,6,C_SEL),(6,6,C_LOAD)], 23),
]

fig, ax = plt.subplots(figsize=(7.0, 5.2))
row_gap = 2.3
y = 0.0
yticks, ylabels = [], []
fastest = min(s[6] for s in SCHEMES)
for tag, name, note, gpu, mlab, mem, crit in reversed(SCHEMES):
    y_mem = y
    y_gpu = y + LANE_H + 0.18
    ax.broken_barh([(s, w) for s, w, _ in gpu], (y_gpu, LANE_H),
                   facecolors=[c for *_, c in gpu], edgecolor="black", linewidth=0.6)
    ax.broken_barh([(s, w) for s, w, _ in mem], (y_mem, LANE_H),
                   facecolors=[c for *_, c in mem], edgecolor="black", linewidth=0.6)
    ax.text(-2, y_gpu + LANE_H/2, "GPU", ha="right", va="center", fontsize=7.5)
    ax.text(-2, y_mem + LANE_H/2, mlab, ha="right", va="center", fontsize=7.5)
    # critical-path marker
    ax.plot([crit, crit], [y_mem - 0.15, y_gpu + LANE_H + 0.15], color="#b00020",
            lw=1.3, ls=(0, (3, 2)), zorder=5)
    ax.text(crit + 1.5, y_gpu + LANE_H/2, f"{crit}",
            color="#b00020", fontsize=7.5, va="center", fontweight="bold")
    ax.text(-13.5, y_gpu + 0.1, f"{tag} {name}", ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.text(0, y_mem - 0.62, note, ha="left", va="center", fontsize=7, style="italic", color="0.35")
    y += row_gap

ax.axvline(fastest, color="#1b7a3d", lw=1.0, ls=":", zorder=1)
ax.set_xlim(-14, 100)
ax.set_ylim(-1.0, y - row_gap + LANE_H + 1.4)
ax.set_xlabel("Decode-step time (schematic) →")
ax.set_yticks([]);
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)

legend = [Patch(fc=C_ATTN, ec="black", label="Attention"),
          Patch(fc=C_FFN, ec="black", label="FFN"),
          Patch(fc=C_LOAD, ec="black", label="Load KV"),
          Patch(fc=C_SEL, ec="black", label="Token selection"),
          plt.Line2D([], [], color="#b00020", ls=(0,(3,2)), lw=1.3, label="critical-path end")]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
          frameon=False, fontsize=8, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout()
fig.savefig("figures/fig6_timeline.png")
print("wrote figures/fig6_timeline.png")
