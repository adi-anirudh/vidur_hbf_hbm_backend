#!/usr/bin/env python3
"""Figure 1 (motivation): total memory (weights + KV cache) vs context length,
against the single-GPU HBM capacity range. Output: memory_gap.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.axisbelow": True,
})

# context grid (tokens)
CTX = np.array([8e3,16e3,32e3,64e3,128e3,256e3,512e3,1e6,2e6,4e6,10e6])

def kvb(layers, kv_heads, head_dim, dt=2):  # bytes/token (K+V, FP16)
    return 2 * layers * kv_heads * head_dim * dt

# (label, weights_GB, kv_bytes/token, color, marker)
MODELS = [
    ("Llama-3.1 405B",        810, kvb(126, 8, 128), "#1f3b73", "o"),
    ("DeepSeek-V3 671B (MLA)", 671, 61*576*2,         "#2e8b57", "D"),  # MLA: compressed latent
    ("Mistral-Large 123B",    246, kvb(88, 8, 128),  "#1aa3a3", "s"),
    ("Llama-3.1 70B",         140, kvb(80, 8, 128),  "#c0392b", "^"),
    ("Qwen2.5 14B",            28, kvb(48, 8, 128),  "#e0a030", "v"),
    ("Llama-3.1 8B",           16, kvb(32, 8, 128),  "#8e44ad", "<"),
]
# single-GPU HBM capacities (GB)
CAPS = [("A100", 40), ("H100", 80), ("H200", 141), ("B200", 192), ("GB300", 288)]
CAP_MAX = CAPS[-1][1]

fig, ax = plt.subplots(figsize=(8.2, 5.4))
xmin, xmax = CTX[0], CTX[-1]

# shaded HBM band + "beyond" region
ax.axhspan(CAPS[0][1], CAP_MAX, color="#dfe7f0", alpha=0.9, zorder=0)
ax.axhspan(CAP_MAX, 1e5, color="#f7ece4", alpha=0.7, zorder=0)
for name, cap in CAPS:
    ax.axhline(cap, color="0.62", ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.text(xmax*1.04, cap, f"{name} {cap}", va="center", ha="left",
            fontsize=8, color="0.35", clip_on=False)
ax.text(xmin*1.15, CAP_MAX*0.80, "single-GPU HBM range",
        fontsize=8.5, color="#4a5a72", style="italic", va="top")
ax.text(xmin*1.15, 6500, "beyond largest single-GPU HBM\n(multi-GPU or capacity tier required)",
        fontsize=9, color="#9c5a2c", style="italic", va="top", ha="left")

for label, w, kvbt, col, mk in MODELS:
    mem = w + kvbt * CTX / 1e9
    ax.plot(CTX, mem, color=col, marker=mk, ms=4.5, lw=2, label=label, zorder=4)
    # crossover marker: first context exceeding the largest HBM
    cross = np.where(mem > CAP_MAX)[0]
    if len(cross):
        i = cross[0]
        ax.plot(CTX[i], mem[i], "o", mfc="white", mec=col, mew=1.6, ms=8, zorder=5)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(xmin*0.9, xmax*1.02); ax.set_ylim(10, 1e4)
ticks = [8e3,32e3,128e3,512e3,2e6,10e6]
ax.set_xticks(ticks)
ax.set_xticklabels(["8K","32K","128K","512K","2M","10M"])
ax.set_yticks([10,100,1000,10000])
ax.set_yticklabels(["10 GB","100 GB","1 TB","10 TB"])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Total memory: weights + KV cache")
ax.grid(True, which="major", color="0.9", lw=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
          columnspacing=1.8, handletextpad=0.5)
fig.tight_layout()
fig.savefig("figures/memory_gap.png")
print("wrote figures/memory_gap.png")
