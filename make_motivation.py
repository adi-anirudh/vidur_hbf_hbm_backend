#!/usr/bin/env python3
"""Two-panel motivation figure, shared academic style (serif/STIX, in-ticks on all
four sides, hairline grid, framed inset legends).
(a) memory-tier read-bandwidth vs capacity landscape -- only HBF has BOTH the
    bandwidth (>=100 GB/s) and the capacity (>=0.34 TB) a 1M-context 70B KV needs.
(b) decode-throughput vs context (Llama-3-70B, B200; user-authored panel):
    SPLASH (sparse) > H3 (dense) > HBM-only, which scales out to n GPUs once the
    KV exceeds a device's HBM. At 1M: SPLASH 5.1x H3, H3 2.2x HBM-only."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12.5,
    "axes.labelsize": 15.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "legend.fontsize": 11,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK = "#111111"
BLUE_A, GREEN, LBLUE, ORANGE, PINK = "#2f6fb0", "#2ca25f", "#74add1", "#f0a838", "#d492bd"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 5.2),
                               gridspec_kw={"width_ratios": [1.0, 1.12]})

# =========================== panel (a): memory-tier landscape ===========================
#  tier          colour  points: (name, cap GB, bw GB/s, dx, dy pt, ha, leader)
TIERS = [
    ("On-chip",    BLUE_A, [("SRAM", 0.05, 10000, 15, 0, "left", False)]),
    ("On-package", GREEN,  [("HBM4", 200, 8000, -11, 13, "right", False),
                            ("HBM3e", 150, 4800, -15, 0, "right", False),
                            ("GDDR7", 32, 1800, -15, 0, "right", False)]),
    ("Off-package", LBLUE, [("LPDDR5X", 64, 77, -15, -8, "right", False),
                            ("DDR5", 128, 48, 0, -17, "center", False)]),
    ("HBF",        ORANGE, [("HBF", 3072, 8000, 0, 0, "center", False)]),
    ("Flash NAND", PINK,   [("NAND SSD", 16000, 14, 0, 20, "center", False)]),
]
axA.add_patch(Rectangle((344, 100), 1e6, 1e6, color="#e0f1e7", zorder=0))
axA.axhline(100, color="#111", ls=(0, (6, 4)), lw=1.3, zorder=1)
axA.axvline(344, color="#111", ls=(0, (6, 3)), lw=2.1, zorder=2)
axA.text(0.04, 122, "min bandwidth (100 GB/s)", fontsize=11.5, color="#333", ha="left", va="bottom", zorder=3)
axA.text(344, 1.03, "1M-context KV: 0.34 TB (Llama-3-70B)", transform=axA.get_xaxis_transform(),
         fontsize=11, color="#111", fontweight="bold", ha="center", va="bottom",
         clip_on=False, zorder=5)
axA.text(3072, 12800, "HBF", fontsize=16, fontweight="bold", color="#8a5a08", ha="center", va="center", zorder=4)
for _, col, pts in TIERS:
    for name, cap, bw, dx, dy, ha, leader in pts:
        axA.plot([cap], [bw], "o", ms=10, color=col, mec="#222", mew=0.9, zorder=6)
        if name == "HBF":
            continue
        axA.annotate(name, xy=(cap, bw), xytext=(dx, dy), textcoords="offset points", fontsize=12,
                     color=INK, ha=ha, va="center", zorder=7,
                     arrowprops=dict(arrowstyle="-", color="#666", lw=0.8) if leader else None)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("Capacity (GB)"); axA.set_ylabel("Read bandwidth (GB/s)")
axA.set_xlim(0.025, 130000); axA.set_ylim(9, 22000)
axA.grid(True, which="major", ls="-", lw=0.5, color="0.88", zorder=0); axA.set_axisbelow(True)
legA = [Line2D([0], [0], marker="o", ls="none", ms=9, color=c, mec="#222", mew=0.8, label=t) for t, c, _ in TIERS]
lgA = axA.legend(handles=legA, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
                 framealpha=1.0, edgecolor="0.7", fancybox=False, borderpad=0.5,
                 handletextpad=0.3, columnspacing=1.5, labelspacing=0.45, fontsize=11.5,
                 title="Memory tiers")
lgA.get_frame().set_linewidth(0.7)
lgA.get_title().set_fontsize(11); lgA.get_title().set_color("black")

# =========================== panel (b): decode throughput (user-authored) ================
BLACK = "#000000"     # SPLASH (ours) -- emphasized
VERM = "#D55E00"      # H3
BLUE = "#0072B2"      # HBM-only

share_x = np.array([4096, 8192, 16384])
share_y = np.array([208, 202, 195])
ctx = np.array([16384, 32768, 65536, 98304, 131072, 196608,
                262144, 393216, 524288, 720896, 1048576])
splash = np.array([195, 190, 185, 180, 174, 167, 160, 150, 140, 125, 108])
h3 = np.array([195, 172, 140, 118, 100, 80, 68, 48, 35, 26, 21])
hbm_x = np.array([16384, 32768, 49152, 65536, 98304, 131072,
                  196608, 262144, 393216, 524288, 786432, 1048576])
hbm_y = np.array([170, 100, 90, 78, 48, 40, 30, 24, 21, 12, 11, 9.5])
gpu_pts = [(1, 16384, 170), (2, 32768, 100), (4, 98304, 48),
           (8, 262144, 24), (16, 1048576, 9.5)]

axB.set_xscale("log", base=2); axB.set_yscale("log")
axB.set_xlim(3900, 1.30e6); axB.set_ylim(8, 235)
axB.grid(True, which="major", ls="-", lw=0.5, color="0.88", zorder=0); axB.set_axisbelow(True)

axB.axvspan(3900, 16384, color="0.925", zorder=0)
LBL = dict(ha="center", va="center", fontsize=16, color="black", fontweight="bold", linespacing=1.15)

axB.plot(share_x, share_y, color=BLACK, lw=1.4, zorder=4)
axB.plot(hbm_x, hbm_y, color=BLUE, lw=1.3, ls=":", marker="^", ms=5,
         mfc="white", mec=BLUE, mew=1.1, zorder=5, label="HBM-only ($n$ GPUs, tensor parallel)")
axB.plot(ctx, h3, color=VERM, lw=1.4, ls="--", marker="s", ms=4.8,
         mfc="white", mec=VERM, mew=1.1, zorder=6, label="H3 (1 GPU, HBM+HBF, dense attention)")
axB.plot(ctx, splash, color=BLACK, lw=1.8, ls="-", marker="o", ms=5.2,
         mfc=BLACK, mec=BLACK, zorder=7, label="SPLASH (1 GPU, HBM+HBF, sparse attention)")

axB.axvline(16384, color="0.4", ls="--", lw=1.0, zorder=2)
axB.text(52000, 23, "KV exceeds\nsingle HBM", **LBL)

gpu_off = {1: (-9, -3), 2: (7, 11), 4: (7, 11), 8: (7, 11), 16: (-10, -1)}
gpu_ha = {1: "right", 2: "left", 4: "left", 8: "left", 16: "right"}
for n, gx, gy in gpu_pts:
    axB.annotate(f"$n{{=}}{n}$", xy=(gx, gy), xytext=gpu_off[n], textcoords="offset points",
                 fontsize=11.5, color=BLUE, fontweight="bold", va="center", ha=gpu_ha[n])

xb = 1.16e6
for (y0, y1, lab, ly) in [(21, 108, r"$5.1\times$", 47.0), (9.5, 21, r"$2.2\times$", 14.1)]:
    axB.annotate("", xy=(xb, y1), xytext=(xb, y0), arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.15"))
    axB.text(8.4e5, ly, lab, fontsize=12.5, color="0.05", va="center", ha="center")
for y in (108, 21, 9.5):
    axB.plot([1.05e6, xb], [y, y], color="0.55", lw=0.6, ls=(0, (1, 2)), zorder=3)

axB.xaxis.set_major_locator(FixedLocator([4096, 16384, 65536, 262144, 1048576]))
axB.xaxis.set_major_formatter(FixedFormatter(["4K", "16K", "64K", "256K", "1M"]))
axB.xaxis.set_minor_locator(NullLocator())
axB.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200]))
axB.yaxis.set_major_formatter(FixedFormatter(["10", "20", "50", "100", "200"]))
axB.yaxis.set_minor_locator(NullLocator())
axB.set_xlabel("Context length (tokens)"); axB.set_ylabel("Normalized throughput (tok/s/GPU)")

h, l = axB.get_legend_handles_labels()
order = [2, 1, 0]
leg = axB.legend([h[i] for i in order], [l[i] for i in order], loc="upper center",
                 bbox_to_anchor=(0.5, -0.17), ncol=1, framealpha=1.0, edgecolor="0.7",
                 fancybox=False, borderpad=0.5, handlelength=2.3, fontsize=11,
                 labelspacing=0.45, title="Llama-3-70B  ·  NVIDIA B200")
leg.get_frame().set_linewidth(0.7)
leg.get_title().set_fontsize(11); leg.get_title().set_color("black")

fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.27, wspace=0.26)
for e in ("png", "pdf"):
    fig.savefig(f"results/plots/motivation_hierarchy.{e}", bbox_inches="tight")
print("wrote combined figure (serif/STIX academic style, panels a+b)")
