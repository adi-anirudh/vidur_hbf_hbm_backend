#!/usr/bin/env python3
"""Co-design ablation, headline metric: sustainable throughput per GPU under the
100 ms decode SLO, normalized to the dense-HBF baseline (H3). Llama-3-8B, 128K,
TP=8. Five systems isolate SPLASH's three co-design decisions. Full-key-scoring /
token-granular / global-top-k multipliers are measured on real Llama-3.1-8B
attention (codesign_factors.jsonl). Same style as the headline throughput figure."""
import csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = "results/ablation.csv"; OUT = "results/plots"; SLO = 100; TP = 8
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.7,
    "grid.alpha": 0.3, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42})

D = {}
for r in csv.DictReader(open(CSV)):
    if r["status"] == "OK":
        D.setdefault(r["system"], []).append((int(r["batch"]), float(r["tpot_p50_ms"])))
def slo_thru(s):
    return max((b * 1000.0 / t / TP for b, t in D[s] if t <= SLO), default=0.0)
h3 = slo_thru("Dense")

# (key, publication label, colour) — each baseline names the co-design decision it drops
SYS = [("SPLASH",         "SPLASH",             "#4c72b0"),
       ("PlaneImbalance", "Global\ntop-$k$",    "#937860"),
       ("Dense",          "H3\n(dense)",        "#c44e52"),
       ("FullPageScore",  "Full-key\nscoring",  "#dd8452"),
       ("TokenGranular",  "Token-\ngranular",   "#8172b3")]
vals = [slo_thru(k) / h3 for k, _, _ in SYS]
labs = [l for _, l, _ in SYS]; cols = [c for _, _, c in SYS]

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.bar(range(len(SYS)), vals, 0.68, color=cols, edgecolor="k", lw=0.5, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + max(vals) * 0.015, f"{v:.2f}$\\times$", ha="center", va="bottom", fontsize=9, zorder=4)
ax.axhline(1.0, color="0.35", lw=0.8, ls="--", zorder=2)   # H3 (dense) reference
ax.set_xticks(range(len(SYS))); ax.set_xticklabels(labs)
ax.set_ylabel("Throughput / GPU at 100 ms SLO\n(normalized to H3)")
ax.set_ylim(0, max(vals) * 1.12)
ax.grid(True, axis="y", which="major", zorder=0)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(f"{OUT}/ablation_tpot.{e}")
print("throughput/GPU @100ms SLO, normalized to H3:",
      {k: round(slo_thru(k) / h3, 2) for k, _, _ in SYS})
print("wrote ablation_tpot")
