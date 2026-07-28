#!/usr/bin/env python3
"""Headline throughput/GPU figures from the corrected B200 sweep (sweep_b200.csv),
at BOTH the 50 ms and 100 ms decode SLO. For each (model, context) each system
uses the (batch, TP) that meets the SLO and maximizes tokens/s per GPU. HPCA
grouped bars, H3 vs SPLASH, 6-model long-context subset, contexts 128K..2M."""
import csv, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict

CSV = "results/sweep_b200_weight_valid.csv"; OUT = "results/plots"
C_H3, C_SP = "#c44e52", "#4c72b0"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "grid.alpha": 0.3, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "OK"]
def f(x):
    try: return float(x)
    except: return None
short = lambda m: (m.split("/")[-1].replace("-Instruct", "").replace("-v0.1", "").replace("Meta-", "")
                   .replace("-A22B", "").replace("-A35B", ""))
CTX = [131072, 196608, 262144, 393216, 524288, 786432, 1048576, 2097152]
CLAB = {131072:"128K",196608:"192K",262144:"256K",393216:"384K",524288:"512K",786432:"768K",1048576:"1M",2097152:"2M"}
D = defaultdict(list)
for r in rows:
    t = f(r["tpot_p50_ms"]); tp = int(r["tp"]) if r["tp"] else 0
    if t and tp: D[(short(r["model"]), int(r["context_length"]), r["baseline"])].append((int(r["batch"]), tp, t))

def slo_thru(m, c, base, slo):
    cand = [b*1000.0/t/tp for (b, tp, t) in D.get((m, c, base), []) if t <= slo]
    return max(cand) if cand else 0.0

models = sorted({short(r["model"]) for r in rows}, key=lambda m: -max(slo_thru(m, c, "Sparse", 100) for c in CTX))

def make(slo):
    ms = [m for m in models if any(slo_thru(m, c, "Sparse", slo) > 0 for c in CTX)]
    fig, ax = plt.subplots(figsize=(9.8, 3.2)); bw = 0.4; intra = 0.12; mgap = 0.98; x = 0.0; ct = []; cl = []
    for gi, m in enumerate(ms):
        x0 = x - intra*0.6
        for c in CTX:
            vs = []
            for base, off, col in (("Dense", 0, C_H3), ("Sparse", bw, C_SP)):
                v = slo_thru(m, c, base, slo); vs.append(v)
                ax.bar(x+off, max(v, 1e-9), bw, color=col, edgecolor="k", lw=0.3, zorder=3)
            if max(vs) <= 0:
                ax.text(x+bw/2, 1.25, "OOM", rotation=90, ha="center", va="bottom", fontsize=4.6, color="0.4", zorder=4)
            ct.append(x+bw/2); cl.append(CLAB[c]); x += 2*bw+intra
        x1 = x-intra+intra*0.6
        if gi % 2 == 1: ax.axvspan(x0, x1, color="0.93", zorder=0)
        ax.text((x0+x1)/2+0.25, -0.34, m, transform=ax.get_xaxis_transform(), ha="right", va="top",
                fontsize=7.0, rotation=35, rotation_mode="anchor")
        x += mgap
    ax.set_yscale("log"); ax.set_ylim(1, ax.get_ylim()[1]); ax.set_xticks(ct); ax.set_xticklabels(cl, rotation=90, fontsize=5.6)
    ax.set_ylabel(f"Throughput / GPU (tokens/s)\nat {slo} ms SLO"); ax.set_xlim(-0.6, x-mgap+0.6)
    ax.grid(True, axis="y", which="major", zorder=0)
    H = [plt.Rectangle((0,0),1,1,color=C_H3,ec="k",lw=.3), plt.Rectangle((0,0),1,1,color=C_SP,ec="k",lw=.3)]
    ax.legend(H, ["H3 (dense)", "SPLASH"], ncol=2, loc="upper right")
    fig.subplots_adjust(bottom=0.42)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/H_slo{slo}_throughput.{e}")
    g = [slo_thru(m,c,"Sparse",slo)/slo_thru(m,c,"Dense",slo) for m in ms for c in CTX
         if slo_thru(m,c,"Dense",slo) > 0 and slo_thru(m,c,"Sparse",slo) > 0]
    gm = math.exp(sum(map(math.log, g))/len(g)) if g else 0
    print(f"  {slo}ms SLO: geomean {gm:.1f}x  max {max(g):.1f}x  n={len(g)}  (wrote H_slo{slo}_throughput)")

for slo in (50, 100): make(slo)
