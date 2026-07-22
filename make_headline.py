#!/usr/bin/env python3
"""Headline: normalized decode throughput PER GPU (tokens/s / TP) under 100ms SLO,
H3 (dense) vs SPLASH, grouped bars across long-context points. FlashAccel Fig-13 style."""
import csv, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CSV, OUT = "results/full_sweep.csv", "results/plots"
SLO = 100
C_H3, C_SP = "#c44e52", "#4c72b0"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
    "mathtext.fontset":"dejavuserif","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "legend.fontsize":8,"xtick.labelsize":7.5,"ytick.labelsize":8,"axes.linewidth":0.7,
    "grid.alpha":0.35,"grid.linewidth":0.4,"axes.axisbelow":True,"figure.dpi":300,
    "savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42})
short=lambda m:(m.split("/")[-1].replace("-Instruct","").replace("-v0.1","").replace("Meta-","").replace("-chat","").replace("-A22B","").replace("-A35B",""))
rows=[r for r in csv.DictReader(open(CSV)) if r["status"]=="OK"]
tpot={}; tp={}
for r in rows:
    k=(short(r["model"]),int(r["batch"]),int(r["context_length"]),r["baseline"])
    tpot[k]=float(r["tpot_p50_ms"]); tp[k]=int(r["tp"]) if r["tp"] else 1
batches=sorted({int(r["batch"]) for r in rows})
def thru(m,c,base):  # tokens/s/GPU at SLO-max batch = max_b (b*1000/tpot/TP), tpot<=SLO
    cand=[b*1000.0/tpot[(m,b,c,base)]/tp[(m,b,c,base)] for b in batches
          if (m,b,c,base) in tpot and tpot[(m,b,c,base)]<=SLO]
    return max(cand) if cand else 0.0
LONG=[131072,262144,524288,1048576]; clab={131072:"128K",262144:"256K",524288:"512K",1048576:"1M"}
# models that are feasible (SPLASH) at >=128K, ordered by 128K SPLASH throughput
models=sorted({short(r["model"]) for r in rows}, key=lambda m:-thru(m,131072,"Sparse"))
models=[m for m in models if thru(m,131072,"Sparse")>0]
n=len(models); x=np.arange(n); w=0.9/(len(LONG)+1)
fig,ax=plt.subplots(figsize=(11,3.6))
blues=plt.cm.Blues(np.linspace(0.45,0.9,len(LONG)))
# H3 reference bar (128K) — leftmost in each group
h3=[thru(m,131072,"Dense") for m in models]
ax.bar(x-0.45+w/2, h3, w, color=C_H3, edgecolor="k", lw=0.4, hatch="///", label="H3 @128K")
for j,v in enumerate(h3):
    if v==0: ax.text(x[j]-0.45+w/2, 1.2, "SLO\nnot met", ha="center", va="bottom", fontsize=4.6, color=C_H3, rotation=90)
for i,c in enumerate(LONG):
    vals=[thru(m,c,"Sparse") for m in models]
    ax.bar(x-0.45+w*(i+1.5), vals, w, color=blues[i], edgecolor="k", lw=0.3, label=f"SPLASH @{clab[c]}")
ax.set_yscale("log"); ax.set_ylim(1,600)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=32, ha="right", fontsize=7)
ax.set_ylabel("Throughput / GPU (tokens/s)\nunder 100 ms TPOT SLO")
ax.set_title("Sustainable decode throughput per GPU (÷ TP), H3 vs SPLASH, long contexts — Blackwell")
ax.legend(ncol=5, fontsize=7, loc="upper right"); ax.grid(True, axis="y", which="both")
fig.tight_layout()
for e in("png","pdf"): fig.savefig(f"{OUT}/0_headline_throughput_per_gpu.{e}")
print("wrote 0_headline_throughput_per_gpu")
# headline numbers
gains=[thru(m,131072,"Sparse")/thru(m,131072,"Dense") for m in models if thru(m,131072,"Dense")>0]
g=math.exp(sum(math.log(v) for v in gains)/len(gains))
print(f"128K per-GPU gain (H3 feasible): geomean {g:.1f}x, n={len(gains)}, max {max(gains):.1f}x")
print(f"models SPLASH-feasible@128K: {len(models)}; H3-feasible@128K: {sum(1 for v in h3 if v>0)}")
