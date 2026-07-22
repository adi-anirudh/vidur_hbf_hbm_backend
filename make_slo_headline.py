#!/usr/bin/env python3
"""Headline perf figure from TP-swept data (full_sweep_tp.csv): sustainable
throughput/GPU under SLO where EACH system uses the (batch,TP) that meets the SLO
and maximizes throughput/GPU. Baselines that need more GPUs (higher TP) show their
real, lower per-GPU throughput. HPCA grouped bars, H3 vs SPLASH, long contexts."""
import csv, math
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CSV="results/full_sweep_tp.csv"; OUT="results/plots"; SLO=100
C_H3,C_SP="#c44e52","#4c72b0"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
 "font.size":8,"axes.labelsize":8.5,"legend.fontsize":7.5,"ytick.labelsize":7,"axes.linewidth":0.7,
 "grid.alpha":0.3,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42})
rows=[r for r in csv.DictReader(open(CSV)) if r["status"]=="OK"]
def f(x):
    try:return float(x)
    except:return None
short=lambda m:(m.split("/")[-1].replace("-Instruct","").replace("-v0.1","").replace("Meta-","").replace("-chat","").replace("-A22B","").replace("-A35B","").replace("deepseek-llm-67b","DeepSeek-67B"))
# index: (model,ctx,baseline) -> list of (batch,tp,tpot)
from collections import defaultdict
D=defaultdict(list)
for r in rows:
    t=f(r["tpot_p50_ms"]); tp=int(r["tp"]) if r["tp"] else 0
    if t and tp: D[(short(r["model"]),int(r["context_length"]),r["baseline"])].append((int(r["batch"]),tp,t))
def slo_thru(m,c,base):
    cand=[b*1000.0/t/tp for (b,tp,t) in D.get((m,c,base),[]) if t<=SLO]
    return max(cand) if cand else 0.0
def min_tp_slo(m,c,base):  # min GPUs to meet SLO (at any batch)
    tps=[tp for (b,tp,t) in D.get((m,c,base),[]) if t<=SLO]
    return min(tps) if tps else None
CTX=[131072,262144,524288,1048576]; clab={131072:"128K",262144:"256K",524288:"512K",1048576:"1M"}
models=sorted({short(r["model"]) for r in rows}, key=lambda m:-max(slo_thru(m,c,"Sparse") for c in CTX))
models=[m for m in models if any(slo_thru(m,c,"Sparse")>0 for c in CTX) and m!="Llama-3.1-405B"]
fig,ax=plt.subplots(figsize=(9.8,3.4)); bw=0.4; intra=0.12; mgap=0.98; x=0.0; ct=[]; cl=[]
for gi,m in enumerate(models):
    x0=x-intra*0.6
    for c in CTX:
        vs=[]
        for base,off,col in (("Dense",0,C_H3),("Sparse",bw,C_SP)):
            v=slo_thru(m,c,base); vs.append(v); ax.bar(x+off,max(v,1e-9),bw,color=col,edgecolor="k",lw=0.3,zorder=3)
        if max(vs)<=0:   # does not fit in memory at any batch/TP -> OOM
            ax.text(x+bw/2,1.25,"OOM",rotation=90,ha="center",va="bottom",fontsize=4.6,color="0.4",zorder=4)
        ct.append(x+bw/2); cl.append(clab[c]); x+=2*bw+intra
    x1=x-intra+intra*0.6
    if gi%2==1: ax.axvspan(x0,x1,color="0.93",zorder=0)          # alternating band per model
    ax.text((x0+x1)/2+0.25,-0.34,m,transform=ax.get_xaxis_transform(),ha="right",va="top",
            fontsize=7.0,rotation=35,rotation_mode="anchor")
    x+=mgap
ax.set_yscale("log"); ax.set_ylim(1,600); ax.set_xticks(ct); ax.set_xticklabels(cl,rotation=90,fontsize=5.6)
ax.set_ylabel("Throughput / GPU (tokens/s)\nat 100 ms SLO"); ax.set_xlim(-0.6,x-mgap+0.6); ax.grid(True,axis="y",which="major",zorder=0)
H=[plt.Rectangle((0,0),1,1,color=C_H3,ec="k",lw=.3),plt.Rectangle((0,0),1,1,color=C_SP,ec="k",lw=.3)]
ax.legend(H,["H3 (dense)","SPLASH"],ncol=2,loc="upper right")
fig.subplots_adjust(bottom=0.46)
for e in("png","pdf"): fig.savefig(f"{OUT}/H_slo_throughput.{e}")
# headline gains (only where H3 feasible under SLO)
g=[]
for m in models:
    for c in CTX:
        h,s=slo_thru(m,c,"Dense"),slo_thru(m,c,"Sparse")
        if h>0 and s>0: g.append(s/h)
if g: print(f"SLO throughput/GPU gain (H3 feasible): geomean {math.exp(sum(map(math.log,g))/len(g)):.1f}x  max {max(g):.1f}x  n={len(g)}")
print("wrote H_slo_throughput")
