#!/usr/bin/env python3
"""HPCA two-column grouped-bar headline (Blackwell, H3 vs SPLASH): throughput/GPU
(tokens/s, /TP) under 100 ms SLO. Models=major groups; long contexts within each.
solid=meets SLO, hatched=best-effort (misses SLO). No chart title."""
import csv
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CSV, OUT = "results/full_sweep_tp.csv", "results/plots"
SLO=100; C_H3,C_SP="#c44e52","#4c72b0"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
 "font.size":8,"axes.labelsize":8.5,"legend.fontsize":7.5,"ytick.labelsize":7,
 "axes.linewidth":0.7,"grid.alpha":0.3,"grid.linewidth":0.35,"axes.axisbelow":True,
 "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42})
rows=[r for r in csv.DictReader(open(CSV)) if r["status"]=="OK"]
def f(x):
    try:return float(x)
    except:return None
short=lambda m:(m.split("/")[-1].replace("-Instruct","").replace("-v0.1","").replace("Meta-","").replace("-chat","").replace("-A22B","").replace("-A35B","").replace("deepseek-llm-67b","DeepSeek-67B"))
tpot={};tp={}
for r in rows:
    k=(short(r["model"]),int(r["batch"]),int(r["context_length"]),r["baseline"]); tpot[k]=f(r["tpot_p50_ms"]); tp[k]=int(r["tp"]) if r["tp"] else 1
batches=sorted({int(r["batch"]) for r in rows})
CTX=[131072,524288,1048576]; clab={131072:"128K",524288:"512K",1048576:"1M"}
def gp(m,c,base):
    ok=[b*1000.0/tpot[(m,b,c,base)]/tp[(m,b,c,base)] for b in batches if (m,b,c,base) in tpot and tpot[(m,b,c,base)]<=SLO]
    if ok: return max(ok),True
    if (m,1,c,base) in tpot and tpot[(m,1,c,base)]: return 1000.0/tpot[(m,1,c,base)]/tp[(m,1,c,base)],False
    return 0.0,False
models=sorted({short(r["model"]) for r in rows}, key=lambda m:-max(gp(m,c,"Sparse")[0] for c in CTX))
fig,ax=plt.subplots(figsize=(7.15,2.9))
bw=0.40; intra=0.14; mgap=0.62
x=0.0; ctx_ticks=[]; ctx_lab=[]; centers=[]
for m in models:
    block=[]
    for c in CTX:
        vs=[]
        for base,off,col in (("Dense",0,C_H3),("Sparse",bw,C_SP)):
            v,feas=gp(m,c,base); vs.append(v)
            ax.bar(x+off,max(v,1e-9),bw,color=col,edgecolor="k",lw=0.3,
                   hatch=(None if feas else "////"),alpha=(1.0 if feas else 0.5))
        if max(vs)<=0:   # OOM: does not fit in memory at batch 1
            ax.text(x+bw/2,1.15,"OOM",rotation=90,ha="center",va="bottom",fontsize=5,color="0.35")
        ctx_ticks.append(x+bw/2); ctx_lab.append(clab[c]); block.append(x+bw/2)
        x+=2*bw+intra
    centers.append(float(np.mean(block))); x+=mgap
ax.set_yscale("log"); ax.set_ylim(1,600)
ax.set_xticks(ctx_ticks); ax.set_xticklabels(ctx_lab, rotation=90, fontsize=5.6)
ax.tick_params(axis="x", length=2, pad=1)
ax.set_ylabel("Throughput / GPU (tokens/s)")
ax.set_xlim(-0.4, x-mgap+0.4); ax.grid(True,axis="y",which="major")
for cx,m in zip(centers,models):
    ax.text(cx,-0.30,m,transform=ax.get_xaxis_transform(),ha="right",va="top",fontsize=6.8,rotation=28)
H=[plt.Rectangle((0,0),1,1,color=C_H3,ec="k",lw=.3),plt.Rectangle((0,0),1,1,color=C_SP,ec="k",lw=.3),
   plt.Rectangle((0,0),1,1,fc="0.7",ec="k",lw=.3,hatch="////",alpha=.5)]
ax.legend(H,["H3 (dense)","SPLASH","misses 100 ms SLO"],ncol=3,loc="upper right",framealpha=.92)
fig.subplots_adjust(bottom=0.30)
for e in("png","pdf"): fig.savefig(f"{OUT}/H_throughput_bars.{e}")
print("wrote H_throughput_bars")
