#!/usr/bin/env python3
"""Flagship style check: throughput speedup across model x device x context, one clean heatmap."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

CSV = "paper_results.csv"; OUT = "slides_figures"; os.makedirs(OUT, exist_ok=True)

MODELS = [("microsoft/phi-2","Phi-2"),("mistralai/Mistral-7B-v0.1","Mistral-7B"),
          ("meta-llama/Meta-Llama-3-8B","Llama-3-8B"),("mistralai/Mixtral-8x7B-v0.1","Mixtral-8x7B"),
          ("deepseek-ai/deepseek-llm-67b-chat","DeepSeek-67B"),("meta-llama/Meta-Llama-3-70B","Llama-3-70B"),
          ("Qwen/Qwen-72B","Qwen-72B"),("Qwen/Qwen2-72B","Qwen2-72B")]
DEVS = [("a40","A40"),("a100","A100"),("h100","H100"),("h200","H200")]

def fnum(x):
    try: return float(x)
    except: return None
ROWS=[]
for r in csv.DictReader(open(CSV)):
    if "FAILED" in str(r.get("note","")) or not r.get("tpot_p50"): continue
    if r["baseline"] not in ("Sparse","Dense"): continue
    t=fnum(r["throughput_tps"])
    if t is None: continue
    ROWS.append(dict(model=r["model"],dev=r["device"],b=int(r["batch"]),c=int(r["ctx"]),
                     bl="SpaKV" if r["baseline"]=="Sparse" else "H3",tps=t))
IDX={(x["model"],x["dev"],x["b"],x["c"],x["bl"]):x for x in ROWS}
CTXS=sorted({x["c"] for x in ROWS})
def cl(c): return f"{c//1024}K" if c<1024*1024 else f"{c//1024//1024}M"

def speedup(m,d,c):
    bh={x["b"] for x in ROWS if x["model"]==m and x["dev"]==d and x["c"]==c and x["bl"]=="H3"}
    bs={x["b"] for x in ROWS if x["model"]==m and x["dev"]==d and x["c"]==c and x["bl"]=="SpaKV"}
    common=bh&bs
    if not common: return None
    b=max(common); h=IDX[(m,d,b,c,"H3")]; s=IDX[(m,d,b,c,"SpaKV")]
    return s["tps"]/h["tps"] if h["tps"]>0 else None

rows=[(m,short,d,dshort) for m,short in MODELS for d,dshort in DEVS]
M=np.full((len(rows),len(CTXS)),np.nan)
for i,(m,_,d,_) in enumerate(rows):
    for j,c in enumerate(CTXS):
        v=speedup(m,d,c)
        if v is not None: M[i,j]=v

plt.rcParams.update({"font.family":"sans-serif","font.size":11,
    "figure.dpi":200,"savefig.dpi":200,"savefig.bbox":"tight"})
fig,ax=plt.subplots(figsize=(8.6,11))
fig.subplots_adjust(left=0.26,right=0.9,top=0.95,bottom=0.07)
cmap=plt.cm.YlOrRd.copy(); cmap.set_bad("#f0f0f0")
norm=Normalize(vmin=1.0,vmax=8.0)
im=ax.imshow(M,aspect="auto",cmap=cmap,norm=norm)

# cell numbers only for wins (declutter)
for i in range(len(rows)):
    for j in range(len(CTXS)):
        v=M[i,j]
        if not np.isnan(v) and v>=1.1:
            ax.text(j,i,f"{v:.0f}" if v>=10 else f"{v:.1f}",ha="center",va="center",
                    fontsize=7.5,color="white" if v>4.5 else "#222",fontweight="medium")

ax.set_xticks(range(len(CTXS))); ax.set_xticklabels([cl(c) for c in CTXS],fontsize=10)
ax.set_xlabel("Context length",fontsize=12,labelpad=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([d for _,_,_,d in rows],fontsize=8)
ax.tick_params(length=0)
for s in ["top","right","left","bottom"]: ax.spines[s].set_visible(False)
# white gridlines between cells
ax.set_xticks(np.arange(-.5,len(CTXS),1),minor=True)
ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0)

# model group labels + separators (every 4 rows)
for k,(m,short) in enumerate(MODELS):
    c0=k*len(DEVS); c1=c0+len(DEVS)-1
    ax.text(-1.9,(c0+c1)/2,short,ha="right",va="center",fontsize=11,fontweight="bold")
    if k>0: ax.axhline(c0-0.5,color="0.35",lw=1.2)

cbar=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.03,extend="max")
cbar.set_label("Throughput speedup  (SpaKV / dense HBF)",fontsize=11)
cbar.ax.tick_params(labelsize=9)
ax.set_title("Throughput speedup across all models, GPUs, and context lengths",
             fontsize=13,fontweight="bold",pad=12)
fig.savefig(os.path.join(OUT,"speedup_grid.png"))
print("wrote",os.path.join(OUT,"speedup_grid.png"))
