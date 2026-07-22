#!/usr/bin/env python3
"""TP-normalized decode latency (TPOT x TP) vs context, ALL models, H3 vs SPLASH."""
import csv
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CSV,OUT="results/full_sweep_tp.csv","results/plots"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
 "font.size":9,"axes.titlesize":10,"axes.labelsize":9.5,"legend.fontsize":6.6,"xtick.labelsize":8,
 "ytick.labelsize":8,"axes.linewidth":0.7,"grid.alpha":0.35,"figure.dpi":300,"savefig.dpi":300,
 "savefig.bbox":"tight","pdf.fonttype":42})
rows=[r for r in csv.DictReader(open(CSV)) if r['status']=='OK']
def f(x):
    try:return float(x)
    except:return None
short=lambda m:m.split("/")[-1].replace("-Instruct","").replace("-v0.1","").replace("Meta-","").replace("-chat","").replace("-A22B","").replace("-A35B","")
nt={}   # normalized TPOT = tpot_p50 * tp
for r in rows:
    t=f(r['tpot_p50_ms']); tp=int(r['tp']) if r['tp'] else 1
    if t: nt[(short(r['model']),int(r['batch']),int(r['context_length']),r['baseline'])]=t*tp
ctxs=sorted({int(r['context_length']) for r in rows})
clab=lambda c:f"{c//1024}K" if c<1<<20 else "1M"
models=sorted({short(r['model']) for r in rows}, key=lambda m:nt.get((m,1,ctxs[0],'Dense'),1e9))
cols=plt.cm.turbo(np.linspace(0.03,0.97,len(models)))
mk=["o","s","^","v","D","P","X","*","<",">","h","d","p"]
fig,ax=plt.subplots(figsize=(7.4,4.6))
B=1
for i,(m,c) in enumerate(zip(models,cols)):
    xh=[x for x in ctxs if (m,B,x,'Dense') in nt];  yh=[nt[(m,B,x,'Dense')] for x in xh]
    xs=[x for x in ctxs if (m,B,x,'Sparse') in nt]; ys=[nt[(m,B,x,'Sparse')] for x in xs]
    if xh: ax.plot(xh,yh,'-',marker=mk[i%len(mk)],color=c,lw=1.25,ms=3.4,label=short(m))
    if xs: ax.plot(xs,ys,'--',marker=mk[i%len(mk)],color=c,lw=1.0,ms=3.0,mfc='white')
ax.set_xscale('log',base=2); ax.set_yscale('log')
ax.set_xticks(ctxs); ax.set_xticklabels([clab(c) for c in ctxs])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel(r"TP-normalized decode latency  TPOT$\times$TP  (ms$\cdot$GPU), batch 1")
ax.set_title("Decode latency per token, TP-normalized: H3 (solid) vs SPLASH (dashed) — Blackwell")
h,_=ax.get_legend_handles_labels()
leg1=ax.legend(h,[short(m) for m in models],ncol=2,fontsize=6.4,loc='upper left',title="model (color)",title_fontsize=6.8)
from matplotlib.lines import Line2D
ax.add_artist(leg1)
ax.legend(handles=[Line2D([],[],color='0.3',ls='-',label='H3 (dense)'),
                   Line2D([],[],color='0.3',ls='--',label='SPLASH')],
          loc='lower right',fontsize=7.5)
ax.grid(True,which='both')
fig.tight_layout()
for e in("png","pdf"): fig.savefig(f"{OUT}/9_tpot_normalized_vs_context.{e}")
print("wrote 9_tpot_normalized_vs_context (all %d models, TPOTxTP)"%len(models))
