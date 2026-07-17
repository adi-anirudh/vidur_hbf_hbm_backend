#!/usr/bin/env python3
"""
Generate the full baseline figure set for several loads (concurrent sessions S),
each into paper_eval/figures/S<load>/.  Model primitives come from eval_baselines.

  figures/
    capacity_batch-vs-gpus_B1-B2_fp16.png   (load-independent, top level)
    capacity_batch-vs-gpus_B1-B2_fp8.png
    S128/ S256/ S512/ S1024/
        cost-latency_*.png  cost-throughput_*.png  cost-at-slo_*.png
        throughput-per-gpu_*.png  disagg-sweep_A-vs-M_*.png
"""
import importlib.util, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "eval_baselines.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF   = 1024.0
EFF   = ev.CLUSTER_EFF
FIGRT = os.path.join(os.path.dirname(__file__), "..", "figures")
COLS  = dict(B1="#c0392b", B2="#2471a3", B3="#27ae60", B4="#8e44ad", Ours="#e67e22")
LAB   = dict(B1="B1 HBM colocated", B2="B2 HBF colocated", B3="B3 HBF disagg (serial)",
             B4="B4 HBF disagg (overlap)", Ours=f"Ours disagg+overlap+clustering ({int(EFF*100)}% experts)")

def frontier(pts):
    pts = sorted([p for p in pts if p], key=lambda p: (p["gpus"], p["tbt"]))
    out, best = [], float("inf")
    for p in pts:
        if p["tbt"] < best - 1e-9: out.append(p); best = p["tbt"]
    return out

def build(S):
    def b1(b):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBM_CAP)
        if mn is None or N<mn: return None
        return dict(gpus=N, tbt=ev.T_attn(b,ev.BW_HBM)+ev.T_moe_colocated(S,N,ev.BW_HBM))
    def b2(b):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBF_CAP)
        if mn is None or N<mn: return None
        return dict(gpus=N, tbt=ev.T_attn(b,HBF)+ev.T_moe_colocated(S,N,HBF))
    def dis(A,M,ov,eff):
        b_a=math.ceil(S/A)
        if b_a>154: return None
        ta=ev.T_attn(b_a,HBF); tm=ev.T_moe_disagg(S,M,HBF,eff)
        return dict(gpus=A+M, tbt=max(ta,tm) if ov else ta+tm)
    Amax=min(S,400)
    return {
      "B1":   frontier([b1(b) for b in range(1,12)]),
      "B2":   frontier([b2(b) for b in range(1,154)]),
      "B3":   frontier([dis(A,M,False,1.0) for A in range(1,Amax+1) for M in range(1,40)]),
      "B4":   frontier([dis(A,M,True ,1.0) for A in range(1,Amax+1) for M in range(1,40)]),
      "Ours": frontier([dis(A,M,True ,EFF) for A in range(1,Amax+1) for M in range(1,40)]),
    }

def fig_cost(S, fr, outdir, ykey, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(9,6))
    for k in ["B1","B2","B3","B4","Ours"]:
        f=sorted(fr[k], key=lambda p:p["gpus"])
        y=[(S*1000.0/p["tbt"] if ykey=="tps" else p["tbt"]) for p in f]
        m={"B1":"o","B2":"s","B3":"^","B4":"D","Ours":"*"}[k]
        ax.plot([p["gpus"] for p in f], y, m+"-", color=COLS[k], label=LAB[k], ms=5)
    ax.set_xlabel(f"total GPUs to serve {S} sessions  (cost →)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.set_xscale("log"); ax.grid(alpha=0.3,which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir,fname),dpi=130); plt.close(fig)

def fig_bars(S, fr, outdir):
    order=["B1","B2","B3","B4","Ours"]; x=np.arange(5); cols=[COLS[k] for k in order]
    def mingpu(f,bud): ok=[p["gpus"] for p in f if p["tbt"]<=bud]; return min(ok) if ok else None
    def peaktps(f): return max(S*1000/(p["tbt"]*p["gpus"]) for p in f)
    g100=[mingpu(fr[k],100) for k in order]; g200=[mingpu(fr[k],200) for k in order]
    peak=[peaktps(fr[k]) for k in order]
    # cost @ SLO
    fig,ax=plt.subplots(figsize=(9,5.5)); w=0.38
    ax.bar(x-w/2,[g or 0 for g in g100],w,color=cols,alpha=0.55,hatch="//",label="TBT ≤ 100 ms")
    ax.bar(x+w/2,[g or 0 for g in g200],w,color=cols,label="TBT ≤ 200 ms")
    for i in range(5):
        ax.text(i-w/2,(g100[i] or 0)+0.5,g100[i] if g100[i] else "n/a",ha="center",fontsize=8)
        ax.text(i+w/2,(g200[i] or 0)+0.5,g200[i] if g200[i] else "n/a",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel(f"GPUs to serve {S} sessions")
    ax.set_title(f"Cost at latency budget — serve {S}×128K decode\nDeepSeek-V3·fp8·HBF=1024 (lower=cheaper)")
    ax.legend(); ax.grid(alpha=0.3,axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(outdir,f"cost-at-slo_serve-{S}x128k_fp8_hbfbw-1024.png"),dpi=130); plt.close(fig)
    # peak tok/s/GPU
    fig,ax=plt.subplots(figsize=(9,5.5)); ax.bar(x,peak,color=cols)
    for i in range(5): ax.text(i,peak[i]+2,f"{peak[i]:.0f}",ha="center",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0,230)
    ax.set_ylabel("peak decode throughput per GPU (tok/s)")
    ax.set_title(f"Peak throughput/GPU — serve {S}×128K decode\nDeepSeek-V3·fp8·128K·HBF=1024")
    ax.grid(alpha=0.3,axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(outdir,f"throughput-per-gpu_optimal_serve-{S}_fp8_hbfbw-1024.png"),dpi=130); plt.close(fig)
    return g100,g200,peak

# capacity (load-independent) — write once at top level
def fig_capacity():
    for kv,tag in [(9.2,"fp16"),(ev.KV_SESS_GB,"fp8")]:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for cap,c,m,lab in [(ev.HBM_CAP,"#c0392b","o","B1 HBM (72 GB)"),(ev.HBF_CAP,"#2471a3","s","B2 HBF (722 GB)")]:
            xs,ys=[],[]
            for b in range(1,90):
                d=cap-ev.R_GB-kv*b
                if d>0: xs.append(b); ys.append(math.ceil(ev.EXPERTS_GB/d))
            ax.plot(xs,ys,m+"-",color=c,lw=2,ms=4,label=lab)
        ax.set_xlabel("batch per GPU b"); ax.set_ylabel("GPUs to fit N")
        ax.set_title(f"Capacity: batch vs GPUs to fit ({tag})"); ax.set_ylim(0,90); ax.set_xlim(0,90)
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(FIGRT,f"capacity_batch-vs-gpus_B1-B2_{tag}.png"),dpi=130); plt.close(fig)

if __name__=="__main__":
    fig_capacity()
    print(f"{'S':>5} {'sys':>5} {'G@100':>6} {'G@200':>6} {'peakTPS/GPU':>11}")
    for S in [128,256,512,1024]:
        outdir=os.path.join(FIGRT,f"S{S}"); os.makedirs(outdir,exist_ok=True)
        fr=build(S)
        fig_cost(S,fr,outdir,"tbt","per-token latency TBT (ms) (↓ better)",
                 f"Cost vs latency — serve {S}×128K decode\nDeepSeek-V3·fp8·HBF=1024",
                 f"cost-latency_serve-{S}x128k_fp8_hbfbw-1024.png")
        fig_cost(S,fr,outdir,"tps","aggregate decode throughput (tok/s) (↑ better)",
                 f"Cost vs throughput — serve {S}×128K decode\nDeepSeek-V3·fp8·HBF=1024",
                 f"cost-throughput_serve-{S}x128k_fp8_hbfbw-1024.png")
        g100,g200,peak=fig_bars(S,fr,outdir)
        for i,k in enumerate(["B1","B2","B3","B4","Ours"]):
            print(f"{S:>5} {k:>5} {str(g100[i]):>6} {str(g200[i]):>6} {peak[i]:>11.1f}")
    print("\nfigures under paper_eval/figures/S128, S256, S512, S1024")
