#!/usr/bin/env python3
"""
Final figure set — per load S, per context -> figures/S<S>/.

  B1            HBM colocated EP (experts sharded in HBM, serial)
  B2            HBF colocated EP (experts sharded on flash, serial)
  B3            HBF disaggregated, static-shard experts, NO overlap
  B4            = B3 with attention/MoE micro-batch overlap
  B5            disaggregated + overlap, attention GPUs on HBM (72GB) +
                MoE GPUs on HBF; static shard
  Ours+HBF      disaggregated + overlap, ALL HBF; dynamic expert-aware activation
                (attention KV on HBF)
  Ours+HBM/HBF  = B5 + dynamic expert-aware activation
                (attention KV on HBM, experts on HBF)

Memory bandwidth is 1024 GB/s for BOTH HBM and HBF (system knob). The only
difference HBM vs HBF is CAPACITY: 72 GB vs 722 GB -> attention-GPU KV capacity
(sessions/GPU) differs, which changes how many attention GPUs are needed.

Sharded MoE model: each MoE GPU streams its active-expert slice from flash.
Per-GPU load (busiest GPU):
  static  = fixed round-robin expert->GPU assignment (imbalanced union)
  dynamic = all experts resident, active union re-dealt evenly -> E_unique(S)/M
No pruning (accuracy-preserving).
Figures: cost-latency, cost-throughput, cost-at-slo, throughput-per-gpu.
"""
import importlib.util, math, os
import numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__),"eval_baselines.py"))
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF=1024.0; STRUCT=0.9; SEEDS=6
NE,K,NG,HOT=256,8,16,32
FIGRT=os.path.join(os.path.dirname(__file__),"..","figures")
ORDER=["B1","B2","B4","B5","Ours+HBF","Ours+HBM/HBF"]
COLS={"B1":"#c0392b","B2":"#2471a3","B4":"#8e44ad",
      "B5":"#16a085","Ours+HBF":"#e67e22","Ours+HBM/HBF":"#c2185b"}
MRK ={"B1":"o","B2":"s","B4":"D","B5":"v","Ours+HBF":"*","Ours+HBM/HBF":"P"}
LAB ={"B1":"B1 HBM coloc","B2":"B2 HBF coloc",
      "B4":"B4 HBF disagg static overlap","B5":"B5 HBM-attn + HBF-MoE (static)",
      "Ours+HBF":"Ours+HBF (dyn act, HBF attn)",
      "Ours+HBM/HBF":"Ours+HBM/HBF (dyn act, HBM attn)"}
MS=list(range(1,129))

DEFS=("B1 HBM colocated EP    B2 HBF colocated EP   (experts sharded, serial)\n"
      "B4 HBF disaggregated static-shard + attention/MoE overlap\n"
      "B5 disaggregated+overlap: attention on HBM (72GB) + experts on HBF, static shard\n"
      "Ours+HBF  disagg+overlap, ALL-HBF, dynamic expert-aware activation (attention KV on HBF)\n"
      "Ours+HBM/HBF  = B5 + dynamic expert-aware activation (attention KV on HBM, experts on HBF)")
def caption(fig):
    fig.text(0.5,0.012,DEFS,ha="center",va="bottom",fontsize=7,color="#333",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5",fc="#f5f5f5",ec="#cccccc",lw=0.6))

def gen(S,rng):
    hot=[rng.choice(NE,HOT,replace=False) for _ in range(NG)]; toks=[]
    for _ in range(S):
        g=rng.integers(NG); ch=set()
        while len(ch)<K: ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(NE)))
        toks.append(frozenset(ch))
    return toks

_eu={}
def E_unique(S):
    if S in _eu: return _eu[S]
    vals=[len(set().union(*gen(S,np.random.default_rng(sd)))) for sd in range(SEEDS)]
    r=float(np.mean(vals)); _eu[S]=r; return r

_sm={}
def static_max(S,M):
    key=(S,M)
    if key in _sm: return _sm[key]
    if M<=1: r=E_unique(S)
    else:
        vals=[]
        for sd in range(SEEDS):
            active=set().union(*gen(S,np.random.default_rng(sd)))
            assign=np.random.default_rng(sd+1).integers(0,M,NE)
            vals.append(max(Counter(int(assign[e]) for e in active).values()))
        r=float(np.mean(vals))
    _sm[key]=r; return r

def T_moe_shard(S,M,E_busiest):
    # E_busiest = experts on the BUSIEST MoE GPU (already per-GPU). Each expert is
    # streamed once (44MB); shared expert lives on every GPU. NO extra /M here.
    load=ev.ms(E_busiest*ev.PER_EXPERT + ev.SHARED_BYTES, HBF)
    comp=ev.moe_compute_ms(S)/M
    copies=M*(1.0-(1.0-1.0/M)**K) if M>1 else 1.0
    xfer=ev.ms(2*S*copies*ev.D*ev.BE_ACT, M*ev.LINK_BW)
    return (load+comp)*ev.L_MOE + xfer*ev.L_MOE

def frontier(pts):
    pts=sorted([p for p in pts if p],key=lambda p:(p["gpus"],p["tbt"])); out=[];best=1e18
    for p in pts:
        if p["tbt"]<best-1e-9: out.append(p); best=p["tbt"]
    return out

def build(S):
    cap_sessions=lambda cap_gb: math.floor((cap_gb-ev.R_GB)/ev.KV_SESS_GB)
    def col(b,cap,bw):
        N=math.ceil(S/b); mn=ev.min_N(b,cap)
        if mn is None or N<mn: return None
        return dict(gpus=N,tbt=ev.T_attn(b,bw)+ev.T_moe_colocated(S,N,bw))
    def dis(A,M,ov,mode,attn_cap_gb):
        b_a=math.ceil(S/A)
        if b_a>cap_sessions(attn_cap_gb): return None      # attn-GPU KV capacity
        load = static_max(S,M) if mode=="static" else E_unique(S)/M
        tm=T_moe_shard(S,M,load); ta=ev.T_attn(b_a,HBF)
        return dict(gpus=A+M,tbt=max(ta,tm) if ov else ta+tm)
    Amax=min(S,400)
    bH=cap_sessions(ev.HBF_CAP)+1; bM=cap_sessions(ev.HBM_CAP)+1
    return {
      "B1":frontier([col(b,ev.HBM_CAP,ev.BW_HBM) for b in range(1,max(2,bM+1))]),
      "B2":frontier([col(b,ev.HBF_CAP,HBF) for b in range(1,max(2,bH+1))]),
      "B4":frontier([dis(A,M,True ,"static",ev.HBF_CAP) for A in range(1,Amax+1) for M in MS]),
      "B5":frontier([dis(A,M,True ,"static",ev.HBM_CAP) for A in range(1,Amax+1) for M in MS]),
      "Ours+HBF":frontier([dis(A,M,True,"dyn",ev.HBF_CAP) for A in range(1,Amax+1) for M in MS]),
      "Ours+HBM/HBF":frontier([dis(A,M,True,"dyn",ev.HBM_CAP) for A in range(1,Amax+1) for M in MS]),
    }

def lineplot(S,fr,outdir,ykey,ylabel,fname,title,ck):
    fig,ax=plt.subplots(figsize=(9,6))
    for k in ORDER:
        f=sorted(fr[k],key=lambda p:p["gpus"])
        if not f: continue
        y=[(S*1000.0/p["tbt"] if ykey=="tps" else p["tbt"]) for p in f]
        ax.plot([p["gpus"] for p in f],y,MRK[k]+"-",color=COLS[k],label=LAB[k],ms=5)
    ax.set_xlabel(f"total GPUs to serve {S}×{ck}K (cost →)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.set_xscale("log"); ax.grid(alpha=0.3,which="both"); ax.legend(fontsize=8)
    caption(fig); fig.tight_layout(rect=[0,0.22,1,1])
    fig.savefig(os.path.join(outdir,fname),dpi=130); plt.close(fig)

def barplots(S,fr,outdir,ck):
    x=np.arange(len(ORDER)); cols=[COLS[k] for k in ORDER]
    def mg(k,bud): ok=[p["gpus"] for p in fr[k] if p["tbt"]<=bud]; return min(ok) if ok else None
    def pk(k): return max(S*1000/(p["tbt"]*p["gpus"]) for p in fr[k]) if fr[k] else 0
    g1=[mg(k,100) for k in ORDER]; g2=[mg(k,200) for k in ORDER]; pkv=[pk(k) for k in ORDER]
    fig,ax=plt.subplots(figsize=(11,5.5)); w=0.38
    ax.bar(x-w/2,[g or 0 for g in g1],w,color=cols,alpha=0.55,hatch="//",label="TBT ≤100ms")
    ax.bar(x+w/2,[g or 0 for g in g2],w,color=cols,label="TBT ≤200ms")
    for i in range(len(ORDER)):
        ax.text(i-w/2,(g1[i] or 0)+0.5,g1[i] if g1[i] else "×",ha="center",fontsize=8)
        ax.text(i+w/2,(g2[i] or 0)+0.5,g2[i] if g2[i] else "×",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=20,ha="right",fontsize=8)
    ax.set_ylabel(f"GPUs to serve {S} sessions")
    ax.set_title(f"Cost at latency budget — {S}×{ck}K decode · fp8 · 1024 GB/s (lower=better)")
    ax.legend(); ax.grid(alpha=0.3,axis="y"); caption(fig); fig.tight_layout(rect=[0,0.24,1,1])
    fig.savefig(os.path.join(outdir,f"cost-at-slo_serve-{S}x{ck}k_fp8_hbfbw-1024.png"),dpi=130); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5)); ax.bar(x,pkv,color=cols)
    for i in range(len(ORDER)): ax.text(i,pkv[i]+2,f"{pkv[i]:.0f}",ha="center",fontweight="bold",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=20,ha="right",fontsize=8)
    ax.set_ylim(0,(max(pkv)*1.18 if pkv else 230))
    ax.set_ylabel("peak decode throughput per GPU (tok/s)")
    ax.set_title(f"Peak throughput/GPU — {S}×{ck}K decode · fp8 · 1024 GB/s")
    ax.grid(alpha=0.3,axis="y"); caption(fig); fig.tight_layout(rect=[0,0.24,1,1])
    fig.savefig(os.path.join(outdir,f"throughput-per-gpu_optimal_serve-{S}x{ck}k_fp8_hbfbw-1024.png"),dpi=130); plt.close(fig)

CONTEXTS=[(65536,"64"),(131072,"128")]

if __name__=="__main__":
    for CTX,CK in CONTEXTS:
        ev.C=CTX; ev.KV_SESS_GB=CTX*ev.KV_TOK_LAY*ev.L_TOTAL/1e9
        print(f"\n=== context {CK}K  (KV/session={ev.KV_SESS_GB:.1f} GB; "
              f"attn cap: HBF={math.floor((ev.HBF_CAP-ev.R_GB)/ev.KV_SESS_GB)} "
              f"HBM={math.floor((ev.HBM_CAP-ev.R_GB)/ev.KV_SESS_GB)} sessions/GPU) ===")
        for S in [128,256,512,1024]:
            outdir=os.path.join(FIGRT,f"S{S}"); os.makedirs(outdir,exist_ok=True)
            fr=build(S)
            lineplot(S,fr,outdir,"tbt","per-token latency TBT (ms) ↓",
                     f"cost-latency_serve-{S}x{CK}k_fp8_hbfbw-1024.png",
                     f"Cost vs latency — {S}×{CK}K decode · fp8 · 1024 GB/s",CK)
            lineplot(S,fr,outdir,"tps","aggregate decode throughput (tok/s) ↑",
                     f"cost-throughput_serve-{S}x{CK}k_fp8_hbfbw-1024.png",
                     f"Cost vs throughput — {S}×{CK}K decode · fp8 · 1024 GB/s",CK)
            barplots(S,fr,outdir,CK)
            print(f"  S={S:5d}: E_unique={E_unique(S):.0f} | 4 figures in figures/S{S}/")
    print("\ndone.")
