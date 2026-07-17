#!/usr/bin/env python3
"""
Recompute B1/B2/B3/B4/Ours with the CORRECTED models:
  B1  HBM colocated EP           (experts sharded 256/N, serial KV+MoE)
  B2  HBF colocated EP           (same, on flash)
  B3  HBF disagg REPLICATED, serial       (MoE GPU streams union of ITS tokens; step = attn+moe)
  B4  HBF disagg REPLICATED, overlap      (random dispatch; step = max(attn,moe), 2 microbatches)
  Ours HBF disagg REPLICATED, overlap + expert-aware clustering (greedy per-cluster union)

Union per MoE GPU comes from the greedy/random clustering harness (structure 0.9),
using the MAX cluster union (parallel pool -> slowest cluster sets latency).
"""
import importlib.util, math, os
import numpy as np
spec=importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__),"eval_baselines.py"))
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HBF=1024.0; STRUCT=0.9; SEEDS=6
N_EXPERTS,TOP_K,N_GROUPS,HOT=256,8,16,32
FIGRT=os.path.join(os.path.dirname(__file__),"..","figures")
COLS=dict(B1="#c0392b",B2="#2471a3",B3="#27ae60",B4="#8e44ad",Ours="#e67e22")
MRK =dict(B1="o",B2="s",B3="^",B4="D",Ours="*")

def gen(S,rng):
    hot=[rng.choice(N_EXPERTS,HOT,replace=False) for _ in range(N_GROUPS)]
    t=[]
    for _ in range(S):
        g=rng.integers(N_GROUPS); ch=set()
        while len(ch)<TOP_K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(N_EXPERTS)))
        t.append(frozenset(ch))
    return t
_cache={}
def union_max(S,M,greedy):
    key=(S,M,greedy)
    if key in _cache: return _cache[key]
    if M>S: _cache[key]=None; return None
    vals=[]
    for seed in range(SEEDS):
        t=gen(S,np.random.default_rng(seed)); cs=len(t)//M
        if greedy:
            rem=set(range(len(t))); u=[]
            for _ in range(M):
                s=next(iter(rem)); rem.discard(s); un=set(t[s]); n=1
                while n<cs and rem:
                    b=min(rem,key=lambda x:len(t[x]-un)); un|=t[b]; rem.discard(b); n+=1
                u.append(len(un))
        else:
            idx=np.random.default_rng(seed+9).permutation(len(t))
            u=[len(set().union(*[t[idx[i*cs+j]] for j in range(cs)])) for i in range(M)]
        vals.append(max(u))
    r=float(np.mean(vals)); _cache[key]=r; return r

def T_moe_repl(S,M,union):
    load=ev.ms(union*ev.PER_EXPERT+ev.SHARED_BYTES,HBF)
    comp=ev.moe_compute_ms(S)/M
    xfer=ev.ms(2*S*ev.D*ev.BE_ACT, M*ev.LINK_BW)   # replicated: 1 copy/token
    return (load+comp)*ev.L_MOE + xfer*ev.L_MOE

MSET=[1,2,3,4,6,8,12,16,24,32,48,64]
def frontier(pts):
    pts=sorted([p for p in pts if p],key=lambda p:(p["gpus"],p["tbt"])); out=[];best=1e18
    for p in pts:
        if p["tbt"]<best-1e-9: out.append(p); best=p["tbt"]
    return out

def build(S):
    def b1(b):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBM_CAP)
        if mn is None or N<mn: return None
        return dict(gpus=N,tbt=ev.T_attn(b,ev.BW_HBM)+ev.T_moe_colocated(S,N,ev.BW_HBM))
    def b2(b):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBF_CAP)
        if mn is None or N<mn: return None
        return dict(gpus=N,tbt=ev.T_attn(b,HBF)+ev.T_moe_colocated(S,N,HBF))
    def dis(A,M,ov,greedy):
        b_a=math.ceil(S/A)
        if b_a>154: return None
        u=union_max(S,M,greedy)
        if u is None: return None
        ta=ev.T_attn(b_a,HBF); tm=T_moe_repl(S,M,u)
        return dict(gpus=A+M,tbt=max(ta,tm) if ov else ta+tm)
    Amax=min(S,400)
    return {
      "B1":frontier([b1(b) for b in range(1,12)]),
      "B2":frontier([b2(b) for b in range(1,154)]),
      "B3":frontier([dis(A,M,False,False) for A in range(1,Amax+1) for M in MSET]),
      "B4":frontier([dis(A,M,True ,False) for A in range(1,Amax+1) for M in MSET]),
      "Ours":frontier([dis(A,M,True ,True ) for A in range(1,Amax+1) for M in MSET]),
    }

if __name__=="__main__":
    print(f"structure={STRUCT}.  Corrected models (replicated disagg + real clustering + max-overlap).\n")
    hdr=f"{'S':>5} {'sys':>5} {'G@100':>6} {'G@200':>6} {'peakTPS/GPU':>11}"
    for S in [128,256,512,1024]:
        fr=build(S); print(hdr if S==128 else "")
        outdir=os.path.join(FIGRT,f"S{S}"); os.makedirs(outdir,exist_ok=True)
        fig,ax=plt.subplots(figsize=(9,6))
        for k in ["B1","B2","B3","B4","Ours"]:
            f=sorted(fr[k],key=lambda p:p["gpus"])
            ax.plot([p["gpus"] for p in f],[p["tbt"] for p in f],MRK[k]+"-",color=COLS[k],label=k,ms=5)
            def mg(bud): ok=[p["gpus"] for p in fr[k] if p["tbt"]<=bud]; return min(ok) if ok else None
            peak=max(S*1000/(p["tbt"]*p["gpus"]) for p in fr[k])
            print(f"{S:>5} {k:>5} {str(mg(100)):>6} {str(mg(200)):>6} {peak:>11.1f}")
        ax.set_xlabel(f"total GPUs to serve {S}×128K (cost →)"); ax.set_ylabel("TBT (ms) ↓")
        ax.set_title(f"Corrected: serve {S}×128K decode · fp8 · HBF=1024\n"
                     f"disagg=replicated MoE · overlap=2 microbatches · clustering={int(STRUCT*100)}% struct")
        ax.set_xscale("log"); ax.grid(alpha=0.3,which="both"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir,f"cost-latency_serve-{S}x128k_fp8_hbfbw-1024.png"),dpi=130); plt.close(fig)
    print("\ncost-latency figures updated in paper_eval/figures/S*/")
