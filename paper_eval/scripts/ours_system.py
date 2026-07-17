#!/usr/bin/env python3
"""
Model of YOUR system: disaggregated, replicated experts on MoE pool, 2-way overlap.
Tokens dispatched to M MoE GPUs; each MoE GPU streams the expert union of ITS tokens.

  NoCluster : tokens split randomly  -> per-MoE-GPU union = random-split max
  Ours      : expert-aware greedy clustering -> per-MoE-GPU union = greedy max

Clustering lowers T_moe  ->  the optimal attention:MoE split (A:M) re-balances
toward attention  ->  lower step.  We RE-OPTIMIZE A:M for each.
Latency uses the SLOWEST MoE GPU (max cluster union), since the pool runs parallel.
"""
import importlib.util, math, os
import numpy as np
spec=importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__),"eval_baselines.py"))
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF=1024.0; STRUCT=0.9
N_EXPERTS,TOP_K,N_GROUPS,HOT=256,8,16,32

def gen(S,rng):
    hot=[rng.choice(N_EXPERTS,HOT,replace=False) for _ in range(N_GROUPS)]
    t=[]
    for _ in range(S):
        g=rng.integers(N_GROUPS); ch=set()
        while len(ch)<TOP_K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(N_EXPERTS)))
        t.append(frozenset(ch))
    return t
def union_max(S,M,greedy):
    if M>S: return None
    vals=[]
    for seed in range(30):
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
    return float(np.mean(vals))

def T_moe(S, M, union_per_gpu):
    load = ev.ms(union_per_gpu*ev.PER_EXPERT + ev.SHARED_BYTES, HBF)
    comp = ev.moe_compute_ms(S)/M
    xfer = ev.ms(2*S*ev.D*ev.BE_ACT, M*ev.LINK_BW)      # clustering: 1 copy/token
    return (load+comp)*ev.L_MOE + xfer*ev.L_MOE

MSET=[1,2,3,4,6,8,12,16,24,32,48,64]

def frontier(S, greedy):
    umap={M:union_max(S,M,greedy) for M in MSET}
    tm={M:(T_moe(S,M,umap[M]) if umap[M] else None) for M in MSET}
    pts=[]
    for M in MSET:
        if tm[M] is None: continue
        for A in range(max(1,math.ceil(S/154)), 3*S+1):
            b_a=math.ceil(S/A)
            if b_a>154: continue
            ta=ev.T_attn(b_a,HBF)
            pts.append(dict(gpus=A+M, step=max(ta,tm[M]), A=A, M=M))
            if ta<=tm[M]: break
    pts.sort(key=lambda p:(p["gpus"],p["step"])); out=[]; best=1e18
    for p in pts:
        if p["step"]<best-1e-9: out.append(p); best=p["step"]
    return out

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    outdir=os.path.join(os.path.dirname(__file__),"..","figures","ours_system"); os.makedirs(outdir,exist_ok=True)
    print(f"structure={STRUCT}.  Cost-latency frontier, YOUR system (disagg+overlap+replicated).")
    print(f"clustering lowers T_moe -> A:M rebalances toward attention -> lower step.\n")
    for S in [128,256,512,1024]:
        fn=frontier(S,False); fo=frontier(S,True)
        fig,ax=plt.subplots(figsize=(8.5,5.5))
        for f,c,lab in [(fn,"#8e44ad","NoCluster (random dispatch)"),
                        (fo,"#e67e22",f"Ours (expert-aware clustering, {int(STRUCT*100)}% struct)")]:
            f=sorted(f,key=lambda p:p["gpus"])
            ax.plot([p["gpus"] for p in f],[p["step"] for p in f],"o-",color=c,label=lab,ms=4)
        ax.set_xlabel(f"total GPUs (A+M) to serve {S}×128K"); ax.set_ylabel("step / TBT (ms)")
        ax.set_title(f"Your system: clustering vs random dispatch — serve {S}×128K\n"
                     f"disagg · overlap · replicated MoE · HBF=1024")
        ax.set_xscale("log"); ax.grid(alpha=0.3,which="both"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir,f"ours-vs-nocluster_S{S}.png"),dpi=130); plt.close(fig)
        # report gain at a few budgets
        def at(f,g): c=[p["step"] for p in f if p["gpus"]<=g]; return min(c) if c else None
        row=[]
        for g in [16,32,64,96]:
            a,b=at(fn,g),at(fo,g)
            row.append(f"G≤{g}: {a:.0f}->{b:.0f}ms ({(1-b/a)*100:+.0f}%)" if a and b else f"G≤{g}: -")
        print(f"S={S:5d}  "+"   ".join(row))
    print("\nfigures -> paper_eval/figures/ours_system/")
