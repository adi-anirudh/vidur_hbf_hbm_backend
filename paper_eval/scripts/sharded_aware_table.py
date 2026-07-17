#!/usr/bin/env python3
"""
Expert-aware SHARDED EP.

B4 = sharded, all 256 experts, arbitrary placement: each MoE GPU holds 256/M
     experts, streams each once. Flash-load floor = (256/M)*PER_EXPERT.

Expert-aware placement of all 256 experts changes nothing for the flash-load
term (still 256/M per GPU, each loaded once) -> we DON'T model it, it can't win.

Expert-aware SELECTION (this file): globally accumulate each expert's gate-score
mass over the whole S-token batch, protect every token's top-1, drop the coldest
experts up to X% of total mass -> E_keep survivors, sharded E_keep/M per GPU.
Each survivor still loaded exactly once (no replication) => strictly below Ours.

Reference rows: B2 (HBF coloc), B4 (sharded no prune), Ours g=4 +10% (replicated).
"""
import importlib.util, math, os
import numpy as np
from collections import Counter
spec=importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__),"eval_baselines.py"))
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF=1024.0; STRUCT=0.9; SEEDS=6
NE,K,NG,HOT=256,8,16,32
SIGMA=1.5

def gen(S,rng):
    hot=[rng.choice(NE,HOT,replace=False) for _ in range(NG)]
    toks=[]
    for _ in range(S):
        g=rng.integers(NG); ch=set()
        while len(ch)<K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(NE)))
        ch=list(ch)
        logits=rng.normal(0,SIGMA,K); w=np.exp(logits-logits.max()); w/=w.sum()
        toks.append({e:float(wi) for e,wi in zip(ch,w)})
    return toks

_c={}
def E_keep_global(S,prune_frac):
    key=(S,prune_frac)
    if key in _c: return _c[key]
    vals=[]
    for sd in range(SEEDS):
        toks=gen(S,np.random.default_rng(sd))
        mass=Counter(); protected=set()
        for sc in toks:
            for e,s in sc.items(): mass[e]+=s
            protected.add(max(sc,key=sc.get))
        if prune_frac<=0:
            vals.append(len(mass)); continue
        total=sum(mass.values())
        prunable=sorted((e for e in mass if e not in protected), key=lambda e:mass[e])
        removed=0.0; drop=0
        for e in prunable:
            if removed+mass[e] > prune_frac*total: break
            removed+=mass[e]; drop+=1
        vals.append(len(mass)-drop)
    r=float(np.mean(vals)); _c[key]=r; return r

def T_moe_shard(S,M,E_keep):
    load=ev.ms((E_keep*ev.PER_EXPERT+ev.SHARED_BYTES)/M, HBF)
    comp=ev.moe_compute_ms(S)/M
    # sharded all-to-all: each token's K routes dispatched to <=min(K,M) GPUs
    copies=M*(1.0-(1.0-1.0/M)**K) if M>1 else 1.0
    xfer=ev.ms(2*S*copies*ev.D*ev.BE_ACT, M*ev.LINK_BW)
    return (load+comp)*ev.L_MOE + xfer*ev.L_MOE

def best_shard(S,budget,prune_frac):
    best=None
    for M in range(1,budget):
        Ek=E_keep_global(S,prune_frac)
        tm=T_moe_shard(S,M,Ek)
        for A in range(1,budget-M+1):
            b_a=math.ceil(S/A)
            if b_a>154: continue
            ta=ev.T_attn(b_a,HBF); step=max(ta,tm)
            if best is None or step<best["step"]:
                best=dict(gpus=A+M,ta=ta,tm=tm,step=step,u=Ek/M,A=A,M=M,Ek=Ek)
            if ta<=tm: break
    return best

def best_b2(S,budget):
    best=None
    for b in range(1,154):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBF_CAP)
        if mn is None or N<mn or N>budget: continue
        ta=ev.T_attn(b,HBF); tm=ev.T_moe_colocated(S,N,HBF); step=ta+tm
        if best is None or step<best["step"]:
            best=dict(gpus=N,ta=ta,tm=tm,step=step,u=ev.E_unique(S)/N,b=b,N=N,Ek=ev.E_unique(S))
    return best

if __name__=="__main__":
    for S,budget in [(512,40),(256,20)]:
        print(f"\n=== Serve {S}x128K, fp8, HBF={int(HBF)}, best within {budget} GPUs ===")
        print(f"{'System':<30}{'GPUs':>5}{'Tatt':>6}{'Tmoe':>6}{'step':>6}{'Ekeep':>7}{'u/gpu':>7}  config")
        def line(name,r,cfg):
            if r is None: print(f"{name:<30}   --"); return
            print(f"{name:<30}{r['gpus']:>5}{r['ta']:>6.0f}{r['tm']:>6.0f}{r['step']:>6.0f}"
                  f"{r['Ek']:>7.0f}{r['u']:>7.0f}  {cfg(r)}")
        line("B2 HBF coloc",best_b2(S,budget),lambda r:f"b={r['b']},N={r['N']}")
        line("B4 sharded, no prune",best_shard(S,budget,0.0),lambda r:f"A={r['A']}:M={r['M']}")
        for pf in [0.05,0.10,0.15,0.20]:
            line(f"Sharded + global prune {int(pf*100)}%",best_shard(S,budget,pf),
                 lambda r:f"A={r['A']}:M={r['M']}")
