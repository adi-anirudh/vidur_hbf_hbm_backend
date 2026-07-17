#!/usr/bin/env python3
"""
Sibling-pairing + cluster-level score-mass pruning table, for g in {1,2,3,4}.

Ours = disagg, REPLICATED experts, expert-aware clustering, 2-way overlap.
  M MoE GPUs, g siblings/cluster  ->  num_clusters = M//g.
  Cluster the S tokens into num_clusters clusters (iso-seed greedy min-union).
  The g siblings of a cluster SHARD that cluster's expert union
      -> per-GPU union = (max cluster union) / g.
  g=1  -> pure clustering (one GPU per cluster).
  g=M  -> one cluster of all tokens sharded M ways = sharded EP (256/M) = B4.

Pruning (cluster level, accuracy-safe):
  accumulate each expert's gate-score mass over the cluster's tokens;
  never drop an expert that is any token's top-1;
  drop the lowest-mass experts until X% of the cluster's total score mass is removed.
  -> shrinks the cluster union before the /g sharding.

Reference rows: B2 (HBF colocated) and B4 (HBF disagg sharded, overlap).
"""
import importlib.util, math, os
import numpy as np
from collections import Counter
spec=importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__),"eval_baselines.py"))
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF=1024.0; STRUCT=0.9; SEEDS=6
NE,K,NG,HOT=256,8,16,32
SIGMA=1.5                       # gate-logit spread for the score model

def gen(S,rng):
    """tokens = list of (frozenset experts, {expert:gate_score})."""
    hot=[rng.choice(NE,HOT,replace=False) for _ in range(NG)]
    toks=[]
    for _ in range(S):
        g=rng.integers(NG); ch=set()
        while len(ch)<K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(NE)))
        ch=list(ch)
        logits=rng.normal(0,SIGMA,K); w=np.exp(logits-logits.max()); w/=w.sum()
        score={e:float(wi) for e,wi in zip(ch,w)}
        toks.append((frozenset(ch),score))
    return toks

def cluster_unions(toks,ncl):
    """iso-seed greedy min-new-expert clustering into ncl clusters.
       returns list of clusters, each a list of token indices."""
    cs=len(toks)//ncl; rem=set(range(len(toks))); clusters=[]
    for _ in range(ncl):
        if not rem: break
        freq=Counter(e for i in rem for e in toks[i][0])
        seed=min(rem,key=lambda i:sum(freq[e] for e in toks[i][0]))
        rem.discard(seed); members=[seed]; un=set(toks[seed][0]); n=1
        while n<cs and rem:
            b=min(rem,key=lambda x:len(toks[x][0]-un)); un|=toks[b][0]; rem.discard(b); members.append(b); n+=1
        clusters.append(members)
    # any leftover tokens (rounding) -> smallest-growth cluster
    for i in list(rem):
        clusters[0].append(i)
    return clusters

def pruned_union(toks,members,prune_frac):
    """cluster union after dropping lowest-mass experts up to prune_frac of
       total score mass, protecting every token's top-1."""
    mass=Counter()
    protected=set()
    for i in members:
        fs,sc=toks[i]
        for e,s in sc.items(): mass[e]+=s
        protected.add(max(sc,key=sc.get))          # this token's top-1
    if prune_frac<=0:
        return len(mass)
    total=sum(mass.values())
    prunable=sorted((e for e in mass if e not in protected), key=lambda e:mass[e])
    removed=0.0; drop=set()
    for e in prunable:
        if removed+mass[e] > prune_frac*total: break
        removed+=mass[e]; drop.add(e)
    return len(mass)-len(drop)

_cache={}
def ours_union_per_gpu(S,M,g,prune_frac):
    """max over clusters of (pruned cluster union) / g, averaged over seeds."""
    ncl=M//g
    if ncl<1 or ncl>S: return None
    key=(S,M,g,prune_frac)
    if key in _cache: return _cache[key]
    vals=[]
    for sd in range(SEEDS):
        toks=gen(S,np.random.default_rng(sd))
        cls=cluster_unions(toks,ncl)
        u=[pruned_union(toks,m,prune_frac) for m in cls]
        vals.append(max(u)/g)
    r=float(np.mean(vals)); _cache[key]=r; return r

def T_moe_repl(S,M,union_per_gpu):
    load=ev.ms(union_per_gpu*ev.PER_EXPERT+ev.SHARED_BYTES,HBF)
    comp=ev.moe_compute_ms(S)/M
    xfer=ev.ms(2*S*ev.D*ev.BE_ACT, M*ev.LINK_BW)
    return (load+comp)*ev.L_MOE + xfer*ev.L_MOE

def best_ours(S,budget,g,prune_frac):
    best=None
    for M in range(g, budget, g):                   # M divisible by g
        u=ours_union_per_gpu(S,M,g,prune_frac)
        if u is None: continue
        tm=T_moe_repl(S,M,u)
        for A in range(1,budget-M+1):
            b_a=math.ceil(S/A)
            if b_a>154: continue
            ta=ev.T_attn(b_a,HBF); step=max(ta,tm)
            if best is None or step<best["step"]:
                best=dict(gpus=A+M,ta=ta,tm=tm,step=step,u=u,A=A,M=M)
            if ta<=tm: break                          # more A won't help
    return best

def best_b2(S,budget):
    best=None
    for b in range(1,154):
        N=math.ceil(S/b); mn=ev.min_N(b,ev.HBF_CAP)
        if mn is None or N<mn or N>budget: continue
        ta=ev.T_attn(b,HBF); tm=ev.T_moe_colocated(S,N,HBF); step=ta+tm
        if best is None or step<best["step"]:
            best=dict(gpus=N,ta=ta,tm=tm,step=step,u=ev.E_unique(S)/N,b=b,N=N)
    return best

def best_b4(S,budget):
    best=None
    for M in range(1,budget):
        tm=ev.T_moe_disagg(S,M,HBF)
        for A in range(1,budget-M+1):
            b_a=math.ceil(S/A)
            if b_a>154: continue
            ta=ev.T_attn(b_a,HBF); step=max(ta,tm)
            if best is None or step<best["step"]:
                best=dict(gpus=A+M,ta=ta,tm=tm,step=step,u=ev.E_unique(S)/M,A=A,M=M)
            if ta<=tm: break
    return best

if __name__=="__main__":
    for S,budget in [(512,40),(256,20)]:
        print(f"\n=== Serve {S}x128K, fp8, HBF={int(HBF)}, best config within {budget} GPUs ===")
        print(f"{'System':<26}{'GPUs':>5}{'Tatt':>6}{'Tmoe':>6}{'step':>6}{'u/gpu':>7}  config")
        def line(name,r,cfg):
            if r is None: print(f"{name:<26}   --"); return
            print(f"{name:<26}{r['gpus']:>5}{r['ta']:>6.0f}{r['tm']:>6.0f}{r['step']:>6.0f}{r['u']:>7.0f}  {cfg(r)}")
        line("B2 HBF coloc",best_b2(S,budget),lambda r:f"b={r['b']},N={r['N']}")
        line("B4 disagg shard overlap",best_b4(S,budget),lambda r:f"A={r['A']}:M={r['M']}")
        for g in [1,2,3,4]:
            for pf in [0.0,0.05,0.10]:
                tag=f"Ours g={g} " + ("no prune" if pf==0 else f"+{int(pf*100)}%")
                line(tag,best_ours(S,budget,g,pf),lambda r:f"A={r['A']}:M={r['M']}")
