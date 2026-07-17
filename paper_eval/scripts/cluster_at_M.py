#!/usr/bin/env python3
"""
Clustering into M groups (one per MoE GPU) at the loads/M we actually use.

For S tokens split into M clusters, compare per-MoE-GPU expert union under:
  RANDOM  M-way split (naive replicated dispatch)
  GREEDY  expert-aware clustering (Ours)
  SHARDED-EP reference = E_unique(S)/M  (each expert loaded once, sharded)

MoE flash time per GPU ∝ the per-GPU union, and the pool runs in PARALLEL,
so what sets latency is the MAX cluster union, not the mean.
"""
import numpy as np

N_EXPERTS, TOP_K, N_GROUPS, HOT = 256, 8, 16, 32
def E_unique(B): return N_EXPERTS*(1-(1-TOP_K/N_EXPERTS)**B)

def gen(S, structure, rng):
    hot=[rng.choice(N_EXPERTS,HOT,replace=False) for _ in range(N_GROUPS)]
    toks=[]
    for _ in range(S):
        g=rng.integers(N_GROUPS); ch=set()
        while len(ch)<TOP_K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<structure else int(rng.integers(N_EXPERTS)))
        toks.append(frozenset(ch))
    return toks

def split_random(toks, M, rng):
    idx=rng.permutation(len(toks)); cs=len(toks)//M
    u=[len(set().union(*[toks[j] for j in idx[i*cs:(i+1)*cs]])) for i in range(M)]
    return u
def split_greedy(toks, M):
    rem=set(range(len(toks))); cs=len(toks)//M; u=[]
    for _ in range(M):
        seed=next(iter(rem)); rem.discard(seed); un=set(toks[seed]); n=1
        while n<cs and rem:
            b=min(rem,key=lambda t:len(toks[t]-un)); un|=toks[b]; rem.discard(b); n+=1
        u.append(len(un))
    return u

if __name__=="__main__":
    STRUCT=0.9   # matches the ~50% reduction you observed at cluster-size 16
    print(f"routing structure={STRUCT}  (latent {N_GROUPS} groups)\n")
    print(f"{'S':>5}{'M':>4}{'clust':>6}{'rnd_mean':>9}{'grd_mean':>9}{'grd_max':>8}"
          f"{'shardEP':>8}   who streams fewest")
    for S in [128,256,512,1024]:
        for M in [2,4,8,16,32]:
            if M>S: continue
            rr,gg,gx=[],[],[]
            for seed in range(4):
                t=gen(S,STRUCT,np.random.default_rng(seed))
                rr.append(np.mean(split_random(t,M,np.random.default_rng(seed+50))))
                g=split_greedy(t,M); gg.append(np.mean(g)); gx.append(np.max(g))
            rmean,gmean,gmax=np.mean(rr),np.mean(gg),np.mean(gx)
            ep=E_unique(S)/M
            best=min([("random",rmean),("greedy",gmean),("shardEP",ep)],key=lambda x:x[1])[0]
            print(f"{S:>5}{M:>4}{S//M:>6}{rmean:>9.0f}{gmean:>9.0f}{gmax:>8.0f}{ep:>8.1f}   {best}")
    print("\ngrd_max = slowest cluster (sets parallel MoE latency).")
    print("shardEP = balanced expert-parallel dispatch, each expert loaded once.")
