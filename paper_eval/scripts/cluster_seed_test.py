#!/usr/bin/env python3
"""
Clustering seed strategy: naive (arbitrary first token) vs isolated-seed
(start each cluster with the most-isolated token = rarest experts), then greedy
min-new-expert fill.  Metric that matters = MAX cluster union (parallel MoE
latency is set by the slowest MoE GPU).
"""
import numpy as np
from collections import Counter

N_EXPERTS,TOP_K,N_GROUPS,HOT,STRUCT=256,8,16,32,0.9

def gen(S,rng):
    hot=[rng.choice(N_EXPERTS,HOT,replace=False) for _ in range(N_GROUPS)]
    t=[]
    for _ in range(S):
        g=rng.integers(N_GROUPS); ch=set()
        while len(ch)<TOP_K:
            ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(N_EXPERTS)))
        t.append(frozenset(ch))
    return t

def cluster(toks, M, seed_mode):
    cs=len(toks)//M; rem=set(range(len(toks))); unions=[]
    for _ in range(M):
        if not rem: break
        if seed_mode=="naive":
            seed=next(iter(rem))
        else:  # isolated: token whose experts are rarest among REMAINING tokens
            freq=Counter(e for i in rem for e in toks[i])
            seed=min(rem, key=lambda i: sum(freq[e] for e in toks[i]))
        rem.discard(seed); un=set(toks[seed]); n=1
        while n<cs and rem:
            b=min(rem, key=lambda x: len(toks[x]-un)); un|=toks[b]; rem.discard(b); n+=1
        unions.append(len(un))
    return np.mean(unions), max(unions)

if __name__=="__main__":
    print(f"structure={STRUCT}.  max cluster union (sets parallel MoE latency).\n")
    print(f"{'S':>5}{'M':>4}{'naive_mean':>11}{'iso_mean':>9}{'naive_MAX':>10}{'iso_MAX':>8}{'MAX red':>8}")
    for S in [256,512,1024]:
        for M in [4,8,16,32]:
            if M>S: continue
            nm,nx,im,ix=[],[],[],[]
            for seed in range(8):
                t=gen(S,np.random.default_rng(seed))
                a,b=cluster(t,M,"naive");  nm.append(a); nx.append(b)
                a,b=cluster(t,M,"iso");     im.append(a); ix.append(b)
            NM,NX,IM,IX=map(np.mean,(nm,nx,im,ix))
            print(f"{S:>5}{M:>4}{NM:>11.0f}{IM:>9.0f}{NX:>10.0f}{IX:>8.0f}{(1-IX/NX)*100:>7.0f}%")
    print("\nnaive = arbitrary seed;  iso = most-isolated (rarest-expert) seed.")
