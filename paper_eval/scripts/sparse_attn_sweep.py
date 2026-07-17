#!/usr/bin/env python3
"""
Sparse-attention sweep for the pre workloads (S in {128,256,512,1024}, ctx 64K/128K).

Dense decode re-reads the FULL context KV every step -> attention-bound.
Sparse attention reads only a bounded BUDGET of selected KV tokens/step
(local window + top-k blocks, a la Quest / native sparse attention). We model:
  - READ per step  = min(C, BUDGET) tokens' KV     (bandwidth -> T_attn shrinks)
  - STORAGE        = full C KV (conservative: no eviction -> capacity UNCHANGED,
                     so attn-GPU session capacity / b_a cap stays as dense)
Everything MoE-side is unchanged; HBF=HBM=1024 GB/s (your knob).

Question: does sparse attention flip the step to MoE-bound, so dynamic
activation (Ours) and global pruning (Ours+g) actually reduce cost/latency?
"""
import importlib.util, math
import numpy as np
from collections import Counter
spec=importlib.util.spec_from_file_location("ev","paper_eval/scripts/eval_baselines.py")
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

HBF=1024.0; NE,K,NG,HOT,STRUCT,SIGMA=256,8,16,32,0.9,1.5; SEEDS=4; CKEEP=0.95

# ---- routing stats generated ONCE per S ----
def stats(S):
    out=[]
    for sd in range(SEEDS):
        rng=np.random.default_rng(sd)
        hot=[rng.choice(NE,HOT,replace=False) for _ in range(NG)]
        mass=Counter(); prot=set(); active=set()
        for _ in range(S):
            g=rng.integers(NG); ch=set()
            while len(ch)<K: ch.add(int(rng.choice(hot[g])) if rng.random()<STRUCT else int(rng.integers(NE)))
            ch=list(ch); lg=rng.normal(0,SIGMA,K); w=np.exp(lg-lg.max()); w/=w.sum()
            active.update(ch); prot.add(ch[int(np.argmax(w))])
            for e,wi in zip(ch,w): mass[e]+=float(wi)
        out.append((active,mass,prot))
    return out
def E_unique(st): return float(np.mean([len(a) for a,_,_ in st]))
def E_keep(st,c):
    vals=[]
    for a,mass,prot in st:
        tot=sum(mass.values()); acc=0; kept=set()
        for e in sorted(mass,key=lambda e:-mass[e]):
            kept.add(e); acc+=mass[e]
            if acc>=c*tot: break
        kept|=prot; vals.append(len(kept))
    return float(np.mean(vals))
def static_max(st,M):
    if M<=1: return E_unique(st)
    vals=[]
    for i,(a,_,_) in enumerate(st):
        asg=np.random.default_rng(i+100).integers(0,M,NE)
        vals.append(max(Counter(int(asg[e]) for e in a).values()))
    return float(np.mean(vals))

def T_attn(b, attended):
    return ev.ms(b*attended*ev.KV_TOK_LAY*ev.L_TOTAL, HBF)+ev.ms(ev.W_ATTN_LAY*ev.L_TOTAL, HBF)
def T_moe(S,M,Eact):
    # Eact = experts on the BUSIEST MoE GPU (already per-GPU). No extra /M.
    load=ev.ms(Eact*ev.PER_EXPERT+ev.SHARED_BYTES,HBF)
    comp=ev.moe_compute_ms(S)/M
    copies=M*(1-(1-1/M)**K) if M>1 else 1
    xfer=ev.ms(2*S*copies*ev.D*ev.BE_ACT,M*ev.LINK_BW)
    return (load+comp)*ev.L_MOE+xfer*ev.L_MOE

def best_split(S,st,G,attended,mode,eu,eg):
    cap=math.floor((ev.HBF_CAP-ev.R_GB)/ev.KV_SESS_GB)     # full-KV storage cap
    best=1e18; bs=None
    for M in range(1,min(G,200)):
        A=G-M; b_a=math.ceil(S/A)
        if b_a>cap: continue
        ta=T_attn(b_a,attended)
        load=static_max(st,M) if mode=="static" else (eu/M if mode=="dyn" else eg/M)
        step=max(ta,T_moe(S,M,load))
        if step<best: best=step; bs=(A,M,ta,T_moe(S,M,load))
    return best,bs
def min_G(S,st,attended,slo,mode,eu,eg):
    lo=math.ceil(S/math.floor((ev.HBF_CAP-ev.R_GB)/ev.KV_SESS_GB))+1
    for G in range(max(2,lo),4000):
        step,bs=best_split(S,st,G,attended,mode,eu,eg)
        if step<=slo: return G,step,bs
    return None,None,None

if __name__=="__main__":
    BUDGETS=[("dense",None),("8K",8192),("4K",4096),("2K",2048)]
    for CTX,CK in [(131072,"128K"),(65536,"64K")]:
        ev.C=CTX; ev.KV_SESS_GB=CTX*ev.KV_TOK_LAY*ev.L_TOTAL/1e9
        print("="*92)
        print(f"CONTEXT {CK}  (KV storage/session={ev.KV_SESS_GB:.1f} GB, unchanged by sparsity)  SLO=100ms")
        print("="*92)
        for S in [512,1024]:
            st=stats(S); eu=E_unique(st); eg=E_keep(st,CKEEP)
            print(f"\n S={S}  (E_unique={eu:.0f}, pruned c={CKEEP} top1={eg:.0f})")
            print(f"  {'budget':>7}{'attended':>9}   {'sys':<7}{'G':>5}{'A':>5}{'M':>4}{'Tattn':>7}{'Tmoe':>7}{'step':>7}{'bound':>6}   vsB4")
            for blab,bud in BUDGETS:
                attended=min(CTX,bud) if bud else CTX
                gB4=None
                for name,mode in [("B4","static"),("Ours","dyn"),("Ours+g","global")]:
                    G,step,bs=min_G(S,st,attended,100,mode,eu,eg)
                    if not bs:
                        print(f"  {blab:>7}{attended:>9}   {name:<7} unreachable"); continue
                    if name=="B4": gB4=G
                    d="" if name=="B4" else f"{G-gB4:+d} ({(G-gB4)/gB4*100:+.0f}%)"
                    bnd="MOE" if bs[3]>=bs[2] else "ATTN"
                    print(f"  {blab:>7}{attended:>9}   {name:<7}{G:>5}{bs[0]:>5}{bs[1]:>4}{bs[2]:>7.0f}{bs[3]:>7.0f}{step:>7.0f}{bnd:>6}   {d}")
                print()
