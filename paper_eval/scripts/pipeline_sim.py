#!/usr/bin/env python3
"""
Discrete-event sim of dependency-driven attention/MoE pipelining (your scheme),
with a FAIR FIFO scheduler and system-throughput measurement.

B microbatches, L layers. Each microbatch is a strict chain of 2L tasks:
  a0,m0,a1,m1,...,a(L-1),m(L-1)  -> one token -> repeat (decode).
Two FIFO servers: attention pool (a-tasks), MoE pool (m-tasks). A server, when
free, takes the head of its ready queue (task whose predecessor is done) — i.e.
"run the next ready task". Total per-pool work fixed: tau_a=T_attn/(B*L),
tau_m=T_moe/(B*L). Metric: steady-state per-token interval (TBT) = wall/rounds.
"""
import heapq
from collections import deque
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os

def simulate(L, B, T_attn, T_moe, warmup=30, measure=120):
    tau_a = T_attn/(B*L); tau_m = T_moe/(B*L)
    stage = [0]*B                     # next task index 0..2L-1 (even=attn, odd=moe)
    qA = deque(range(B)); qM = deque()
    freeA = [True]; freeM = [True]
    ev = []                            # heap of (finish_time, server, b)
    tnow = [0.0]
    tokens = [0]*B
    def start():
        if freeA[0] and qA:
            b = qA.popleft(); freeA[0] = False
            heapq.heappush(ev, (tnow[0]+tau_a, 0, b))
        if freeM[0] and qM:
            b = qM.popleft(); freeM[0] = False
            heapq.heappush(ev, (tnow[0]+tau_m, 1, b))
    start()
    t_warm = None; tok_warm = 0; total = 0
    while total < B*(warmup+measure):
        tnow[0], server, b = heapq.heappop(ev)
        if server == 0: freeA[0] = True
        else:           freeM[0] = True
        stage[b] += 1
        if stage[b] == 2*L:                      # token complete
            stage[b] = 0; tokens[b] += 1; total += 1
            qA.append(b)
            if t_warm is None and min(tokens) >= warmup:
                t_warm = tnow[0]; tok_warm = total
        elif stage[b] % 2 == 0:                  # next task is attention
            qA.append(b)
        else:                                    # next task is moe
            qM.append(b)
        start()
    rounds = (total - tok_warm)/B
    return (tnow[0] - t_warm)/rounds              # per-session TBT

if __name__ == "__main__":
    L = 58
    cases = [("balanced  (T_attn=339, T_moe=333)", 339, 333),
             ("moe-light (T_attn=339, T_moe=150)", 339, 150),
             ("moe-heavy (T_attn=150, T_moe=333)", 150, 333)]
    outdir = os.path.join(os.path.dirname(__file__), "..", "figures", "ours_system")
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    for name, Ta, Tm in cases:
        Bs = list(range(1, 9)); tbts = [simulate(L, B, Ta, Tm) for B in Bs]
        ax.plot(Bs, tbts, "o-", label=name)
        print(f"{name}:  serial(B=1)={Ta+Tm}  overlap-limit max()={max(Ta,Tm)}")
        print("   " + "  ".join(f"B{B}:{t:.0f}" for B,t in zip(Bs,tbts)))
    for _, Ta, Tm in cases:
        ax.axhline(max(Ta,Tm), ls=":", color="gray", alpha=0.4)
    ax.set_xlabel("microbatches in flight (B)"); ax.set_ylabel("steady-state TBT (ms)")
    ax.set_title("Dependency-driven pipeline (fair FIFO): TBT vs microbatches\n"
                 "B=1 serial (T_attn+T_moe); B≥2 converges to max(T_attn,T_moe). L=58")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pipeline_tbt-vs-microbatches.png"), dpi=130)
    print("figure -> paper_eval/figures/ours_system/pipeline_tbt-vs-microbatches.png")
