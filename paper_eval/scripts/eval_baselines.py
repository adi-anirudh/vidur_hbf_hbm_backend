#!/usr/bin/env python3
"""
Baseline evaluation for the HBF attention/MoE study — DeepSeek-V3, fp8, 128K decode.

Systems
  B1  HBM  colocated  EP        (experts in HBM)
  B2  HBF  colocated  EP        (experts on flash)
  B3  HBF  disaggregated serial (attention pool + MoE pool, no overlap)

Figures written to paper_eval/figures/ with descriptive names.
The HBF bandwidth is a CLI knob so we can test "HBF == HBM bandwidth":

    python eval_baselines.py --hbf-bw 768     # real HBF (default)
    python eval_baselines.py --hbf-bw 1024    # HBF as fast as HBM

Model terms lifted verbatim from vidur/memory_backends/moe_flash.py and
hbf_execution_time_predictor.py. Serial (no micro-batch overlap).
"""
import argparse
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# ── DeepSeek-V3 geometry ───────────────────────────────────────────────────
D, He, SHARED_He = 7168, 2048, 2048
E, K             = 256, 8
L_TOTAL, L_MOE   = 61, 58
C                = 131072
Nq, Nkv, Dh      = 224, 9, 32

BE_EXPERT, BE_KV, BE_ACT = 1, 1, 2          # fp8 experts, fp8 KV, bf16 wire
KV_TOK_LAY   = 2 * Nkv * Dh * BE_KV                       # 576 B
PER_EXPERT   = (2*D*He + He*D) * BE_EXPERT               # 44.04 MB
SHARED_BYTES = (2*D*SHARED_He + SHARED_He*D) * BE_EXPERT
W_ATTN_LAY   = (D*Nq*Dh + 2*D*Nkv*Dh + Nq*Dh*D) * BE_EXPERT   # QKV + O

# ── hardware ────────────────────────────────────────────────────────────────
BW_HBM   = 1024.0        # bytes/ns
LINK_BW  = 900.0         # NVLink bytes/ns
FLOPS_MS = 1000e12/1e3   # H100 fp16 flops per ms

# ── capacity constants ──────────────────────────────────────────────────────
EXPERTS_GB, R_GB = 656.0, 15.0
KV_SESS_GB       = C * KV_TOK_LAY * L_TOTAL / 1e9     # 4.6 GB (fp8, 128K, 61 layers)
HBM_CAP, HBF_CAP = 72.0, 722.0

def ms(b, bw):   return (b / bw) / 1e6
def E_unique(B): return E * (1.0 - (1.0 - K/E)**B)
def moe_compute_ms(t): return (t*K*6*D*He + t*6*D*SHARED_He) / FLOPS_MS
def min_N(b, cap):
    d = cap - R_GB - KV_SESS_GB*b
    return math.ceil(EXPERTS_GB/d) if d > 0 else None

def T_attn(b, bw_kv):
    return ms(b*C*KV_TOK_LAY*L_TOTAL, bw_kv) + ms(W_ATTN_LAY*L_TOTAL, BW_HBM)

def T_moe_colocated(tok, N, bw):
    load = ms((E_unique(tok)*PER_EXPERT + SHARED_BYTES)/N, bw)
    comp = moe_compute_ms(tok)/N
    a2a  = ms(2*(tok/N)*K*D*BE_ACT*(N-1)/N, LINK_BW) if N > 1 else 0.0
    return (load + comp)*L_MOE + a2a*L_MOE

def T_moe_disagg(tok, M, bw, cluster_eff=1.0):
    # cluster_eff<1: expert-aware clustering shrinks the routed union that must
    # be streamed (shared expert always on). Compute & transfer unchanged.
    routed = cluster_eff * E_unique(tok) * PER_EXPERT
    load = ms((routed + SHARED_BYTES)/M, bw)
    comp = moe_compute_ms(tok)/M
    copies = M*(1.0-(1.0-1.0/M)**K) if M > 1 else 1.0
    xfer = ms(2*tok*copies*D*BE_ACT, M*LINK_BW)      # min(A,M)~M
    return (load + comp)*L_MOE + xfer*L_MOE

CLUSTER_EFF = 0.20   # "Ours": only 20% of routed experts active after clustering


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — capacity: batch-per-GPU vs GPUs-to-fit (BW-independent)
# ═══════════════════════════════════════════════════════════════════════════
def fig_capacity(kv_gb, tag):
    def N_of(b, cap):
        d = cap - R_GB - kv_gb*b
        return math.ceil(EXPERTS_GB/d) if d > 0 else None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cap, c, m, lab in [(HBM_CAP,"#c0392b","o","B1  HBM (72 GB)"),
                           (HBF_CAP,"#2471a3","s","B2  HBF (722 GB)")]:
        xs, ys = [], []
        for b in range(1, 90):
            n = N_of(b, cap)
            if n is not None: xs.append(b); ys.append(n)
        ax.plot(xs, ys, m+"-", color=c, lw=2, ms=4, label=lab)
    ax.set_xlabel("batch per GPU  b  (128K sessions)")
    ax.set_ylabel("GPUs needed to fit  N")
    ax.set_title(f"Capacity: batch-per-GPU vs GPUs to fit ({tag})\n"
                 f"N = ceil(656 / (cap - 15 - {kv_gb:.1f}·b))")
    ax.set_ylim(0, 90); ax.set_xlim(0, 90); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"capacity_batch-vs-gpus_B1-B2_{tag}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print("  wrote", os.path.relpath(path))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — cost vs latency frontier to serve S sessions
# ═══════════════════════════════════════════════════════════════════════════
def frontier(points):
    pts = sorted(points, key=lambda p: (p["gpus"], p["tbt"]))
    out, best = [], float("inf")
    for p in pts:
        if p["tbt"] < best - 1e-9:
            out.append(p); best = p["tbt"]
    return out

def fig_cost(S, hbf_bw, tag):
    def b1c(b):
        N = math.ceil(S/b); mn = min_N(b, HBM_CAP)
        if mn is None or N < mn: return None
        return dict(gpus=N, tbt=T_attn(b, BW_HBM)+T_moe_colocated(S, N, BW_HBM))
    def b2c(b):
        N = math.ceil(S/b); mn = min_N(b, HBF_CAP)
        if mn is None or N < mn: return None
        return dict(gpus=N, tbt=T_attn(b, hbf_bw)+T_moe_colocated(S, N, hbf_bw))
    def b3c(A, M):      # serial ping-pong: attn + moe   (fine A sweep)
        b_a = math.ceil(S/A)
        if b_a > 154: return None
        return dict(gpus=A+M, tbt=T_attn(b_a, hbf_bw)+T_moe_disagg(S, M, hbf_bw))
    def b4c(A, M):      # B3 + 2-way overlap: max(attn, moe)
        b_a = math.ceil(S/A)
        if b_a > 154: return None
        return dict(gpus=A+M, tbt=max(T_attn(b_a, hbf_bw), T_moe_disagg(S, M, hbf_bw)))
    def oursc(A, M):    # Ours: B4 + clustering (routed union -> CLUSTER_EFF)
        b_a = math.ceil(S/A)
        if b_a > 154: return None
        return dict(gpus=A+M, tbt=max(T_attn(b_a, hbf_bw),
                                      T_moe_disagg(S, M, hbf_bw, CLUSTER_EFF)))
    f1 = frontier([r for b in range(1,12)  if (r:=b1c(b))])
    f2 = frontier([r for b in range(1,154) if (r:=b2c(b))])
    f3 = frontier([r for A in range(1,S+1) for M in range(1,40) if (r:=b3c(A,M))])
    f4 = frontier([r for A in range(1,S+1) for M in range(1,40) if (r:=b4c(A,M))])
    f5 = frontier([r for A in range(1,S+1) for M in range(1,40) if (r:=oursc(A,M))])
    fig, ax = plt.subplots(figsize=(9, 6))
    for f, c, m, lab in [(f1,"#c0392b","o","B1  HBM colocated"),
                         (f2,"#2471a3","s","B2  HBF colocated"),
                         (f3,"#27ae60","^","B3  HBF disagg (serial)"),
                         (f4,"#8e44ad","D","B4  HBF disagg (2-way overlap)"),
                         (f5,"#e67e22","*",f"Ours  disagg+overlap+clustering ({int(CLUSTER_EFF*100)}% experts)")]:
        f = sorted(f, key=lambda p: p["gpus"])
        ax.plot([p["gpus"] for p in f], [p["tbt"] for p in f], m+"-", color=c, label=lab)
    ax.set_xlabel(f"total GPUs to serve {S} sessions  (cost →)")
    ax.set_ylabel("per-token latency TBT (ms)  (↓ better)")
    ax.set_title(f"Cost vs latency — serve {S}×128K decode\n"
                 f"DeepSeek-V3 · fp8 · serial · HBF BW = {hbf_bw:.0f} GB/s (HBM = {BW_HBM:.0f})")
    ax.set_xscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"cost-latency_serve-{S}x128k_fp8_hbfbw-{int(hbf_bw)}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print("  wrote", os.path.relpath(path))

    # companion: cost vs aggregate throughput (tok/s = S*1000 / TBT_ms)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for f, c, m, lab in [(f1,"#c0392b","o","B1  HBM colocated"),
                         (f2,"#2471a3","s","B2  HBF colocated"),
                         (f3,"#27ae60","^","B3  HBF disagg (serial)"),
                         (f4,"#8e44ad","D","B4  HBF disagg (2-way overlap)"),
                         (f5,"#e67e22","*",f"Ours  disagg+overlap+clustering ({int(CLUSTER_EFF*100)}% experts)")]:
        f = sorted(f, key=lambda p: p["gpus"])
        ax2.plot([p["gpus"] for p in f],
                 [S*1000.0/p["tbt"] for p in f], m+"-", color=c, label=lab)
    ax2.set_xlabel(f"total GPUs to serve {S} sessions  (cost →)")
    ax2.set_ylabel("aggregate decode throughput (tok/s)  (↑ better)")
    ax2.set_title(f"Cost vs throughput — serve {S}×128K decode\n"
                  f"DeepSeek-V3 · fp8 · HBF BW = {hbf_bw:.0f} GB/s (HBM = {BW_HBM:.0f})")
    ax2.set_xscale("log"); ax2.grid(alpha=0.3, which="both"); ax2.legend()
    fig2.tight_layout()
    path2 = os.path.join(FIG_DIR, f"cost-throughput_serve-{S}x128k_fp8_hbfbw-{int(hbf_bw)}.png")
    fig2.savefig(path2, dpi=130); plt.close(fig2)
    print("  wrote", os.path.relpath(path2))
    print(f"    B1 floor={min(p['gpus'] for p in f1)} | "
          f"B2 floor={min(p['gpus'] for p in f2)} | "
          f"B3 floor={min(p['gpus'] for p in f3)} | "
          f"B4 floor={min(p['gpus'] for p in f4)} GPUs")
    # TBT at a few matched GPU budgets, B2 vs B4
    def tbt_at(f, g):
        c=[p for p in f if p["gpus"]<=g]
        return min((p["tbt"] for p in c), default=None)
    for g in [8,16,32,64]:
        t2,t4,t5=tbt_at(f2,g),tbt_at(f4,g),tbt_at(f5,g)
        if t2 and t4 and t5:
            print(f"    @≤{g:3d} GPUs:  B2={t2:6.0f}  B4={t4:6.0f}  Ours={t5:6.0f} ms  "
                  f"(Ours {(1-t5/t2)*100:4.0f}% faster than B2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbf-bw", type=float, default=1024.0,
                    help="HBF flash bandwidth GB/s (default: matches HBM=1024)")
    ap.add_argument("--load", type=int, default=512, help="sessions to serve")
    args = ap.parse_args()

    print(f"HBF BW = {args.hbf_bw:.0f} GB/s   HBM BW = {BW_HBM:.0f} GB/s   "
          f"KV/session = {KV_SESS_GB:.1f} GB")
    fig_capacity(9.2, "fp16")           # capacity is BW-independent
    fig_capacity(KV_SESS_GB, "fp8")
    fig_cost(args.load, args.hbf_bw, "")
