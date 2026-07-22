#!/usr/bin/env python3
"""Energy figure (Fig-12 style) from the Blackwell all-baselines sweep.

Normalized tokens/J to SPLASH (Ours) = 1, per model, across ALL swept contexts.
Bars: H3 (dense, full KV), Naive (token-granular, page-amplified), SPLASH (Ours,
page+plane sparse). Infeasible operating points (no batch meets the TPOT-p50 SLO)
are drawn as red x x x. HBM-only omitted (infeasible at these contexts).
Constants: HBM3e 2.99 pJ/bit [4], hybrid-bonded Flash 8.0 pJ/bit [62],
B200 0.89 pJ/FLOP, Naive read-amp 3.5636. MoE: total weights -> weight-read,
active params -> compute FLOPs. Self-contained.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HBM_PJ_PER_BIT = 2.99
HBF_PJ_PER_BIT = 8.0
PJ_PER_FLOP    = 0.889
NAIVE_AMP      = 3.5636
SPARSE_OVERHEAD = 0.05
SLO_MS = 100.0
ROOT = "/home/adityaan/vidur_hbf_hbm_backend"

MI = {
    "meta-llama/Meta-Llama-3-8B":        dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=16.0,  active_gb=16.0),
    "mistralai/Mistral-7B-v0.1":         dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=14.0,  active_gb=14.0),
    "deepseek-ai/deepseek-llm-67b-chat": dict(layers=95,  kv_heads=8,  head_dim=128, weights_gb=134.0, active_gb=134.0),
    "meta-llama/Meta-Llama-3-70B":       dict(layers=80,  kv_heads=8,  head_dim=128, weights_gb=140.0, active_gb=140.0),
    "Qwen/Qwen2-72B":                    dict(layers=80,  kv_heads=8,  head_dim=128, weights_gb=145.0, active_gb=145.0),
    "meta-llama/Meta-Llama-3.1-405B":    dict(layers=126, kv_heads=8,  head_dim=128, weights_gb=810.0, active_gb=810.0),
    "mistralai/Mixtral-8x7B-v0.1":       dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=93.0,  active_gb=26.0),
    "mistralai/Mixtral-8x22B-v0.1":      dict(layers=56,  kv_heads=8,  head_dim=128, weights_gb=282.0, active_gb=78.0),
    "Qwen/Qwen3-235B-A22B":              dict(layers=94,  kv_heads=4,  head_dim=128, weights_gb=470.0, active_gb=44.0),
}


def per_token_pJ(model, batch, ctx, kv_rf):
    m = MI[model]
    kv_per_tok = 2 * m["kv_heads"] * m["head_dim"] * 2
    hbf_kv = batch * ctx * kv_per_tok * m["layers"] * kv_rf
    weight_bytes = m["weights_gb"] * 1e9
    e_hbm = weight_bytes * 8 * HBM_PJ_PER_BIT
    e_hbf = hbf_kv * 8 * HBF_PJ_PER_BIT
    active_params = m["active_gb"] * 1e9 / 2
    flops = 2.0 * active_params * batch * (1 + (SPARSE_OVERHEAD if kv_rf < 1 else 0))
    e_comp = flops * PJ_PER_FLOP
    return (e_hbm + e_hbf + e_comp) / batch


KRF = {"H3": 1.0, "Naive": 0.1 * NAIVE_AMP, "Sparse": 0.1}
SWEEP_BL = {"H3": "Dense", "Naive": "Naive", "Sparse": "Sparse"}   # our label -> sweep baseline

rows = [r for r in csv.DictReader(open(ROOT + "/results/sweep_b200.csv")) if r.get("status") == "OK"]
feas, anyb = {}, {}
for r in rows:
    if r["model"] not in MI:
        continue
    try:
        tp50 = float(r["tpot_p50_ms"]); b = int(r["batch"]); ctx = int(r["context_length"])
    except (ValueError, KeyError):
        continue
    k = (r["model"], ctx, r["baseline"])
    anyb[k] = max(anyb.get(k, 0), b)
    if tp50 <= SLO_MS:
        feas[k] = max(feas.get(k, 0), b)


def op_point(model, ctx, our_bl):
    k = (model, ctx, SWEEP_BL[our_bl])
    if k in feas:
        return feas[k], True
    if k in anyb:
        return anyb[k], False
    return None, False


def tokJ(model, ctx, our_bl, batch):
    e = per_token_pJ(model, batch, ctx, KRF[our_bl])
    return 1e12 / e if e > 0 else None


MODELS = [("Qwen/Qwen3-235B-A22B", "Qwen3-235B"),
          ("mistralai/Mixtral-8x22B-v0.1", "Mixtral-8x22B"),
          ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
          ("meta-llama/Meta-Llama-3.1-405B", "Llama3.1-405B")]
CTXS = [(131072, "128K"), (196608, "192K"), (262144, "256K"), (393216, "384K"),
        (524288, "512K"), (786432, "768K"), (1048576, "1M"), (2097152, "2M")]
BARS = [("H3", "#b07a3f", None), ("Naive", "#e0902e", None), ("Sparse", "#4a4a8a", "////")]
LBL = {"H3": "H3", "Naive": "Naive Sparse", "Sparse": "SPLASH (Ours)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "ytick.labelsize": 10,
    "legend.fontsize": 11, "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.dpi": 200, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(15.0, 2.05))
bw = 0.085
cgap = 0.05
mgap = 0.5
x = 0.0
xt, xtl, gc, gl, seen, printed = [], [], [], [], set(), []
for mkey, mlbl in MODELS:
    g0 = x
    for ctx, clbl in CTXS:
        sb, s_feas = op_point(mkey, ctx, "Sparse")
        ref = tokJ(mkey, ctx, "Sparse", sb) if sb else None
        for j, (bl, col, hatch) in enumerate(BARS):
            b, feasible = op_point(mkey, ctx, bl)
            xp = x + j * bw
            if feasible and ref:
                norm = tokJ(mkey, ctx, bl, b) / ref
                lab = LBL[bl] if bl not in seen else None; seen.add(bl)
                ax.bar(xp, norm, bw, color=col, edgecolor="black", lw=0.35,
                       label=lab, hatch=hatch)
                printed.append((mlbl, clbl, bl, norm, True))
            else:
                ax.text(xp, 0.06, "x\nx\nx", ha="center", va="bottom",
                        fontsize=5.5, color="#d11", fontweight="bold", linespacing=0.85)
                printed.append((mlbl, clbl, bl, float("nan"), False))
        xt.append(x + bw); xtl.append(clbl)
        x += 3 * bw + cgap
    gc.append((g0 + x - 3 * bw - cgap) / 2 + bw); gl.append(mlbl)
    x += mgap

for y in (0.5, 1.0):
    ax.axhline(y, color="#cfcfcf", ls="--", lw=0.6, zorder=0)
ax.set_ylabel("Normalized\ntokens/J")
ax.set_xticks(xt); ax.set_xticklabels(xtl, rotation=90, fontsize=6.5)
ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0])
for c, l in zip(gc, gl):
    ax.text(c, -0.62, l, ha="center", va="top", fontsize=9.5)
ax.margins(x=0.004)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False,
          handlelength=1.4, columnspacing=2.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.subplots_adjust(left=0.06, right=0.997, top=0.78, bottom=0.30)
for ext in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figures/energy_normalized.{ext}", bbox_inches="tight", facecolor="white")
print("wrote figures/energy_normalized.png / .pdf   (normalized to SPLASH=1)")
for m, c, bl, n, f in printed:
    print(f"  {m:14} {c:>4} {bl:7} {('xxx' if not f else f'{n:.2f}')}")
