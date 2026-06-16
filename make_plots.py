#!/usr/bin/env python3
"""ACM camera-ready plots from results/full_sweep.csv (Blackwell, H3 vs SPaKV).

Styled to match the FlashAccel paper: serif fonts, compact column widths,
grouped bars (Fig-13 style), vector PDF + 300-dpi PNG for LaTeX.

  1. Throughput improvement (SPaKV / H3) vs context — all models, one plot
  2. TPOT latency p50 / p99 vs context — H3 vs SPaKV
  3. Sustainable throughput/GPU under TPOT SLO (50 / 100 ms) — grouped bars
  4. Geomean SPaKV speedup per model
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, OUT = "results/full_sweep.csv", "results/plots"
os.makedirs(OUT, exist_ok=True)
DISP = {"Dense": "H3", "Sparse": "SPaKV"}
C_H3, C_SP = "#c44e52", "#4c72b0"          # muted ACM-ish red / blue

# ── ACM camera-ready style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 3.2,
    "grid.alpha": 0.35, "grid.linewidth": 0.4, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.borderpad": 0.3,
    "legend.handlelength": 1.6, "legend.columnspacing": 1.0,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)

rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "OK"]
tpot, tp = {}, {}
for r in rows:
    k = (r["model"], int(r["batch"]), int(r["context_length"]), r["baseline"])
    tpot[k] = float(r["tpot_p50_ms"]); tp[k] = int(r["tp"])
models   = sorted({r["model"] for r in rows})
batches  = sorted({int(r["batch"]) for r in rows})
contexts = sorted({int(r["context_length"]) for r in rows})
short = lambda m: (m.split("/")[-1].replace("-Instruct", "").replace("-v0.1", "")
                   .replace("Meta-", "").replace("-chat", ""))
def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ctx_label = lambda c: f"{c//1024}K" if c < 1<<20 else f"{c//(1<<20)}M"

# ── 1. Throughput improvement vs context (all models) ───────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.0))
colors = plt.cm.turbo(np.linspace(0.04, 0.96, len(models)))
markers = ["o","s","^","v","D","P","X","*","<",">","h","d","p"]
for i, (m, c) in enumerate(zip(models, colors)):
    xs, ys = [], []
    for ctx in contexts:
        sp = [tpot[(m,b,ctx,"Dense")]/tpot[(m,b,ctx,"Sparse")] for b in batches
              if (m,b,ctx,"Dense") in tpot and (m,b,ctx,"Sparse") in tpot]
        if sp: xs.append(ctx); ys.append(geomean(sp))
    if xs:
        ax.plot(xs, ys, marker=markers[i%len(markers)], color=c, lw=1.1,
                ms=3.2, label=short(m))
ax.axhline(10, ls="--", c="0.4", lw=0.8)
ax.text(contexts[0], 10.3, r"$1/s=10\times$ (flash-bound limit)", fontsize=7, color="0.4")
ax.set_xscale("log", base=2)
ax.set_xticks(contexts); ax.set_xticklabels([ctx_label(c) for c in contexts])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel(r"Throughput gain (SPaKV / H3)")
ax.legend(fontsize=6.5, ncol=4, loc="upper left", handletextpad=0.4)
ax.set_ylim(0.9, 11.5); ax.grid(True)
save(fig, "1_throughput_improvement")

# ── 2. Per-model TPOT p50 and p99 — H3 vs SPaKV (two plots, same style) ─────
# Per model, the percentile is taken over that model's feasible (batch x context)
# configs: p50 = typical operating point, p99 = heaviest (large batch, long ctx).
def pct(m, base, q):
    v = [tpot[k] for k in tpot if k[0] == m and k[3] == base]
    return np.percentile(v, q) if v else 0.0
order2 = sorted(models, key=lambda m: pct(m, "Sparse", 50))   # shared order, by SPaKV p50
def latency_bar(q, fname):
    y = np.arange(len(order2)); h = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    xm = max(pct(m, "Dense", q) for m in order2)
    for base, off, col, hatch in (("Dense", h/2, C_H3, "///"), ("Sparse", -h/2, C_SP, None)):
        vals = [pct(m, base, q) for m in order2]
        ax.barh(y+off, vals, h, color=col, edgecolor="k", lw=0.4, hatch=hatch, label=DISP[base])
        for j, vv in enumerate(vals):
            ax.text(vv+xm*0.012, y[j]+off, f"{vv:.0f}", va="center", fontsize=5.6,
                    color=(C_SP if base == "Sparse" else "k"))
    for slo in (50, 100):
        ax.axvline(slo, ls="--", c="0.4", lw=0.7)
        ax.text(slo, len(order2)-0.3, f"{slo} ms", rotation=90, fontsize=6, color="0.4", va="top", ha="right")
    ax.set_yticks(y); ax.set_yticklabels([short(m) for m in order2], fontsize=6.5)
    ax.set_xlim(0, xm*1.14)
    ax.set_xlabel(f"TPOT p{q} latency (ms)")
    ax.set_title(f"Per-model decode latency — TPOT p{q}  (H3 vs SPaKV)")
    ax.grid(True, axis="x"); ax.legend(loc="lower right")
    save(fig, fname)
latency_bar(50, "2a_tpot_p50")
latency_bar(99, "2b_tpot_p99")

# ── 3. Sustainable throughput/GPU under SLO at a fixed context (horiz bars) ──
# Fixed context so H3 and SPaKV are computed at the SAME point (comparable);
# throughput/GPU = (max batch meeting SLO) / TPOT / TP. A system that can't meet
# the SLO at any batch is marked "x SLO not met" rather than a blank bar.
CTX_FIX = 65536
def thru_per_gpu(m, base, slo):
    c = [b*1000.0/tpot[(m,b,CTX_FIX,base)]/tp[(m,b,CTX_FIX,base)] for b in batches
         if (m,b,CTX_FIX,base) in tpot and tpot[(m,b,CTX_FIX,base)] <= slo]
    return max(c) if c else 0.0
order3 = sorted(models, key=lambda m: thru_per_gpu(m,"Sparse",100))   # ascending -> big at top
y = np.arange(len(order3)); h = 0.38
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
for ax, slo in zip(axes, (50, 100)):
    h3  = [thru_per_gpu(m,"Dense", slo) for m in order3]
    spk = [thru_per_gpu(m,"Sparse",slo) for m in order3]
    ax.barh(y+h/2, h3,  h, color=C_H3, edgecolor="k", lw=0.4, hatch="///", label="H3")
    ax.barh(y-h/2, spk, h, color=C_SP, edgecolor="k", lw=0.4, label="SPaKV")
    xmax = max(spk + [1])
    for j, (vh, vs) in enumerate(zip(h3, spk)):
        if vh == 0: ax.text(xmax*0.012, y[j]+h/2, "SLO not met", va="center", fontsize=5.3, color=C_H3, style="italic")
        else:       ax.text(vh+xmax*0.01, y[j]+h/2, f"{vh:.0f}", va="center", fontsize=5.3)
        if vs == 0: ax.text(xmax*0.012, y[j]-h/2, "SLO not met", va="center", fontsize=5.3, color=C_SP, style="italic")
        else:       ax.text(vs+xmax*0.01, y[j]-h/2, f"{vs:.0f}", va="center", fontsize=5.3, color=C_SP)
    ax.set_xlim(0, xmax*1.18)
    ax.set_title(f"TPOT SLO = {slo} ms")
    ax.set_xlabel("Throughput / GPU (tokens/s)")
    ax.grid(True, axis="x"); ax.legend(loc="lower right")
axes[0].set_yticks(y); axes[0].set_yticklabels([short(m) for m in order3], fontsize=6.5)
fig.suptitle(f"Sustainable decode throughput per GPU under SLO  (context = {ctx_label(CTX_FIX)})",
             y=1.0, fontsize=9.5)
save(fig, "3_throughput_under_slo")

# ── 4. Throughput gain under 100 ms SLO vs context (per model) ──────────────
# Max sustainable throughput/GPU under the SLO = (best batch meeting SLO)/TPOT/TP.
# gain = SPaKV/H3 at each context. We EXCLUDE points where H3 only meets the SLO
# at batch=1 (weight-bound strawman -> explosive ratios); those models are listed
# as H3-can't-sustainably-serve. Clean curves climb toward the 1/s = 10x ceiling.
SLO4 = 100
def mt(m, c, base):
    cand = [(b, b*1000.0/tpot[(m,b,c,base)]/tp[(m,b,c,base)]) for b in batches
            if (m,b,c,base) in tpot and tpot[(m,b,c,base)] <= SLO4]
    return max(cand, key=lambda z: z[1]) if cand else None
ctx4 = [c for c in contexts if c <= 262144]             # stop at 256K
fig, ax = plt.subplots(figsize=(7.2, 3.4))
allg = []
for i, (m, col) in enumerate(zip(models, colors)):
    xs, ys = [], []
    for c in ctx4:
        h, s = mt(m, c, "Dense"), mt(m, c, "Sparse")
        if h and s and h[0] >= 2:                       # H3 sustains >= batch 2
            xs.append(c); ys.append(s[1]/h[1]); allg.append(s[1]/h[1])
    if xs:
        ax.plot(xs, ys, marker=markers[i%len(markers)], color=col, lw=1.2, ms=3.4, label=short(m))
g = math.exp(np.mean(np.log(allg)))
ax.set_xscale("log", base=2); ax.set_xticks(ctx4); ax.set_xticklabels([ctx_label(c) for c in ctx4])
ax.set_ylim(0, max(allg)*1.08)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Throughput gain under 100 ms SLO\n(SPaKV / H3)")
ax.set_title(f"Decode throughput gain under 100 ms SLO   (geomean {g:.1f}×)")
ax.grid(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5, ncol=1, borderaxespad=0.)
save(fig, "4_throughput_gain_slo")

# ── 5. SLO feasibility scatter: ours (SPaKV) vs dense (H3) TPOT ─────────────
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
def slo_scatter(slo, fname):
    xs, ys, cs = [], [], []
    for m in models:
        for b in batches:
            for c in contexts:
                kd, ks = (m,b,c,"Dense"), (m,b,c,"Sparse")
                if kd in tpot and ks in tpot:
                    xs.append(tpot[kd]); ys.append(tpot[ks]); cs.append(c)
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)
    both    = int(((xs<=slo) & (ys<=slo)).sum())
    neither = int(((xs> slo) & (ys> slo)).sum())
    ours    = int(((xs> slo) & (ys<=slo)).sum())
    dense   = int(((xs<=slo) & (ys> slo)).sum())
    lo, hi = min(xs.min(), ys.min())*0.8, max(xs.max(), ys.max())*1.2
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    ax.add_patch(Rectangle((slo, lo), hi-slo, slo-lo, facecolor="#f3e0b5", alpha=0.6, zorder=0))
    ax.plot([lo, hi], [lo, hi], ls="--", c="0.35", lw=0.9, zorder=1)          # parity
    ax.axvline(slo, c="0.35", lw=0.9); ax.axhline(slo, c="0.35", lw=0.9)
    sc = ax.scatter(xs, ys, c=cs, norm=LogNorm(vmin=4096, vmax=1048576), cmap="viridis",
                    s=16, alpha=0.8, edgecolor="none", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("H3 (dense) TPOT p50 (ms)"); ax.set_ylabel("SPaKV (ours) TPOT p50 (ms)")
    ax.set_title(f"All {len(xs)} configs vs the {slo} ms TPOT SLO  (Blackwell)")
    bb = dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5)
    ax.text(0.03, 0.04, f"Both meet SLO\nn={both}", transform=ax.transAxes, fontsize=7, va="bottom", bbox=bb)
    ax.text(0.97, 0.97, f"Neither\nn={neither}", transform=ax.transAxes, fontsize=7, ha="right", va="top", bbox=bb)
    ax.text(0.97, 0.04, f"Served only\nby ours\nn={ours}", transform=ax.transAxes, fontsize=7, ha="right", va="bottom", bbox=bb)
    ax.text(0.03, 0.97, "parity (y=x):\nbelow = ours faster", transform=ax.transAxes, fontsize=6.5, va="top")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Context length (K tokens)")
    cb.set_ticks([4096, 16384, 65536, 262144, 1048576]); cb.set_ticklabels([4, 16, 64, 256, 1024])
    save(fig, fname)
    return both, neither, ours, dense
s50 = slo_scatter(50, "5_slo_scatter_50ms")
s100 = slo_scatter(100, "5b_slo_scatter_100ms")

print("wrote plots (png+pdf) to", OUT,
      f"| throughput gain @100ms geomean={g:.1f}x max={max(allg):.1f}x (ctx<=256K)")
print(f"  SLO scatter @50ms (both,neither,ours-only,dense-only) = {s50}")
print(f"  SLO scatter @100ms = {s100}")
