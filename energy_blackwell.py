"""Energy from the Blackwell HBF sweep (full_sweep.csv), using the real
characterization constants: HBM3e 2.99 pJ/bit [4], hybrid-bonded Flash 8 pJ/bit
[62], B200 ~0.89 pJ/FLOP. tokens/J for Dense (H3, full KV read) vs Sparse
(SPLASH, 10% KV read), each at its largest batch meeting the TPOT-p50 SLO.
KV lives on HBF (hbm_frac=0); weights on HBM3e."""
import csv, math
from collections import defaultdict
from analyze_paper_results import energy_per_output_token_pJ, MODEL_INFO

SLO = 100.0        # ms, TPOT p50
HBM_FRAC = 0.0     # KV on HBF; weights on HBM
KRF = {"Dense": 1.0, "Sparse": 0.1}

rows = [r for r in csv.DictReader(open("results/full_sweep.csv")) if r.get("status") == "OK"]
best = {}   # (model,ctx,baseline) -> (batch,row) at max batch meeting SLO
for r in rows:
    try:
        tp50 = float(r["tpot_p50_ms"]); b = int(r["batch"]); ctx = int(r["context_length"])
    except (ValueError, KeyError):
        continue
    if tp50 > SLO:
        continue
    k = (r["model"], ctx, r["baseline"])
    if k not in best or b > best[k][0]:
        best[k] = (b, r)

def tokJ(model, batch, ctx, bl):
    e, _ = energy_per_output_token_pJ(model, "blackwell", batch, ctx, HBM_FRAC, KRF[bl])
    return (1e12 / e) if e and e > 0 else None

def geomean(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")

by_ctx = defaultdict(lambda: {"dense": [], "sparse": [], "ratio": []})
covered, skipped = set(), set()
for (model, ctx, bl) in list(best):
    if bl != "Dense":
        continue
    if model not in MODEL_INFO:
        skipped.add(model); continue
    if (model, ctx, "Sparse") not in best:
        continue
    bd, _ = best[(model, ctx, "Dense")]; bs, _ = best[(model, ctx, "Sparse")]
    td = tokJ(model, bd, ctx, "Dense"); ts = tokJ(model, bs, ctx, "Sparse")
    if td and ts:
        by_ctx[ctx]["dense"].append(td); by_ctx[ctx]["sparse"].append(ts)
        by_ctx[ctx]["ratio"].append(ts / td); covered.add(model)

print(f"covered models: {len(covered)}  | skipped (no MODEL_INFO): {sorted(skipped)}")
print(f"{'ctx':>8} {'n':>3} {'dense tok/J':>12} {'SPLASH tok/J':>13} {'energy ratio':>12}")
for ctx in sorted(by_ctx):
    d = by_ctx[ctx]
    print(f"{ctx:>8} {len(d['ratio']):>3} {geomean(d['dense']):>12.1f} "
          f"{geomean(d['sparse']):>13.1f} {geomean(d['ratio']):>11.2f}x")
allr = [x for c in by_ctx.values() for x in c["ratio"]]
print(f"\noverall SPLASH/Dense energy-efficiency geomean = {geomean(allr):.2f}x  (n={len(allr)})")
