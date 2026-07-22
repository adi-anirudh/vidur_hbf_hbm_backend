#!/usr/bin/env python3
"""Build results/norm_throughput.json for the result plot: per-GPU decode throughput
at the 100 ms TPOT SLO, normalized to SPLASH (=1.0), for a curated logical sweep of
architecturally-unique models x all 8 contexts (128K..2M). No GeoMean."""
import csv, json
from collections import defaultdict

rows = [r for r in csv.DictReader(open("results/sweep_b200.csv")) if r["status"] == "OK"]
SLO = 100.0
best = defaultdict(lambda: None)
for r in rows:
    t = float(r["tpot_p50_ms"])
    if t > SLO:
        continue
    thr = int(r["batch"]) * 1000.0 / t / int(r["tp"])
    k = (r["model"], int(r["context_length"]), r["baseline"])
    if best[k] is None or thr > best[k]:
        best[k] = thr

# curated size x architecture sweep (small -> huge; dense/MoE; MHA/GQA-8/GQA-4)
MODELS = [
    ("microsoft/phi-2",                  "Phi-2"),
    ("meta-llama/Meta-Llama-3-8B",       "Llama-3-8B"),
    ("mistralai/Mixtral-8x7B-v0.1",      "Mixtral-8×7B"),
    ("meta-llama/Meta-Llama-3-70B",      "Llama-3-70B"),
    ("Qwen/Qwen3-235B-A22B",             "Qwen3-235B"),
    ("meta-llama/Meta-Llama-3.1-405B",   "Llama-3.1-405B"),
]
SYS = [("HBM-only", "HBM-only"), ("Dense", "H3"), ("Naive", "Naive"), ("Sparse", "SPLASH")]
CTX = [(131072, "128K"), (196608, "192K"), (262144, "256K"), (393216, "384K"),
       (524288, "512K"), (786432, "768K"), (1048576, "1M"), (2097152, "2M")]

work = []
for mk, md in MODELS:
    for ck, cd in CTX:
        sp = best[(mk, ck, "Sparse")]
        if not sp:
            work.append(dict(model=md, ctxlabel=cd, vals={sd: None for _, sd in SYS},
                             infeasible=[sd for _, sd in SYS]))
            continue
        vals, infeas = {}, []
        for sk, sd in SYS:
            v = best[(mk, ck, sk)]
            vals[sd] = (v / sp) if v is not None else None
            if v is None:
                infeas.append(sd)
        work.append(dict(model=md, ctxlabel=cd, vals=vals, infeasible=infeas))

out = dict(slo_ms=SLO, systems=[sd for _, sd in SYS],
           models=[md for _, md in MODELS], ctxs=[cd for _, cd in CTX], workloads=work)
json.dump(out, open("results/norm_throughput.json", "w"))
present = sum(1 for w in work if any(v is not None for v in w["vals"].values()))
print(f"workloads: {len(work)} ({present} with data)  models={out['models']}")
for w in work:
    print(f"  {w['model']:15s} {w['ctxlabel']:>5}: " +
          " ".join(f"{s}={'x' if w['vals'][s] is None else format(w['vals'][s], '.2f')}" for s in out["systems"]))
