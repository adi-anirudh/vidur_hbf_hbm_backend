#!/usr/bin/env python3
"""Held-out fixed-cohort no-pruning replay of normalized R1 routing.

Work is measured only in indivisible expert-token jobs. This script does not
model latency, throughput, attention, network traffic, or combined fetch cost.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PAPER / "experiments/deepseek_r1_awq_no_prune_replay"
LAYERS, EXPERTS, TOPK = 58, 256, 8
DEFAULT_STEPS = tuple(sorted(set(int(x) for x in np.linspace(0, 127, 16).round())))
DEFAULT_WORKLOADS = ("Chinese-SimpleQA", "livecodebench", "mmlu", "mmlu_ZH_CN", "mixed")


def ints(text, allow_zero=False):
    values = tuple(int(x) for x in text.split(",") if x.strip())
    if not values or any(x < (0 if allow_zero else 1) for x in values):
        raise argparse.ArgumentTypeError("invalid integer list")
    return tuple(sorted(set(values)))


def strings(text):
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty list")
    return values


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sessions", type=ints, default=(8, 16, 32, 64, 128))
    p.add_argument("--microbatch-count", type=int, default=2)
    p.add_argument("--moe-gpus", type=ints, default=(8, 16, 32))
    p.add_argument("--workloads", type=strings, default=DEFAULT_WORKLOADS)
    p.add_argument("--decode-steps", type=lambda x: ints(x, True), default=DEFAULT_STEPS)
    p.add_argument("--write-event-trace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-cohorts", type=int, default=8)
    p.add_argument("--seed", default="flexep-r1-fixed-cohort-v1")
    return p.parse_args()


def load(root):
    rows = list(csv.DictReader(open(root / "manifests/requests.csv", encoding="utf-8")))
    meta = {int(r["request_index"]): r for r in rows}
    if sorted(meta) != list(range(len(meta))):
        raise ValueError("non-contiguous request manifest")
    profiles = defaultdict(lambda: np.zeros((LAYERS, EXPERTS), np.int64))
    profile_n, test = defaultdict(int), {}
    seen = []
    for path in sorted((root / "normalized/shards").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            ids, off = z["expert_ids"], z["request_offsets"]
            buckets = defaultdict(list)
            for j, raw_index in enumerate(z["request_indices"]):
                i = int(raw_index); seen.append(i); r = meta[i]
                start = int(off[j]) + int(z["prompt_lengths"][j])
                route = ids[start:start + int(z["decode_lengths"][j])]
                if route.shape != (int(r["valid_decode_tokens"]), LAYERS, TOPK):
                    raise ValueError(f"shape mismatch request {i}")
                if r["split"] == "train":
                    buckets[r["workload"]].append(route); profile_n[r["workload"]] += 1
                else:
                    test[i] = route.copy()
            for workload, arrays in buckets.items():
                data = np.concatenate(arrays)
                for layer in range(LAYERS):
                    profiles[workload][layer] += np.bincount(
                        data[:, layer].reshape(-1), minlength=EXPERTS)
    if seen != list(range(len(meta))):
        raise ValueError("shard request ordering mismatch")
    profiles["mixed"] = np.sum(list(profiles.values()), axis=0)
    profile_n["mixed"] = sum(profile_n.values())
    return meta, dict(profiles), dict(profile_n), test


def static_owners(profile, m):
    if EXPERTS % m:
        raise ValueError("MoE GPU count must divide 256")
    cap = EXPERTS // m
    left, loads = np.full(m, cap), np.zeros(m, np.int64)
    owners = np.full(EXPERTS, -1, np.int16)
    for expert in np.lexsort((np.arange(EXPERTS), -profile)):
        candidates = np.flatnonzero(left)
        gpu = int(candidates[np.argmin(loads[candidates])])
        owners[expert] = gpu; left[gpu] -= 1; loads[gpu] += int(profile[expert])
    assert np.all(np.bincount(owners, minlength=m) == cap)
    return owners


def static_event(counts, owners, m):
    active = np.flatnonzero(counts); assigned = owners[active]
    loads = np.bincount(assigned, weights=counts[active], minlength=m)
    fetch = np.bincount(assigned, minlength=m)
    return int(loads.max()), int(fetch.max())


def assign_event(counts, m, score=None, previous=None):
    active = np.flatnonzero(counts); owners = np.full(EXPERTS, -1, np.int16)
    if score is None:
        score = counts.astype(float)
    else:
        score = np.asarray(score, float)
    score_loads = np.zeros(m, float)
    token_loads, fetch = np.zeros(m, np.int32), np.zeros(m, np.int32)
    for expert in active[np.lexsort((active, -counts[active], -score[active]))]:
        candidates = np.flatnonzero(np.isclose(score_loads, score_loads.min()))
        old = -1 if previous is None else int(previous[expert])
        gpu = old if old in candidates else int(candidates[0])
        owners[expert] = gpu
        score_loads[gpu] += float(score[expert])
        token_loads[gpu] += int(counts[expert])
        fetch[gpu] += 1
    churn = None
    if previous is not None:
        common = active[previous[active] >= 0]
        if common.size: churn = float(np.mean(owners[common] != previous[common]))
    return int(token_loads.max()), int(fetch.max()), owners, churn, float(score_loads.max())


def dynamic_event(counts, m, previous=None):
    token_load, fetch, owners, churn, _ = assign_event(counts, m, previous=previous)
    return token_load, fetch, owners, churn


def active_count_event(counts, m, previous=None):
    score = np.where(counts > 0, 1.0, 0.0)
    return assign_event(counts, m, score=score, previous=previous)


def cost_event(counts, m, previous=None):
    score = np.where(counts > 0, counts.astype(float) + 1.0, 0.0)
    return assign_event(counts, m, score=score, previous=previous)


def order(indices, meta, label, seed):
    return sorted(indices, key=lambda i: hashlib.sha256(
        f"{seed}\0{label}\0{meta[i]['relative_path']}".encode()).digest())


def pct(values, q): return float(np.percentile(np.asarray(values, float), q))


def event_fieldnames():
    return [
        "workload", "sessions", "microbatch_count", "sessions_per_microbatch",
        "cohort", "microbatch", "decode_step", "layer", "moe_gpus",
        "active_sessions", "total_routed_expert_tokens", "active_experts",
        "hottest_expert_tokens", "expert_token_skew", "lower_bound_token_count",
        "static_max_gpu_tokens", "static_max_fetched_experts",
        "active_count_dynamic_max_gpu_tokens",
        "active_count_dynamic_max_fetched_experts",
        "token_count_dynamic_max_gpu_tokens",
        "token_count_dynamic_max_fetched_experts",
        "cost_dynamic_max_gpu_tokens", "cost_dynamic_max_fetched_experts",
        "cost_dynamic_max_score", "token_count_dynamic_churn",
    ]


def summarize(a, workload, s, mu, m, cohorts, available, profiled):
    static, dynamic, lower = map(lambda k: np.asarray(a[k], float), ("static", "dynamic", "lower"))
    sp, dp = pct(static, 95), pct(dynamic, 95)
    return {
        "workload": workload, "sessions": s, "microbatch_count": mu,
        "initial_sessions_per_microbatch": s // mu, "moe_gpus": m,
        "cohorts": cohorts, "test_requests_available": available,
        "profile_train_requests": profiled, "layer_events": len(static),
        "active_sessions_mean": float(np.mean(a["active_sessions"])),
        "active_experts_mean": float(np.mean(a["active_experts"])),
        "active_experts_p95": pct(a["active_experts"], 95),
        "max_expert_tokens_p95": pct(a["hottest"], 95),
        "expert_token_skew_mean": float(np.mean(a["skew"])),
        "static_max_gpu_tokens_mean": float(np.mean(static)),
        "dynamic_max_gpu_tokens_mean": float(np.mean(dynamic)),
        "static_max_gpu_tokens_p95": sp, "dynamic_max_gpu_tokens_p95": dp,
        "mean_load_reduction_pct": float((np.mean(static)-np.mean(dynamic))/np.mean(static)*100),
        "p95_load_reduction_pct": float((sp-dp)/sp*100),
        "dynamic_slower_event_fraction": float(np.mean(dynamic > static)),
        "dynamic_gap_to_lower_bound_p95": pct(dynamic/lower, 95),
        "static_max_fetched_experts_p95": pct(a["static_fetch"], 95),
        "dynamic_max_fetched_experts_p95": pct(a["dynamic_fetch"], 95),
        "dynamic_reassignment_fraction_mean": float(np.mean(a["churn"])) if a["churn"] else 0.0,
    }


def replay(args, meta, profiles, profile_n, test):
    if args.microbatch_count < 1 or args.max_cohorts < 1:
        raise ValueError("invalid microbatch/cohort count")
    if any(s % args.microbatch_count for s in args.sessions) or max(args.decode_steps) > 127:
        raise ValueError("invalid sessions or decode steps")
    test_by = defaultdict(list)
    for i, r in meta.items():
        if r["split"] == "test": test_by[r["workload"]].append(i)
    test_by["mixed"] = sorted(test)
    placements = {(w,m): np.stack([static_owners(profiles[w][l], m) for l in range(LAYERS)])
                  for w in args.workloads if w in profiles for m in args.moe_gpus}
    event_stream = None
    event_writer = None
    if args.write_event_trace:
        result_dir = args.experiment_dir.resolve() / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        event_stream = open(result_dir / "layer_events.csv", "w", newline="", encoding="utf-8")
        event_writer = csv.DictWriter(event_stream, fieldnames=event_fieldnames())
        event_writer.writeheader()
    result = []
    keys = ("active_sessions","active_experts","hottest","skew","static","dynamic",
            "lower","static_fetch","dynamic_fetch","churn")
    for w in args.workloads:
        available = test_by.get(w, [])
        if w not in profiles: continue
        for s in args.sessions:
            ncohort = min(args.max_cohorts, len(available)//s)
            if not ncohort:
                print(f"skip {w} S={s}: {len(available)} test requests", flush=True); continue
            chosen = order(available, meta, f"{w}:S={s}", args.seed)[:ncohort*s]
            acc = {m:{k:[] for k in keys} for m in args.moe_gpus}
            b = s//args.microbatch_count
            for c in range(ncohort):
                cohort = chosen[c*s:(c+1)*s]
                for mb_i, mb in enumerate(cohort[x:x+b] for x in range(0,s,b)):
                    previous = {
                        name: {m:np.full((LAYERS,EXPERTS),-1,np.int16) for m in args.moe_gpus}
                        for name in ("active_count_dynamic", "token_count_dynamic", "cost_dynamic")
                    }
                    for step in args.decode_steps:
                        active = [i for i in mb if len(test[i]) > step]
                        if not active: continue
                        selected = np.stack([test[i][step] for i in active])
                        for layer in range(LAYERS):
                            counts = np.bincount(selected[:,layer].reshape(-1), minlength=EXPERTS).astype(np.int16)
                            ne, total, hot = np.count_nonzero(counts), int(counts.sum()), int(counts.max())
                            skew = hot/(total/ne)
                            for m in args.moe_gpus:
                                st, sf = static_event(counts, placements[(w,m)][layer], m)
                                ac, af, ac_own, _, _ = active_count_event(
                                    counts, m, previous["active_count_dynamic"][m][layer])
                                dy, df, own, churn = dynamic_event(
                                    counts, m, previous["token_count_dynamic"][m][layer])
                                cy, cf, cost_own, _, cscore = cost_event(
                                    counts, m, previous["cost_dynamic"][m][layer])
                                previous["active_count_dynamic"][m][layer] = ac_own
                                previous["token_count_dynamic"][m][layer] = own
                                previous["cost_dynamic"][m][layer] = cost_own
                                a = acc[m]
                                lower = max(math.ceil(total/m),hot)
                                values = (len(active),ne,hot,skew,st,dy,
                                          lower,sf,df,churn)
                                for k,v in zip(keys,values):
                                    if v is not None: a[k].append(v)
                                if event_writer is not None:
                                    event_writer.writerow({
                                        "workload": w,
                                        "sessions": s,
                                        "microbatch_count": args.microbatch_count,
                                        "sessions_per_microbatch": b,
                                        "cohort": c,
                                        "microbatch": mb_i,
                                        "decode_step": step,
                                        "layer": layer,
                                        "moe_gpus": m,
                                        "active_sessions": len(active),
                                        "total_routed_expert_tokens": total,
                                        "active_experts": ne,
                                        "hottest_expert_tokens": hot,
                                        "expert_token_skew": skew,
                                        "lower_bound_token_count": lower,
                                        "static_max_gpu_tokens": st,
                                        "static_max_fetched_experts": sf,
                                        "active_count_dynamic_max_gpu_tokens": ac,
                                        "active_count_dynamic_max_fetched_experts": af,
                                        "token_count_dynamic_max_gpu_tokens": dy,
                                        "token_count_dynamic_max_fetched_experts": df,
                                        "cost_dynamic_max_gpu_tokens": cy,
                                        "cost_dynamic_max_fetched_experts": cf,
                                        "cost_dynamic_max_score": cscore,
                                        "token_count_dynamic_churn": "" if churn is None else churn,
                                    })
            for m in args.moe_gpus:
                result.append(summarize(acc[m],w,s,args.microbatch_count,m,ncohort,
                                        len(available),profile_n[w]))
            print(f"finished {w} S={s} cohorts={ncohort}", flush=True)
    if event_stream is not None:
        event_stream.close()
    return result


def outputs(root, args, rows):
    result, figures = root/"results", root/"figures"
    result.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    with open(result/"load_balance_summary.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mixed=sorted((r for r in rows if r["workload"]=="mixed"),key=lambda r:(r["moe_gpus"],r["sessions"]))
    lines=["# DeepSeek-R1 held-out no-pruning replay","",
      "Units are expert-token calls, not latency, throughput, or TBT.","",
      "| GPUs | S | Sessions/MB | Static p95 | Dynamic p95 | Reduction | Dynamic/LB p95 | Churn |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in mixed: lines.append(f"| {r['moe_gpus']} | {r['sessions']} | {r['initial_sessions_per_microbatch']} | {r['static_max_gpu_tokens_p95']:.1f} | {r['dynamic_max_gpu_tokens_p95']:.1f} | {r['p95_load_reduction_pct']:.2f}% | {r['dynamic_gap_to_lower_bound_p95']:.3f} | {100*r['dynamic_reassignment_fraction_mean']:.2f}% |")
    lines += ["","AIME is included only in mixed (seven test requests). LiveCodeBench S=128 is unavailable. Churn is between sampled decode steps.",""]
    (result/"load_balance_summary.md").write_text("\n".join(lines),encoding="utf-8")
    gpu=sorted({r["moe_gpus"] for r in rows}); fig,axes=plt.subplots(1,len(gpu),figsize=(4.4*len(gpu),3.7),sharey=True); axes=np.atleast_1d(axes)
    for ax,m in zip(axes,gpu):
        data=sorted((r for r in mixed if r["moe_gpus"]==m),key=lambda r:r["sessions"]); x=[r["sessions"] for r in data]
        ax.plot(x,[r["static_max_gpu_tokens_p95"] for r in data],"o-",color="#777777",label="Profile-fixed EP")
        ax.plot(x,[r["dynamic_max_gpu_tokens_p95"] for r in data],"s-",color="#0072B2",label="Dynamic EP"); ax.set_title(f"{m} MoE GPUs"); ax.set_xlabel("Sessions (S)"); ax.grid(axis="y",alpha=.25)
    axes[0].set_ylabel("p95 slowest-GPU expert-token calls"); axes[-1].legend(frameon=False); fig.tight_layout()
    fig.savefig(figures/"fig1_p95_slowest_gpu_work.png",dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,len(gpu),figsize=(4.4*len(gpu),3.7),sharey=True); axes=np.atleast_1d(axes)
    for ax,m in zip(axes,gpu):
        for w in args.workloads:
            data=sorted((r for r in rows if r["workload"]==w and r["moe_gpus"]==m),key=lambda r:r["sessions"])
            if data: ax.plot([r["sessions"] for r in data],[r["p95_load_reduction_pct"] for r in data],"o-",label=w)
        ax.axhline(0,color="black",lw=.7); ax.set_title(f"{m} MoE GPUs"); ax.set_xlabel("Sessions (S)"); ax.grid(axis="y",alpha=.25)
    axes[0].set_ylabel("p95 load reduction (%)"); axes[-1].legend(frameon=False,fontsize=7); fig.tight_layout()
    fig.savefig(figures/"fig2_p95_reduction_by_workload.png",dpi=220)
    plt.close(fig)
    config={"sessions":list(args.sessions),"microbatch_count":args.microbatch_count,"moe_gpus":list(args.moe_gpus),"workloads":list(args.workloads),"decode_steps":list(args.decode_steps),"max_cohorts":args.max_cohorts,"seed":args.seed,"work_unit":"expert-token call","static":"train-profiled equal-capacity LPT","dynamic":"current-microbatch stability-aware LPT; experts indivisible","no_pruning":True,"latency_modeled":False}
    (root/"run_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")


def main():
    args=arguments(); root=args.experiment_dir.resolve()
    print("loading shards and building train-only profiles",flush=True)
    meta,profiles,profile_n,test=load(root); print(f"loaded {len(meta)} requests; test={len(test)}",flush=True)
    rows=replay(args,meta,profiles,profile_n,test)
    if not rows: raise RuntimeError("no results")
    outputs(root,args,rows); print(f"wrote {len(rows)} result rows",flush=True)

if __name__ == "__main__": main()
