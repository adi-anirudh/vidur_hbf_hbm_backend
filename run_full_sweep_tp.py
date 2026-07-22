#!/usr/bin/env python3
"""TP-swept HBF sweep: for every (model,device,batch,ctx,baseline) run ALL feasible
TP in {1,2,4,8} (not just min), so the SLO operating point at higher TP is measured.
Seeds from results/full_sweep.csv (min-TP rows) and only runs the extra higher-TP
points. Streams to results/full_sweep_tp.csv (resumable; job key includes tp)."""
from __future__ import annotations
import csv, re, subprocess, sys, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from functools import lru_cache
from sweep_capacity import weight_bytes, HBM_GB, HBF_GB
from vidur.config.model_config import BaseModelConfig

ROOT=Path(__file__).parent; PY=sys.executable
RUNNER=str(ROOT/"run_point_hbf.py"); TOML=str(ROOT/"configs/hbf_paper.toml")
SRC=ROOT/"results"/"full_sweep.csv"; OUT=ROOT/"results"/"full_sweep_tp.csv"
MODELS=["microsoft/phi-2","mistralai/Mistral-7B-v0.1","meta-llama/Meta-Llama-3-8B",
 "mistralai/Mixtral-8x7B-v0.1","deepseek-ai/deepseek-llm-67b-chat","meta-llama/Meta-Llama-3-70B",
 "Qwen/Qwen-72B","Qwen/Qwen2-72B","meta-llama/Meta-Llama-3.1-405B",
 "meta-llama/Llama-3.3-70B-Instruct","mistralai/Mixtral-8x22B-v0.1",
 "Qwen/Qwen3-235B-A22B","Qwen/Qwen3-Coder-480B-A35B-Instruct"]
DEVICES=["blackwell"]; BATCHES=[1,2,4,8,16,32,64,128]
CONTEXTS=[4096,8192,16384,32768,65536,131072,262144,524288,1048576]
BASELINES=[("Dense",1.0),("Sparse",0.1)]
FIELDS=["model","device","batch","context_length","baseline","tp","footprint_gb",
        "sparsity_fraction","tpot_p50_ms","tpot_p99_ms","status","wall_s"]

@lru_cache(maxsize=None)
def _dimcache(model):
    c=BaseModelConfig.create_from_name(model)
    tp_cap=1 if getattr(c,"no_tensor_parallel",False) else min(8,c.num_kv_heads)
    return c.num_layers, c.num_kv_heads, c.head_size(), tp_cap
def feasible_tps(model,device,batch,ctx):
    n_lay,n_kv,head_dim,tp_cap=_dimcache(model)
    total=weight_bytes(model)+batch*ctx*n_lay*2*n_kv*head_dim*2
    cap=(HBM_GB[device]+HBF_GB)*1e9
    return [tp for tp in (1,2,4,8) if tp<=tp_cap and total<=tp*cap], total/1e9

def run_point(model,device,batch,ctx,tp,label,sp,timeout_s):
    cmd=[PY,RUNNER,"--model",model,"--device",device,"--batch_size",str(batch),
         "--context_length",str(ctx),"--decode_tokens","4","--num_requests",str(batch),
         "--tensor_parallel_size",str(tp),"--sparsity_fraction",str(sp),"--hbfsim_toml",TOML,
         "--cache_dir",str(ROOT/"pred_cache")]
    t0=time.perf_counter()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout_s,cwd=str(ROOT))
        wall=time.perf_counter()-t0
        m=re.search(r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)",p.stdout)
        if m: return dict(tpot_p50_ms=m.group(1),tpot_p99_ms=m.group(2),status="OK",wall_s=f"{wall:.1f}")
        return dict(tpot_p50_ms="",tpot_p99_ms="",status=f"FAIL({p.returncode})",wall_s=f"{wall:.1f}")
    except subprocess.TimeoutExpired:
        return dict(tpot_p50_ms="",tpot_p99_ms="",status=f"TIMEOUT>{timeout_s}s",wall_s="")

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--timeout",type=int,default=600); ap.add_argument("--limit",type=int,default=0); args=ap.parse_args()
    if not OUT.exists() and SRC.exists(): shutil.copy(SRC,OUT)   # seed with min-TP rows
    done_ok=set(); done_inf=set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            if r["status"]=="OK" and r["tp"]:
                done_ok.add((r["model"],r["device"],int(r["batch"]),int(r["context_length"]),r["baseline"],int(r["tp"])))
            elif r["status"]=="INFEASIBLE":
                done_inf.add((r["model"],r["device"],int(r["batch"]),int(r["context_length"]),r["baseline"]))
    fh=open(OUT,"a",newline=""); w=csv.DictWriter(fh,fieldnames=FIELDS)
    runnable=[]
    for model in MODELS:
        for device in DEVICES:
            for batch in BATCHES:
                for ctx in CONTEXTS:
                    tps,foot=feasible_tps(model,device,batch,ctx)
                    for label,sp in BASELINES:
                        if not tps:
                            if (model,device,batch,ctx,label) not in done_inf:
                                w.writerow(dict(model=model,device=device,batch=batch,context_length=ctx,
                                    baseline=label,tp="",footprint_gb=f"{foot:.1f}",sparsity_fraction=sp,
                                    tpot_p50_ms="",tpot_p99_ms="",status="INFEASIBLE",wall_s=""))
                            continue
                        for tp in tps:
                            if (model,device,batch,ctx,label,tp) in done_ok: continue
                            runnable.append(dict(model=model,device=device,batch=batch,context_length=ctx,
                                baseline=label,tp=tp,footprint_gb=f"{foot:.1f}",sparsity_fraction=sp))
    if args.limit: runnable=runnable[:args.limit]
    fh.flush()
    print(f"NEW runnable TP points: {len(runnable)} (seeded {len(done_ok)} existing)",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(run_point,b["model"],b["device"],b["batch"],b["context_length"],
                        b["tp"],b["baseline"],b["sparsity_fraction"],args.timeout):b for b in runnable}
        n=0
        for fut in as_completed(futs):
            b=futs[fut]
            try: res=fut.result()
            except Exception as e: res=dict(tpot_p50_ms="",tpot_p99_ms="",status=f"ERR:{type(e).__name__}",wall_s="")
            w.writerow({**b,**res}); fh.flush(); n+=1
            if n%50==0: print(f"  {n}/{len(runnable)}",flush=True)
    fh.close(); print("TP SWEEP DONE ->",OUT,flush=True)

if __name__=="__main__": main()
