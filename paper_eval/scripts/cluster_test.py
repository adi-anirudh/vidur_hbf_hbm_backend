#!/usr/bin/env python3
"""
Expert-union reduction from greedy expert-aware clustering.

Take N tokens, each with its top-k expert activations. Partition them into
clusters of a fixed size and measure the mean per-cluster expert union under:
  (1) RANDOM partition   (what B4 balanced routing does today)
  (2) GREEDY min-union   (Ours: build each cluster by adding the token that
                          contributes the fewest NEW experts)

The reduction depends entirely on how STRUCTURED the routing is:
  structure = 0.0  -> uniform random top-k (no exploitable structure)
  structure -> 1.0 -> tokens drawn from latent groups that share hot experts

To get the REAL number for DeepSeek-V3, replace gen_routing() with actual
gate top-k traces (per layer, per step) dumped from a model run.
"""
import numpy as np

N_TOKENS   = 512
N_EXPERTS  = 256
TOP_K      = 8
CLUSTER    = 16                # microbatch size
N_GROUPS   = 16                # latent topic groups (for structured routing)
HOT_POOL   = 32                # experts each group prefers

def gen_routing(structure, rng):
    """Return list of frozensets: each token's top-k experts."""
    toks = []
    # each group's preferred (hot) expert pool
    hot = [rng.choice(N_EXPERTS, HOT_POOL, replace=False) for _ in range(N_GROUPS)]
    for _ in range(N_TOKENS):
        g = rng.integers(N_GROUPS)
        chosen = set()
        while len(chosen) < TOP_K:
            if rng.random() < structure:
                chosen.add(int(rng.choice(hot[g])))     # from group's hot pool
            else:
                chosen.add(int(rng.integers(N_EXPERTS))) # uniform
        toks.append(frozenset(chosen))
    return toks

def random_partition(toks, rng):
    idx = rng.permutation(len(toks))
    unions = []
    for i in range(0, len(toks), CLUSTER):
        u = set().union(*[toks[j] for j in idx[i:i+CLUSTER]])
        unions.append(len(u))
    return np.mean(unions)

def greedy_partition(toks):
    remaining = set(range(len(toks)))
    unions = []
    while remaining:
        seed = next(iter(remaining)); remaining.discard(seed)
        cluster_union = set(toks[seed]); n = 1
        while n < CLUSTER and remaining:
            best = min(remaining, key=lambda t: len(toks[t] - cluster_union))
            cluster_union |= toks[best]; remaining.discard(best); n += 1
        unions.append(len(cluster_union))
    return np.mean(unions)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{N_TOKENS} tokens, top-{TOP_K} of {N_EXPERTS}, clusters of {CLUSTER}")
    print(f"(full-batch union of {N_TOKENS} tokens is ~256 either way; "
          f"this measures the PER-CLUSTER union)\n")
    print(f"{'structure':>10}{'random union':>14}{'greedy union':>14}{'reduction':>11}")
    for s in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        # average over a few random routings for stability
        rr, gg = [], []
        for seed in range(5):
            toks = gen_routing(s, np.random.default_rng(seed))
            rr.append(random_partition(toks, np.random.default_rng(seed+100)))
            gg.append(greedy_partition(toks))
        r, g = np.mean(rr), np.mean(gg)
        print(f"{s:>10.1f}{r:>14.1f}{g:>14.1f}{(1-g/r)*100:>10.0f}%")
    print("\nstructure=0.0 : uniform routing (worst case for clustering)")
    print("structure=1.0 : every token's experts come from its group's 32-expert pool")
    print("Swap gen_routing() for real DeepSeek-V3 gate traces to get the true number.")
