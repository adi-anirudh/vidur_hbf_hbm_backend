# FlexEP Paper Roadmap

## Core paper direction

FlexEP upgrades existing HBM clusters with add-on HBF MoE accelerators. HBM retains write-intensive attention and KV state, while HBF capacity fully replicates read-mostly experts. Replication turns expert placement from a fixed capacity constraint into a per-microbatch scheduling decision.

The paper should not claim that FlexEP universally beats HBF-colocated execution. It should establish the operating regimes where FlexEP provides:

- Better incremental deployment cost for existing HBM clusters.
- Lower MoE straggler time than capacity-constrained static or partial-replica EP.
- Lower HBF write traffic than an all-HBF colocated design.
- A measured crossover where colocated execution wins at sufficiently high load.

## Systems to compare

| System | Attention/KV | Experts | Placement and scheduling |
|---|---|---|---|
| HBM-Coloc | HBM | HBM, sharded | Optimized conventional EP |
| HBM-Disagg-Static | HBM | HBM, sharded | Profile-optimized fixed ownership |
| HBM-Disagg-PartialDyn | HBM | HBM, capacity-limited replicas | Dynamic among eligible replicas |
| FlexEP-Sharded | HBM | HBF, sharded | Fixed ownership |
| FlexEP-Replica-Fixed | HBM | HBF, fully replicated | Fixed assignment |
| FlexEP | HBM | HBF, fully replicated | Per-microbatch dynamic assignment |
| HBF-Coloc-Oracle | HBF | HBF, sharded | Best colocated configuration |

Report existing HBM GPUs and added HBF MoE nodes separately instead of treating every device as a fungible GPU. Keep the attention pool identical across disaggregated systems when making direct comparisons.

Run important comparisons both without pruning and with matched pruning. The no-pruning system is the primary result; pruning is an optional optimization whose model-quality impact must be measured.

## Immediate next steps

### 1. Freeze the claim and evaluation rules

- Write the final one-sentence claim and use it consistently throughout the paper.
- Define the add-on HBF MoE-node organization, staging memory, network, and resource accounting.
- Replace random static ownership with a profile-optimized static baseline.
- Match pruning, device budgets, SLOs, bandwidth, compute, and capacity assumptions across comparisons.
- Treat HBF capacity, rather than higher bandwidth, as the source of expert-replication freedom.

### 2. Collect real routing traces

- Begin with DeepSeek-V2-Lite on the available single GPU.
- Run faithful full-model inference and hook router outputs. Attention-only execution is not sufficient because previous MoE outputs affect later hidden states and routing decisions.
- Record selected expert IDs, router weights, tokens per expert, active experts, maximum and mean load, and cumulative router mass for every layer and decode microbatch.
- Sweep supported context lengths, session counts, prompt domains, and microbatch counts `1, 2, 4, 8`.
- Keep `microbatch_count = 2` as the default only if real traces confirm that it is near the latency optimum.
- Seek a limited DeepSeek-V3 trace source or collaborator. V2-Lite can validate the mechanism, but it cannot alone validate a headline claim about 256-expert DeepSeek-V3.

### 3. Validate dynamic load balancing without pruning

- Compare fixed scheduling with count-aware, routed-token-aware, and predicted-cost-aware scheduling.
- Balance total predicted work, not only the number of active experts.
- Use whole-expert jobs by default to retain efficient batched execution.
- Evaluate splitting a hot expert across replica nodes only when the predicted saved execution time exceeds the added communication and kernel overhead.
- Show whether gains come from active-expert-count balancing, hot-expert compute balancing, or both.

### 4. Validate pruning separately

- Define pruning from the expert set before pruning versus the retained set after pruning.
- Apply pruning per microbatch using aggregated router mass.
- Plot the unpruned and pruned FlexEP results side by side.
- Measure perplexity and downstream task accuracy instead of presenting latency-only pruning results.
- Remove pruning from the core contribution if its performance gain requires unacceptable quality loss.

### 5. Correct and calibrate the performance model

Include:

- Fixed installed HBM attention capacity and incremental HBF MoE nodes.
- HBM and HBF memory capacity constraints.
- Activation traffic between the attention and MoE pools.
- Expert reads at equal HBM and HBF bandwidth.
- HBF program latency and write traffic.
- Router and scheduling overhead.
- Collective communication and synchronization.
- Kernel-efficiency loss at small per-expert token counts.

Validate representative analytical predictions against the detailed HBF backend. Aim for less than 10% TBT prediction error before relying on large analytical sweeps.

## Experimental sweeps

Sweep:

- Context length from short context through the maximum supported context.
- Sessions: `32, 64, 128, 256`, extending where useful.
- Microbatch count: `1, 2, 4, 8`.
- No pruning and quality-safe pruning thresholds.
- HBF MoE-node count.
- Network bandwidth and latency.
- Router skew and workload/domain shift.
- Static, partial-replica, and fully replicated expert placement.
- Count-aware, token-aware, and cost-aware dynamic scheduling.

For each configuration, report TBT, throughput, tail latency, active experts per microbatch, maximum/mean node work, HBM GPU count, incremental HBF-node count, network traffic, and HBF write traffic.

## Intended figure story

1. Real router-weight concentration and per-microbatch imbalance.
2. Expert population before and after quality-safe pruning.
3. Microbatch sensitivity and the selected default.
4. Full replication versus sharding with scheduling held fixed.
5. Dynamic versus fixed scheduling with replication held fixed.
6. Count balancing versus hot-expert cost balancing.
7. TBT-resource Pareto with HBM GPUs and HBF add-ons reported separately.
8. FlexEP versus the HBF-colocated performance oracle, including the crossover.
9. HBF write traffic and endurance exposure.
10. Pruning benefit versus model-quality loss.

## Work order

1. Freeze the claim, topology, baseline matrix, and fair-comparison rules.
2. Collect real V2-Lite routing traces.
3. Replay no-pruning static and dynamic scheduling on the traces.
4. Decide whether the real imbalance is strong enough to support the paper.
5. Implement optimized static and partial-replica baselines.
6. Add cost-aware scheduling, optional hot-expert splitting, and calibrated overheads.
7. Evaluate pruning with model-quality measurements.
8. Run the complete context, workload, resource, and network sweeps.
9. Produce matched Pareto, crossover, and endurance results.
10. Rewrite the paper around measured results, including negative regimes and limitations.

## Go/no-go criteria

Continue with dynamic scheduling as a main contribution if real traces show at least one of:

- At least 10% TBT improvement over the strongest matched static or partial-replica baseline in multiple realistic regimes.
- At least 15% fewer incremental MoE accelerators at the same TBT SLO.
- A material tail-latency reduction under router or workload-distribution shift.

If the evidence differs from the current hypothesis:

- If replication helps but dynamic scheduling does not, focus the paper on capacity-enabled deployment and remove the scheduling headline.
- If gains appear only with unsafe pruning, remove pruning from the core contribution.
- If HBF-colocated wins on performance everywhere, focus FlexEP on installed-cluster reuse and reduced HBF KV-write exposure, provided those advantages are quantitatively strong.
- If no credible V3-scale trace is obtained, label 256-expert results as projected sensitivity rather than measured evidence.

## Current defaults

- Target: ASPLOS 2027 September submission cycle.
- Organization: add-on HBF MoE nodes connected to existing HBM attention GPUs.
- Primary result: real-trace, no-pruning, per-microbatch dynamic scheduling.
- Pruning: secondary and quality constrained.
- HBM and HBF read bandwidth: equal at 1 TB/s unless explicitly varied.
- HBF advantage: capacity and read-mostly expert storage, not faster bandwidth.
