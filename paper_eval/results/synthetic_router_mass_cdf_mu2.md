# Synthetic router cumulative-mass summary at μ=2

> This characterizes the router generator used by the analytical sweep. It is not a DeepSeek-V3 measurement.
> Counts use router mass before adding top-1-protected experts.

Each distribution contains `4 trials × 58 MoE layers × 2 microbatches = 464` samples.

| Sessions | Tokens/MB | Target | Experts on mean curve | Median/sample | 10th–90th percentile |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 90% | 48 | 48 | 44–53 |
| 32 | 16 | 95% | 60 | 60 | 55–65 |
| 32 | 16 | 98% | 72 | 72 | 66–77 |
| 32 | 16 | 99% | 79 | 79 | 73–84 |
| 128 | 64 | 90% | 109 | 109 | 103–115 |
| 128 | 64 | 95% | 131 | 131 | 124–137 |
| 128 | 64 | 98% | 153 | 153 | 145–159 |
| 128 | 64 | 99% | 165 | 165 | 157–171 |
| 512 | 256 | 90% | 158 | 158 | 152–162 |
| 512 | 256 | 95% | 182 | 182 | 177–187 |
| 512 | 256 | 98% | 203 | 203 | 198–208 |
| 512 | 256 | 99% | 215 | 214 | 209–219 |
| 1024 | 512 | 90% | 169 | 169 | 165–173 |
| 1024 | 512 | 95% | 193 | 193 | 189–197 |
| 1024 | 512 | 98% | 214 | 213 | 210–218 |
| 1024 | 512 | 99% | 224 | 224 | 220–227 |

Gate logits are normalized over each token's selected top-8 experts; mass outside that selected set is unavailable in the current synthetic trace.
