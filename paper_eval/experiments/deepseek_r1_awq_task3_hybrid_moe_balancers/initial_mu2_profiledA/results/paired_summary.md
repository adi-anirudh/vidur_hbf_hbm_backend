## Paired comparison summary

These compare only cells where both systems exist. Positive reduction vs HBM-only means faster than capacity-aware HBM-only static. Negative reduction vs all-HBF means slower than all-HBF dynamic.

| comparison | cells | avg step delta | median step delta | win cells |
| --- | ---: | ---: | ---: | ---: |
| HBF=1 vs HBM-only | 74 | 5.29% | 6.01% | 50 |
| HBF=2 vs HBM-only | 74 | 7.28% | 9.04% | 52 |
| HBF=4 vs HBM-only | 74 | 8.69% | 10.45% | 52 |
| HBF=all vs HBM-only | 74 | 10.90% | 12.70% | 52 |
| HBF=1 vs all-HBF | 99 | -21.98% | -9.20% | 0 |
| HBF=2 vs all-HBF | 99 | -9.59% | -6.89% | 0 |
| HBF=4 vs all-HBF | 97 | -3.58% | -2.43% | 0 |

- B=8: HBM-only has no feasible paired cells under this capacity model.
- B=16: HBM-only feasible paired cells = 24.
- B=24: HBM-only feasible paired cells = 25.
- B=32: HBM-only feasible paired cells = 25.
