# Section 6 — Power Analysis and Limitations of Null Results
# Generated: 2026-11-19

## 6.1 Minimum Detectable Effect

With n=3 seeds per condition, the minimum detectable effect (paired t-test,
alpha=0.05 two-sided, 1-beta=0.80) is:

| Scale | Noise sigma | MDE |
|---|---|---|
| 8B | 4.09 pt (seed+retrain noise, Phase3-D) | 12.7 pt |
| 14B | 4.57 pt (paired adaq-qlora std, 3 seeds) | 14.2 pt |

All observed differences between adaq and qlora fall below 1.6pt (8B: 0.66pt;
14B post-chat-template: 0.73pt), far below the MDE. The experiment can confidently
exclude practically large effects (>=12.7pt) but is underpowered for small effects
in the 1-3pt range.

## 6.2 TOST Equivalence

Two one-sided tests (TOST) for paired differences at equivalence boundary delta:

| delta | 8B required n | 14B required n |
|---|---|---|
| 1 pt | 209 | 260 |
| 3 pt | 25 | 31 |
| 5 pt | 10 | 12 |

At the current n=3, we can only establish equivalence at delta >= 12.7pt (which
coincides with the MDE). Formal equivalence at meaningful effect sizes (1-3pt)
requires substantially more seeds. We recommend n >= 6 for future work to reduce
the MDE below 8pt, and n >= 10 for TOST at delta = 5pt.

## 6.3 95% Confidence Intervals

14B paired difference (adaq - qlora) post chat-template correction:
- Mean: +0.73pt (was +8.87pt before correction)
- 95% CI: [-10.6, +12.1] pt
- The CI spans zero with width 22.7pt, consistent with both a small advantage
  and a small disadvantage.
- The pre-correction +8.87pt signal was a format artifact (sign flips across
  seeds: +1.74, -0.23, +0.68).

## 6.4 Effect Sizes (Cohen's d)

| Comparison | d | Interpretation |
|---|---|---|
| 8B phase3-A (adaq vs qlora) | 0.016 | Trivial |
| 14B post-chat-template | 0.16 | Trivial |
| 14B pre-correction | 1.94 | Large (but artifact) |

## 6.5 Recommended Reporting

We report that no practically meaningful advantage of error-aware over uniform
rank allocation was detected. The experiment can reliably exclude effects >=12.7pt
at 80% power. The observed differences (<=1.6pt) are consistent with both true
null and a small effect below the detection threshold. We do NOT claim strict
equivalence, as n=3 is insufficient for TOST at meaningful boundaries.