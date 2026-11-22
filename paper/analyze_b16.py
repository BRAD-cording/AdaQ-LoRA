# -*- coding: utf-8 -*-
"""Paired analysis of the uniform-protocol (5-shot, batch 16) INT4-RTN GSM8K data,
plus a re-run of the full six-cell table with the new INT4-RTN row.

Reuses the Student-t CDF used in recompute_stats.py (no scipy).
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


from recompute_stats import (t_p_two_sided as t_p, t_quantile as t_q,
                             bh_fdr as bh, tost_n)

def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


def tost_n(sd_abs, sesoi_pt, power=0.80, alpha=0.05):
    delta = sesoi_pt / 100.0
    for n in range(2, 20001):
        df = n - 1
        if (t_q(1 - alpha, df) + t_q(power, df)) * sd_abs / math.sqrt(n) <= delta:
            return n
    return None


rows = list(csv.DictReader(open(ROOT / "rtn_b16_gsm8k.csv", encoding="utf-8-sig")))
by = {}
for r in rows:
    by.setdefault(int(r["seed"]), {})[r["method"]] = float(r["gsm8k_strict"])
seeds = sorted(s for s in by if len(by[s]) == 2)
diffs = [(by[s]["qlora"] - by[s]["adaq"]) for s in seeds]  # negative favours AdaQ
pp = [d * 100 for d in diffs]

n = len(pp)
mean = sum(pp) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in pp) / (n - 1))
se = sd / math.sqrt(n)
t = mean / se
p = t_p(t, n - 1)
d = mean / sd
tc = t_q(0.975, n - 1)
ci = (mean - tc * se, mean + tc * se)
mde = (tc + t_q(0.80, n - 1)) * se

print("=== INT4-RTN GSM8K, uniform protocol (5-shot, batch 16) ===")
print("seeds:", seeds)
print("n=%d  mean_delta=%+.3f pp  sd=%.3f pp  se=%.3f pp" % (n, mean, sd, se))
print("t(%d)=%+.3f  p=%.4f  d=%+.3f" % (n - 1, t, p, d))
print("95%% CI = [%+.3f, %+.3f] pp" % ci)
print("MDE80 = %.3f pp" % mde)
print("per-seed paired diffs (pp):", ["%+.2f" % x for x in pp])
for sesoi in (0.5, 1.0, 2.0, 3.0):
    inside = ci[0] > -sesoi and ci[1] < sesoi
    print("  SESOI %.1f pp -> equivalence %s ; power-based n=%s"
          % (sesoi, "YES" if inside else "no", tost_n(sd / 100, sesoi)))

# ---- rebuild the six-cell table with the new INT4-RTN GSM8K row
cells = [
    ("NF4", "MMLU", 3, -0.07, 0.03, -2.02),
    ("NF4", "GSM8K", 3, -0.55, 0.39, -1.41),
    ("INT4-RTN", "MMLU", 3, -0.12, 0.11, -1.10),
    ("INT4-RTN", "GSM8K", n, mean, sd, d),
    ("INT8", "MMLU", 5, +0.02, 0.29, +0.06),
    ("INT8", "GSM8K", 5, +1.46, 1.36, +1.07),
]
out = []
for q, b, nn, m, s, dd in cells:
    se_ = s / math.sqrt(nn)
    tt = m / se_ if se_ else float("inf")
    pp_ = t_p(tt, nn - 1)
    lo, hi = m - t_q(0.975, nn - 1) * se_, m + t_q(0.975, nn - 1) * se_
    md = (t_q(0.975, nn - 1) + t_q(0.80, nn - 1)) * se_
    out.append([q, b, nn, m, lo, hi, dd, pp_, md])

qs = bh([c[7] for c in out])
print("\n=== Table 1 (recomputed) ===")
print("| Quantizer | Benchmark | n | delta | 95% CI | d | q(BH) | MDE80 |")
for c, q in zip(out, qs):
    print("| %s | %s | %d | %+.2f | [%+.2f, %+.2f] | %+.2f | %.3f | %.2f |"
          % (c[0], c[1], c[2], c[3], c[4], c[5], c[6], q, c[8]))
print("\nmax |CI| per cell (largest SESOI still supported):")
for c in out:
    print("  %s %s -> %.2f pp" % (c[0], c[1], max(abs(c[4]), abs(c[5]))))
