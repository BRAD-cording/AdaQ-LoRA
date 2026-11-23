"""final_table.py -- single source of truth for Table 1.

Every cell is computed here from per-seed scores, with no hand-entered means,
SDs or confidence intervals. Earlier versions of the manuscript mixed
hand-rounded values with recomputed ones, which produced a row whose d and CI
could not both be true; this script exists so that cannot happen again.

Data provenance per cell:
  NF4 MMLU      6 values transcribed from epsilon_and_tables.txt (5-shot, 14042)
  NF4 GSM8K     10 seeds, nf4_gsm8k_n10.csv (5-shot, batch 16)
  INT4 MMLU     3 seeds, rtn_b16_mmlu.csv (5-shot, batch 4)
  INT4 GSM8K    12 seeds, rtn_b16_gsm8k.csv (5-shot, batch 16)
  INT8 MMLU     5 seeds (3 original + 2 retrained at grad-accum 16)
  INT8 GSM8K    5 seeds (same)
"""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from recompute_stats import t_p_two_sided as tp, t_quantile as tq, bh_fdr  # noqa: E402

# cell -> {seed: (adaq, qlora)} in raw accuracy
CELLS = {
    ("NF4", "MMLU"): {42: (0.6543, 0.6532), 3407: (0.6533, 0.6528), 2025: (0.6545, 0.6540)},
    ("INT4-RTN", "MMLU"): {},   # filled from rtn_b16_mmlu.csv
    ("INT8", "MMLU"): {7: (0.628757, 0.628757), 42: (0.629754, 0.629041),
                       100: (0.628116, 0.626905), 2025: (0.624769, 0.628543),
                       3407: (0.626549, 0.628970)},
    ("INT8", "GSM8K"): {7: (0.423806, 0.430629), 42: (0.496588, 0.489007),
                        100: (0.437453, 0.420015), 2025: (0.492798, 0.488249),
                        3407: (0.477635, 0.473844)},
}


def load_csv(path, method_col, value_col, seed_col="seed"):
    out = {}
    with open(ROOT / path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                out.setdefault(int(row[seed_col]), {})[row[method_col]] = float(row[value_col])
            except (KeyError, TypeError, ValueError):
                continue
    return out


nf4 = load_csv("nf4_gsm8k_n10.csv", "method", "gsm8k_strict")
CELLS[("NF4", "GSM8K")] = {s: (v["adaq"], v["qlora"]) for s, v in nf4.items() if len(v) == 2}

rtn_m = load_csv("rtn_b16_mmlu.csv", "method", "mmlu")
CELLS[("INT4-RTN", "MMLU")] = {s: (v["adaq"], v["qlora"]) for s, v in rtn_m.items() if len(v) == 2}

rtn_g = load_csv("rtn_b16_gsm8k.csv", "method", "gsm8k_strict")
CELLS[("INT4-RTN", "GSM8K")] = {s: (v["adaq"], v["qlora"]) for s, v in rtn_g.items() if len(v) == 2}


def analyse(d):
    diffs = [(d[s][1] - d[s][0]) * 100 for s in sorted(d)]   # QLoRA - AdaQ
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se
    tc = tq(0.975, df)
    return dict(n=n, mean=mean, sd=sd, lo=mean - tc * se, hi=mean + tc * se,
                d=mean / sd, t=t, df=df, p=tp(t, df),
                mde=(tc + tq(0.80, df)) * se)


ORDER = [("NF4", "MMLU"), ("NF4", "GSM8K"), ("INT4-RTN", "MMLU"),
         ("INT4-RTN", "GSM8K"), ("INT8", "MMLU"), ("INT8", "GSM8K")]
res = {k: analyse(CELLS[k]) for k in ORDER}
qs = bh_fdr([res[k]["p"] for k in ORDER])

print("| Quantizer | Benchmark | n | Δ (pp) | 95% CI | d | t | p | q(BH) | MDE |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, q in zip(ORDER, qs):
    r = res[k]
    print("| %s | %s | %d | %+.2f | [%+.2f, %+.2f] | %+.2f | %+.2f | %.3f | %.3f | %.2f |" % (
        k[0], k[1], r["n"], r["mean"], r["lo"], r["hi"], r["d"], r["t"], r["p"], q, r["mde"]))

print()
print("min q = %.3f" % min(qs))
print()
print("LaTeX rows:")
for k, q in zip(ORDER, qs):
    r = res[k]
    print("%-9s & %-6s & %-2d & $%+.2f$ & [$%+.2f$, $%+.2f$] & $%+.2f$ & %.3f & %.2f \\\\" % (
        k[0], k[1], r["n"], r["mean"], r["lo"], r["hi"], r["d"], q, r["mde"]))
