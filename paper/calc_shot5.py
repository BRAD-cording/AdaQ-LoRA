"""Recompute all six cells with the 5-shot GSM8K re-scores.

NF4 and INT8 GSM8K come from shot5_gsm8k.csv (measured on the same adapters at
5-shot / batch 16). The INT4-RTN GSM8K cell keeps its 12-seed values and the
MMLU cells are unchanged, since MMLU was 5-shot everywhere already.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from recompute_stats import t_p_two_sided as tp, t_quantile as tq, bh_fdr  # noqa: E402


def load():
    groups = {}
    with open(ROOT / "shot5_gsm8k.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if not row.get("quantizer") or row["quantizer"].startswith("#"):
                continue
            key = (row["quantizer"], int(row["seed"]))
            groups.setdefault(key, {})[row["method"]] = float(row["gsm8k_strict"])
    return groups


def paired(d):
    seeds = sorted(s for (q, s), v in d.items() if len(v) == 2)
    diffs = [d[(q, s)]["qlora"] - d[(q, s)]["adaq"] for (q, s) in d if len(d[(q, s)]) == 2]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se
    tc = tq(0.975, df)
    return dict(n=n, mean=mean * 100, sd=sd * 100, lo=(mean - tc * se) * 100,
                hi=(mean + tc * se) * 100, d=mean / sd, p=tp(t, df),
                mde=(tc + tq(0.80, df)) * se * 100)


g = load()
nf4 = paired({k: v for k, v in g.items() if k[0] == "phase3_main"})
int8 = paired({k: v for k, v in g.items() if k[0] == "phase3_int8"})

print("NF4 GSM8K  (5-shot, n=%d): delta=%+.2f pp  sd=%.2f  CI=[%+.2f,%+.2f]  d=%+.2f  p=%.3f  MDE=%.2f"
      % (nf4["n"], nf4["mean"], nf4["sd"], nf4["lo"], nf4["hi"], nf4["d"], nf4["p"], nf4["mde"]))
print("INT8 GSM8K (5-shot, n=%d): delta=%+.2f pp  sd=%.2f  CI=[%+.2f,%+.2f]  d=%+.2f  p=%.3f  MDE=%.2f"
      % (int8["n"], int8["mean"], int8["sd"], int8["lo"], int8["hi"], int8["d"], int8["p"], int8["mde"]))

cells = [
    ("NF4", "MMLU", 3, -0.07, 0.03, -2.02),
    ("NF4", "GSM8K", nf4["n"], nf4["mean"], nf4["sd"], nf4["d"]),
    ("INT4-RTN", "MMLU", 3, -0.0451, 0.0748, -0.603),
    ("INT4-RTN", "GSM8K", 12, -0.64, 1.38, -0.467),
    ("INT8", "MMLU", 5, 0.02, 0.29, 0.056),
    ("INT8", "GSM8K", int8["n"], int8["mean"], int8["sd"], int8["d"]),
]
out = []
for q, b, n, m, s, d in cells:
    se = s / math.sqrt(n)
    t = m / se
    tc = tq(0.975, n - 1)
    out.append([q, b, n, m, m - tc * se, m + tc * se, d, tp(t, n - 1),
                (tc + tq(0.80, n - 1)) * se])
qs = bh_fdr([c[7] for c in out])
print("\n| Quantizer | Benchmark | n | delta | 95% CI | d | p | q(BH) | MDE | max|CI| |")
for c, q in zip(out, qs):
    print("| %s | %s | %d | %+.2f | [%+.2f, %+.2f] | %+.2f | %.3f | %.3f | %.2f | %.2f |"
          % (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], q, c[8], max(abs(c[4]), abs(c[5]))))
