# -*- coding: utf-8 -*-
"""Recompute all seed-level statistics for the AdaQ-LoRA audit paper.

Reads per-seed evaluation values that were copied verbatim from:
  - aggregated_results.txt   (INT8 block lines 1-384, RTN block lines 385-768)
  - epsilon_and_tables.txt   (NF4 per-seed detail table, section 2)

Outputs a markdown table with paired t statistics, Cohen's d, 95% CI,
MDE at 80% power, BH-FDR adjusted q, and TOST sample-size requirements.

No scipy: the Student-t CDF is implemented from the regularized incomplete
beta function (Lentz continued fraction), and the inverse is obtained by
bisection.  Deterministic, no random numbers.
"""
import math
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- t distribution
def _betacf(a, b, x, maxit=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student-t with df degrees of freedom, t >= 0."""
    if t <= 0:
        return 0.5
    return 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))


def t_p_two_sided(t, df):
    return 2.0 * t_sf(abs(t), df)


def t_quantile(p, df):
    """Inverse CDF by bisection: returns t with P(T <= t) = p, p > 0.5."""
    lo, hi = 0.0, 1.0
    while 1.0 - t_sf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- data loading
def parse_aggregated():
    """Extract per-seed values for top-level metrics from aggregated_results.txt."""
    txt = (ROOT / "aggregated_results.txt").read_text(encoding="utf-8-sig", errors="replace")
    blocks = {}
    cur = None
    for line in txt.splitlines():
        m = re.match(r"^=== (.+?) AGGREGATE ===$", line.strip())
        if m:
            cur = m.group(1).strip()
            blocks[cur] = {}
            continue
        if cur is None:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        adapter, metric = parts[0], parts[1]
        if metric not in ("mmlu/acc,none",
                          "gsm8k/exact_match,strict-match",
                          "gsm8k/exact_match,flexible-extract"):
            continue
        seed_part = next((p for p in parts[2:] if p.startswith("seeds=")), "")
        val_part = next((p for p in parts[2:] if p.startswith("vals=")), "")
        seeds = re.findall(r"'([^']+)'", seed_part)
        vals = [float(v) for v in re.findall(r"[-+]?[0-9]*\.?[0-9]+", val_part)]
        if len(seeds) != len(vals):
            continue
        blocks[cur].setdefault(adapter, {})[metric] = dict(zip(seeds, vals))
    return blocks


def parse_nf4_per_seed():
    """Per-seed NF4 detail table from epsilon_and_tables.txt section 2."""
    txt = (ROOT / "epsilon_and_tables.txt").read_text(encoding="utf-8", errors="replace")
    rows = {}
    for line in txt.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        adapter, metric = cells[0], cells[1]
        if metric not in ("mmlu/acc", "gsm8k/exact_match,strict-match"):
            continue
        try:
            v42, v2025, v3407 = float(cells[2]), float(cells[3]), float(cells[4])
        except ValueError:
            continue
        key = ("mmlu/acc,none" if metric == "mmlu/acc"
               else "gsm8k/exact_match,strict-match")
        rows.setdefault(adapter, {})[key] = {"42": v42, "2025": v2025, "3407": v3407}
    return rows


# ---------------------------------------------------------------- statistics
def summarize(adapter_vals, base_vals, metric):
    """Paired statistics on seed intersection: delta = base - adapter."""
    common = [s for s in base_vals.get(metric, {}) if s in adapter_vals.get(metric, {})]
    if len(common) < 2:
        return None
    diffs = [base_vals[metric][s] - adapter_vals[metric][s] for s in common]
    n = len(diffs)
    mean = sum(diffs) / n
    ss = sum((d - mean) ** 2 for d in diffs)
    sd = math.sqrt(ss / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se if se > 0 else float("inf")
    p = t_p_two_sided(t, df)
    d = mean / sd if sd > 0 else float("inf")
    tcrit = t_quantile(0.975, df)
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    tpow = t_quantile(0.80, df)
    mde = (tcrit + tpow) * se
    return dict(seeds=sorted(common), n=n, delta_pt=mean * 100, sd_pt=sd * 100,
                ci_lo_pt=ci_lo * 100, ci_hi_pt=ci_hi * 100, d=d, t=t, df=df,
                p=p, mde_pt=mde * 100,
                q_pt_power_05=(0.5 / mde) ** 2 if mde else None)


def bh_fdr(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * m / rank
        prev = min(prev, val)
        q[i] = prev
    return q


def tost_n(sd, sesoi_pt, power=0.80, alpha=0.05):
    """Approximate TOST sample size: smallest n with (t_a + t_b) * sd/sqrt(n) <= sesoi."""
    delta = sesoi_pt / 100.0
    for n in range(2, 20001):
        df = n - 1
        need = (t_quantile(1 - alpha, df) + t_quantile(power, df)) * sd / math.sqrt(n)
        if need <= delta:
            return n
    return None


def summarize_map(d):
    """Paired stats from {seed: {adaq: x, qlora: y}}; delta = qlora - adaq."""
    seeds = sorted(s for s, v in d.items() if len(v) == 2)
    if len(seeds) < 2:
        return None
    diffs = [d[s]["qlora"] - d[s]["adaq"] for s in seeds]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se if se else float("inf")
    tcrit = t_quantile(0.975, df)
    return dict(seeds=seeds, n=n, delta_pt=mean * 100, sd_pt=sd * 100,
                ci_lo_pt=(mean - tcrit * se) * 100,
                ci_hi_pt=(mean + tcrit * se) * 100,
                d=mean / sd if sd else float("inf"), t=t, df=df,
                p=t_p_two_sided(t, df),
                mde_pt=(tcrit + t_quantile(0.80, df)) * se * 100, sd_diff=sd)


def load_b16_rtn_gsm8k():
    """Per-seed strict-match scores of the uniform-protocol INT4-RTN GSM8K runs.

    Source: rtn_b16_gsm8k.csv, produced on the training host by
    scripts/collect_b16.py from results_v2_rtn_b16/*/gsm8k/results_*.json.
    Every run uses 5-shot and batch 16; the legacy results had mixed 4/5-shot.
    Returns {seed: {method: acc}}.
    """
    path = ROOT / "rtn_b16_gsm8k.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                out.setdefault(int(row["seed"]), {})[row["method"]] = float(row["gsm8k_strict"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_b16_flex():
    """Same as load_b16_rtn_gsm8k but for the flexible-extract score."""
    path = ROOT / "rtn_b16_gsm8k.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                out.setdefault(int(row["seed"]), {})[row["method"]] = float(row["gsm8k_flex"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def fmt(x, nd=2, sign=False):
    if x is None:
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


def main():
    blocks = parse_aggregated()
    nf4 = parse_nf4_per_seed()

    sources = [
        ("NF4", nf4),
        ("INT4-RTN", blocks.get("RTN", {})),
        ("INT8", blocks.get("INT8", {})),
    ]
    cells = []
    for qname, data in sources:
        for metric, mlabel in (("mmlu/acc,none", "MMLU"),
                               ("gsm8k/exact_match,strict-match", "GSM8K-strict")):
            if "qlora" not in data or "adaq" not in data:
                continue
            r = summarize(data["adaq"], data["qlora"], metric)
            if r is None:
                continue
            r.update(quantizer=qname, metric=mlabel, metric_key=metric,
                     sd_diff=r["sd_pt"] / 100.0)
            cells.append(r)

    q_all = bh_fdr([c["p"] for c in cells])

    # replace the INT4-RTN GSM8K cell with the uniform-protocol (5-shot, batch 16) rerun
    b16 = load_b16_rtn_gsm8k()
    r16 = summarize_map(b16) if b16 else None
    if r16:
        for i, c in enumerate(cells):
            if c["quantizer"] == "INT4-RTN" and c["metric"] == "GSM8K-strict":
                r16.update(quantizer="INT4-RTN", metric="GSM8K-strict",
                           metric_key=c["metric_key"], protocol="5-shot, batch 16 (uniform)")
                cells[i] = r16
        q_all = bh_fdr([c["p"] for c in cells])

    for c, q in zip(cells, q_all):
        c["q_bh"] = q
        c["tost"] = {s: tost_n(c["sd_diff"], s) for s in (0.5, 1.0, 2.0, 5.0)}

    # flexible-extract robustness for the two quantizers where it exists
    flex = []
    r_rtn_flex = summarize_map(load_b16_flex())
    if r_rtn_flex:
        flex.append(("INT4-RTN", r_rtn_flex))
    r_int8_flex = summarize(blocks.get("INT8", {}).get("adaq", {}),
                            blocks.get("INT8", {}).get("qlora", {}),
                            "gsm8k/exact_match,flexible-extract")
    if r_int8_flex:
        flex.append(("INT8", r_int8_flex))
    qflex = bh_fdr([r["p"] for _, r in flex]) if flex else []
    for (qname, r), q in zip(flex, qflex):
        r["q_bh"] = q

    lines = []
    lines.append("## Table 2 (recomputed from per-seed data)\n")
    lines.append("| Quantizer | Benchmark | n | delta (pp) | SD_diff | 95% CI | d | t | p | q(BH) | MDE80 (pp) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        lines.append("| {q} | {m} | {n} | {dl} | {sd} | [{lo}, {hi}] | {d} | {t} | {p} | {qbh} | {mde} |".format(
            q=c["quantizer"], m=c["metric"], n=c["n"],
            dl=fmt(c["delta_pt"], 2, True), sd=fmt(c["sd_pt"], 2),
            lo=fmt(c["ci_lo_pt"], 2, True), hi=fmt(c["ci_hi_pt"], 2, True),
            d=fmt(c["d"], 3, True), t=fmt(c["t"], 2, True),
            p=fmt(c["p"], 3), qbh=fmt(c["q_bh"], 3), mde=fmt(c["mde_pt"], 2)))

    lines.append("\n## TOST sample size needed (alpha=.05, power=.80)\n")
    lines.append("| Quantizer | Benchmark | SESOI 0.5pp | 1pp | 2pp | 5pp |")
    lines.append("|---|---|---|---|---|---|")
    for c in cells:
        t = c["tost"]
        lines.append("| {q} | {m} | {a} | {b} | {c} | {d} |".format(
            q=c["quantizer"], m=c["metric"],
            a=t[0.5], b=t[1.0], c=t[2.0], d=t[5.0]))

    lines.append("\n## GSM8K flexible-extract robustness check\n")
    lines.append("| Quantizer | n | delta (pp) | 95% CI | d | p | q(BH) |")
    lines.append("|---|---|---|---|---|---|---|")
    for qname, r in flex:
        lines.append("| {q} | {n} | {dl} | [{lo}, {hi}] | {d} | {p} | {q2} |".format(
            q=qname, n=r["n"], dl=fmt(r["delta_pt"], 2, True),
            lo=fmt(r["ci_lo_pt"], 2, True), hi=fmt(r["ci_hi_pt"], 2, True),
            d=fmt(r["d"], 3, True), p=fmt(r["p"], 3), q2=fmt(r["q_bh"], 3)))

    # cross-check: reproduce the previously reported n=3 INT8 values if possible
    lines.append("\n## Sanity: old forest-plot claims vs recomputation\n")
    lines.append("- INT8 n used in recomputation: %d seeds" %
                 (cells[4]["n"] if len(cells) > 4 else -1))
    out = "\n".join(lines)
    print(out)
    (ROOT / "recomputed_stats.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
