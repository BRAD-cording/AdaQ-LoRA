#!/usr/bin/env python3
"""P2-1 成本归一化聚合 + 预注册判定。
读 results_v2_p21/{arm}_s{seed}/ 的 gsm8k(flex/strict)+mmlu(acc)，按臂聚合 3 seed，
算配对差 dPM/dU/dR，机械套用预注册（4107552+修正案 448330a）R1/R1b/R2/R3。
噪声底 N 动态读 Phase3-D（results_v2_phase3_noise）实测，缺失则用预注册冻结值 4.09pt。
输出 results_v2_p21/summary.json + REPORT_p21.md。判定只填数不改规则（pre-unblind 冻结）。
"""
import os, sys, json, glob, math, statistics as st
from datetime import datetime, timezone

ROOT = "/home/ubuntu/AdaQ-LoRA"
RESDIR = os.path.join(ROOT, "results_v2_p21")
ARMS = ["cost_norm", "raw_score", "uniform", "uniform_pm"]
SEEDS = [42, 3407, 2025]
N_PREREG = 4.09  # 修正案 448330a 冻结的 Phase3-D 8B 噪声底（pt）
# 参数量（gen_p21_ranks.py 实算，确定性；运行时会尝试从 training_stats.json 覆写）
PARAMS = {"cost_norm": 3693568, "raw_score": 3782656, "uniform": 3407872, "uniform_pm": 3833856}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def find_results(run_dir):
    fs = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    return sorted(fs)[-1] if fs else None


def read_run(arm, seed):
    """返回 {gsm8k_flex,gsm8k_strict,mmlu} 百分比值，缺则 None。"""
    d = os.path.join(RESDIR, f"{arm}_s{seed}")
    f = find_results(d) if os.path.isdir(d) else None
    if not f:
        return None
    try:
        r = json.load(open(f))
    except Exception:
        return None
    res = r.get("results", {})
    g = res.get("gsm8k", {})
    m = res.get("mmlu", {})
    out = {
        "gsm8k_flex": g.get("exact_match,flexible-extract"),
        "gsm8k_strict": g.get("exact_match,strict-match"),
        "mmlu": m.get("acc,none"),
        "src": os.path.relpath(f, ROOT),
    }
    # 读 training_stats 的实测参数量（若有）
    ts = os.path.join(ROOT, "outputs", "p21", f"{arm}_s{seed}", "training_stats.json")
    if os.path.isfile(ts):
        try:
            tj = json.load(open(ts))
            if tj.get("params_actual"):
                out["params_actual"] = tj["params_actual"]
        except Exception:
            pass
    return out


def pct(x):
    return None if x is None else round(x * 100, 4)


def load_noise_floor():
    """动态读 Phase3-D 9run gsm8k-flex 极差作噪声底 N（pt）；缺失返回 N_PREREG。"""
    ddir = os.path.join(ROOT, "results_v2_phase3_noise")
    vals = []
    if os.path.isdir(ddir):
        for run in glob.glob(os.path.join(ddir, "product_s*_r*")):
            f = find_results(run)
            if not f:
                continue
            try:
                r = json.load(open(f))
                v = r.get("results", {}).get("gsm8k", {}).get("exact_match,flexible-extract")
                if v is not None:
                    vals.append(v * 100)
            except Exception:
                pass
    if len(vals) >= 2:
        return round(max(vals) - min(vals), 4), len(vals), "Phase3-D 实测"
    return N_PREREG, 0, "预注册冻结值(Phase3-D 数据缺失)"


def agg_arm(arm):
    per = {}
    for s in SEEDS:
        per[s] = read_run(arm, s)
    flex = [pct(per[s]["gsm8k_flex"]) for s in SEEDS if per[s] and per[s]["gsm8k_flex"] is not None]
    strict = [pct(per[s]["gsm8k_strict"]) for s in SEEDS if per[s] and per[s]["gsm8k_strict"] is not None]
    mmlu = [pct(per[s]["mmlu"]) for s in SEEDS if per[s] and per[s]["mmlu"] is not None]
    params = [per[s].get("params_actual") for s in SEEDS if per[s] and per[s].get("params_actual")]
    p = params[0] if params else PARAMS.get(arm)

    def stat(xs):
        if not xs:
            return None
        return {
            "n": len(xs), "mean": round(sum(xs) / len(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs),
            "range": round(max(xs) - min(xs), 4),
            "values": xs,
        }
    return {"per_seed": per, "gsm8k_flex": stat(flex), "gsm8k_strict": stat(strict),
            "mmlu": stat(mmlu), "params": p}


def paired(armA, armB, metric, data):
    """逐 seed 配对差 armA−armB（pt），返回 (diffs, n_pos, mean)。"""
    diffs = []
    for s in SEEDS:
        a = data[armA]["per_seed"].get(s)
        b = data[armB]["per_seed"].get(s)
        if a and b and a.get(metric) is not None and b.get(metric) is not None:
            diffs.append(round(pct(a[metric]) - pct(b[metric]), 4))
    npos = sum(1 for d in diffs if d > 0)
    mean = round(sum(diffs) / len(diffs), 4) if diffs else None
    return diffs, npos, mean


def main():
    os.makedirs(RESDIR, exist_ok=True)
    N, n_noise, n_src = load_noise_floor()
    data = {arm: agg_arm(arm) for arm in ARMS}

    # 配对差（主判据 gsm8k_flex）
    dPM, dPM_pos, dPM_mean = paired("cost_norm", "uniform_pm", "gsm8k_flex", data)   # R1 主
    dU, dU_pos, dU_mean = paired("cost_norm", "uniform", "gsm8k_flex", data)         # R1b 等秩和
    dR, dR_pos, dR_mean = paired("cost_norm", "raw_score", "gsm8k_flex", data)       # R3 成本归一化 vs 原始分

    # 参数对照（cost_norm 是否更省）
    p_cn = data["cost_norm"]["params"]
    p_pm = data["uniform_pm"]["params"]
    cn_cheaper = (p_cn is not None and p_pm is not None and p_cn < p_pm)

    # R1 判定：3/3 种子 dPM>0 且 mean(dPM)>N 且 cost_norm 更省参数
    have3 = len(dPM) == 3
    r1_sign = have3 and dPM_pos == 3
    r1_mag = dPM_mean is not None and dPM_mean > N
    R1 = bool(r1_sign and r1_mag and cn_cheaper)
    # R1b（等秩和次对照，须与 R1 一致）
    r1b_sign = len(dU) == 3 and dU_pos == 3
    r1b_mag = dU_mean is not None and dU_mean > N
    R1b = bool(r1b_sign and r1b_mag)

    if R1 and R1b:
        verdict = "R1 命中（正向）→ 触发 R1-confirm 扩 6 seed 确证；venue 上探 KBS(Q1)"
    elif R1 and not R1b:
        verdict = "R1 命中但 R1b（等秩和）不一致 → 信号可能来自参数/秩和差异而非成本归一化本身，降级审慎，需 R1-confirm 厘清"
    else:
        verdict = "R2（阴性）：cost_norm 未稳定超过 uniform_pm（参数匹配）→ 维持「机制+阴性+功效边界」四件套，退 TMLR/Neural Networks/Neurocomputing"

    summary = {
        "_meta": {"generated": now(), "prereg": "docs/preregistration-p21-costnorm.md (4107552+448330a)",
                  "noise_floor_N_pt": N, "noise_floor_n_runs": n_noise, "noise_floor_src": n_src,
                  "primary_metric": "gsm8k_flex", "n_seeds": len(SEEDS), "seeds": SEEDS},
        "arms": {arm: {k: v for k, v in data[arm].items() if k != "per_seed"} for arm in ARMS},
        "paired_gsm8k_flex": {
            "dPM_cost_norm_minus_uniform_pm": {"diffs": dPM, "n_pos": dPM_pos, "mean": dPM_mean},
            "dU_cost_norm_minus_uniform": {"diffs": dU, "n_pos": dU_pos, "mean": dU_mean},
            "dR_cost_norm_minus_raw_score": {"diffs": dR, "n_pos": dR_pos, "mean": dR_mean},
        },
        "params": {"cost_norm": p_cn, "uniform_pm": p_pm, "cost_norm_cheaper": cn_cheaper,
                   "ratio_cn_over_pm": round(p_cn / p_pm, 4) if (p_cn and p_pm) else None},
        "decision": {"R1": R1, "R1_sign_3of3": bool(r1_sign), "R1_mean_gt_N": bool(r1_mag),
                     "R1b": R1b, "verdict": verdict},
    }
    with open(os.path.join(RESDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # REPORT_p21.md
    L = []
    L.append(f"# P2-1 成本归一化秩分配 — 预注册判定报告\n\n生成：{now()}  \n预注册：docs/preregistration-p21-costnorm.md（4107552 + 修正案 448330a）\n")
    L.append(f"## 噪声底\n\nN = **{N} pt**（来源：{n_src}；Phase3-D 9run gsm8k-flex 极差，n={n_noise}）\n")
    L.append("## 各臂 GSM8K-flex（%，3 seed）\n\n| 臂 | 参数 | mean | std | min | max | 各种子值 |\n|---|---|---|---|---|---|---|")
    for arm in ARMS:
        a = data[arm]["gsm8k_flex"]
        if a:
            L.append(f"| {arm} | {data[arm]['params']:,} | {a['mean']} | {a['std']} | {a['min']} | {a['max']} | {a['values']} |")
        else:
            L.append(f"| {arm} | {data[arm]['params']} | — | — | — | — | 数据缺失 |")
    L.append("\n## 配对差（gsm8k-flex，pt，逐 seed）\n")
    L.append(f"- **dPM = cost_norm − uniform_pm（R1 主判据）**：{dPM}  同号 {dPM_pos}/3  均值 {dPM_mean}")
    L.append(f"- dU = cost_norm − uniform（R1b 等秩和）：{dU}  同号 {dU_pos}/3  均值 {dU_mean}")
    L.append(f"- dR = cost_norm − raw_score（R3 成本归一化 vs 原始分）：{dR}  同号 {dR_pos}/3  均值 {dR_mean}")
    L.append(f"\n参数对照：cost_norm {p_cn:,} vs uniform_pm {p_pm:,} → cost_norm 更省参数 = **{cn_cheaper}**\n")
    L.append("## 判定（机械套用预注册，只填数不改规则）\n")
    L.append(f"- R1（3/3 dPM>0 且 mean>{N} 且 cost_norm 更省参数）= **{R1}**（sign {r1_sign} / mag {r1_mag} / cheaper {cn_cheaper}）")
    L.append(f"- R1b（等秩和 3/3 dU>0 且 mean>{N}）= **{R1b}**")
    L.append(f"\n### 结论\n\n**{verdict}**\n")
    L.append("\n## 强制披露（预注册 R5 + 评审）\n")
    L.append("- n=3 种子欠功效；strict 永不作判据（R4，Qwen/Llama 格式）；MMLU 仅报告不判定。")
    L.append("- 主判据为 BF16 基座 gsm8k-flex；与 NF4 表不可横比。")
    L.append("- cost_norm 与 uniform_pm 参数差 3.7%（cn 更省）→ 对照排除参数量混杂；与 uniform 等秩和（512）对照隔离成本归一化本身。")
    L.append("- 若 R1 命中：先扩 6 seed（R1-confirm）再写正向，防 14B n=3 单种子扛均值覆辙。")
    with open(os.path.join(RESDIR, "REPORT_p21.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"[p21-agg] N={N}pt({n_src}) R1={R1} R1b={R1b} dPM_mean={dPM_mean} → {verdict}")
    print(f"[p21-agg] 写 {RESDIR}/summary.json + REPORT_p21.md")


if __name__ == "__main__":
    main()
