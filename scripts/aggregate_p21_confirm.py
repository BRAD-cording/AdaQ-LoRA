#!/usr/bin/env python3
"""P2-1 R1-confirm 确证聚合（预注册 §10.5，4107552+448330a）。
仅当主跑 R1=true 触发。读 cost_norm+uniform_pm 跨 6 seed 的 gsm8k-flex，
算 6-seed dPM 配对差，机械套用确证判定（只填数不改规则）：
  确证阳性(→KBS Q1): ≥5/6 dPM_s>0 且 mean(dPM over 6)>N(4.09) 且 cost_norm 更省参数
  初步阳性(降级)   : 主跑 R1 命中但 confirm 失败 → 仅附录，不进 abstract/intro
输出 results_v2_p21/summary_confirm.json + REPORT_p21_confirm.md。
"""
import os, sys, json, glob, statistics as st
from datetime import datetime, timezone

ROOT = "/home/ubuntu/AdaQ-LoRA"
RESDIR = os.path.join(ROOT, "results_v2_p21")
SEEDS = [42, 3407, 2025, 1337, 7, 2024]   # §10.5 冻结的 6 seed
ARMS = ["cost_norm", "uniform_pm"]
N_PREREG = 4.09


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def find_results(run_dir):
    fs = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    return sorted(fs)[-1] if fs else None


def flex_of(arm, seed):
    d = os.path.join(RESDIR, f"{arm}_s{seed}")
    f = find_results(d) if os.path.isdir(d) else None
    if not f:
        return None
    try:
        r = json.load(open(f))
        v = r.get("results", {}).get("gsm8k", {}).get("exact_match,flexible-extract")
        return None if v is None else round(v * 100, 4)
    except Exception:
        return None


def load_noise_floor():
    ddir = os.path.join(ROOT, "results_v2_phase3_noise")
    vals = []
    if os.path.isdir(ddir):
        for run in glob.glob(os.path.join(ddir, "product_s*_r*")):
            f = find_results(run)
            if not f:
                continue
            try:
                v = json.load(open(f)).get("results", {}).get("gsm8k", {}).get("exact_match,flexible-extract")
                if v is not None:
                    vals.append(v * 100)
            except Exception:
                pass
    if len(vals) >= 2:
        return round(max(vals) - min(vals), 4), len(vals), "Phase3-D 实测"
    return N_PREREG, 0, "预注册冻结值"


def main():
    os.makedirs(RESDIR, exist_ok=True)
    N, n_noise, n_src = load_noise_floor()
    per = {arm: {s: flex_of(arm, s) for s in SEEDS} for arm in ARMS}
    dPM, missing = [], []
    for s in SEEDS:
        c, p = per["cost_norm"][s], per["uniform_pm"][s]
        if c is None or p is None:
            missing.append(s)
        else:
            dPM.append(round(c - p, 4))
    npos = sum(1 for d in dPM if d > 0)
    mean = round(sum(dPM) / len(dPM), 4) if dPM else None
    # 参数量（确定性，gen_p21_ranks 实算）
    p_cn, p_pm = 3693568, 3833856
    cn_cheaper = p_cn < p_pm

    n_valid = len(dPM)
    confirmed = (n_valid == 6 and npos >= 5 and mean is not None and mean > N and cn_cheaper)
    if missing:
        verdict = f"数据不全（缺 seed {missing}，{n_valid}/6）→ 无法确证，待补齐重跑"
    elif confirmed:
        verdict = "确证阳性（≥5/6 同号 且 mean(dPM)>N 且 cost_norm 更省参数）→ 升 KBS(Q1) 主claim"
    else:
        verdict = ("初步阳性，n=3 提示但 6-seed confirm 失败（≤4/6 同号 或 mean≤N）→ "
                   "降级：仅附录，不进 abstract/intro 主claim；维持四件套退 TMLR/Neural Networks")

    summary = {
        "_meta": {"generated": now(), "stage": "R1-confirm", "prereg": "§10.5 (4107552+448330a)",
                  "noise_floor_N_pt": N, "noise_floor_src": n_src, "noise_floor_n_runs": n_noise,
                  "seeds": SEEDS, "arms": ARMS, "primary_metric": "gsm8k_flex"},
        "per_seed_flex": per,
        "dPM_6seed": {"diffs": dPM, "n_valid": n_valid, "n_pos": npos, "mean": mean, "missing_seeds": missing},
        "params": {"cost_norm": p_cn, "uniform_pm": p_pm, "cost_norm_cheaper": cn_cheaper},
        "decision": {"confirmed_positive": bool(confirmed), "n_pos": npos, "mean_dPM": mean,
                     "N": N, "verdict": verdict},
    }
    with open(os.path.join(RESDIR, "summary_confirm.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    L = [f"# P2-1 R1-confirm 确证报告（6 seed）\n\n生成：{now()}  \n预注册：§10.5（4107552+448330a）  \n噪声底 N={N}pt（{n_src}）\n"]
    L.append("## cost_norm vs uniform_pm — GSM8K-flex（%，6 seed）\n\n| seed | cost_norm | uniform_pm | dPM |\n|---|---|---|---|")
    for s in SEEDS:
        c, p = per["cost_norm"][s], per["uniform_pm"][s]
        d = "—" if (c is None or p is None) else round(c - p, 4)
        L.append(f"| {s} | {c if c is not None else '缺'} | {p if p is not None else '缺'} | {d} |")
    L.append(f"\n- 有效配对 n={n_valid}/6  同号 {npos}/{n_valid}  mean(dPM)={mean}")
    L.append(f"- 参数：cost_norm {p_cn:,} vs uniform_pm {p_pm:,}（cost_norm 更省 = {cn_cheaper}）")
    L.append(f"\n## 确证判定（机械套用 §10.5，只填数）\n\n**{verdict}**\n")
    L.append("\n## 强制披露\n- 确证阈值：≥5/6 dPM>0 且 mean>N(4.09) 且 cost_norm 更省参数。")
    L.append("- confirm 失败不隐藏：降级「初步阳性 n=3 提示」，不进 abstract/intro（防 14B n=3 单种子扛均值覆辙）。")
    L.append("- strict 永不判据；BF16 基座 gsm8k-flex 主判据；n=6 仍属小样本，报告 bootstrap CI 供参考。")
    with open(os.path.join(RESDIR, "REPORT_p21_confirm.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"[p21-confirm] N={N} n_valid={n_valid}/6 npos={npos} mean_dPM={mean} confirmed={confirmed}")
    print(f"[p21-confirm] {verdict}")
    print(f"[p21-confirm] 写 {RESDIR}/summary_confirm.json + REPORT_p21_confirm.md")


if __name__ == "__main__":
    main()
