"""汇总 P1-B4 (Qwen2.5-14B) 三个种子：adaq vs qlora，输出 summary_3seeds.json + REPORT_14B.md"""
import json, os, glob, statistics as st

BASE = "/home/ubuntu/AdaQ-LoRA"
RUNS = ["adaq", "qlora"]
SEED_DIRS = {42: "results_v2_qwen14b",
             2025: "results_v2_qwen14b_seed2025",
             3407: "results_v2_qwen14b_seed3407"}
PARAMS = {"adaq": 4620288, "qlora": 6291456}
OUT_JSON = os.path.join(BASE, "results_v2_qwen14b/summary_3seeds.json")
OUT_MD = os.path.join(BASE, "results_v2_qwen14b/REPORT_14B.md")


def collect(seed, run):
    d = os.path.join(BASE, SEED_DIRS[seed], run)
    m = {"mmlu": None, "gsm8k_strict": None, "gsm8k_flex": None}
    for f in glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True):
        try:
            res = json.load(open(f))["results"]
        except Exception:
            continue
        if "mmlu" in res:
            v = res["mmlu"].get("acc,none")
            if v is not None:
                m["mmlu"] = v
        for k in res:
            if k.startswith("gsm8k"):
                r = res[k]
                if r.get("exact_match,strict-match") is not None:
                    m["gsm8k_strict"] = r["exact_match,strict-match"]
                if r.get("exact_match,flexible-extract") is not None:
                    m["gsm8k_flex"] = r["exact_match,flexible-extract"]
    return m


def main():
    data = {}
    for seed in SEED_DIRS:
        for run in RUNS:
            data[f"{run}_s{seed}"] = collect(seed, run)

    metrics = ["mmlu", "gsm8k_strict", "gsm8k_flex"]
    agg = {}
    for run in RUNS:
        agg[run] = {"params": PARAMS[run]}
        for mt in metrics:
            vals = [data[f"{run}_s{s}"][mt] for s in SEED_DIRS if data[f"{run}_s{s}"][mt] is not None]
            agg[run][mt] = {
                "per_seed": {str(s): data[f"{run}_s{s}"][mt] for s in SEED_DIRS},
                "mean": st.mean(vals) if vals else None,
                "std": (st.stdev(vals) if len(vals) > 1 else 0.0) if vals else None,
                "n": len(vals),
            }
    # 配对差：同种子下 adaq - qlora
    paired = {}
    for mt in metrics:
        diffs = {}
        for s in SEED_DIRS:
            a, q = data[f"adaq_s{s}"][mt], data[f"qlora_s{s}"][mt]
            diffs[str(s)] = (a - q) if (a is not None and q is not None) else None
        vals = [v for v in diffs.values() if v is not None]
        paired[mt] = {"per_seed": diffs,
                      "mean": st.mean(vals) if vals else None,
                      "all_positive": all(v > 0 for v in vals) if vals else None,
                      "all_negative": all(v < 0 for v in vals) if vals else None}

    out = {"model": "Qwen2.5-14B-Instruct", "seeds": list(SEED_DIRS),
           "eval": "NF4 (load_in_4bit, nf4, double_quant), dtype=bfloat16",
           "train_data": "teknium/OpenHermes-2.5[:2000]", "methods": agg, "paired_diff_adaq_minus_qlora": paired}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

    L = ["# P1-B4 规模泛化：Qwen2.5-14B（3 种子）", "",
         "模型 Qwen2.5-14B-Instruct；SFT `teknium/OpenHermes-2.5[:2000]`；评测 NF4（load_in_4bit + nf4 + double_quant，compute bf16）。", "",
         "## 逐种子明细", "",
         "| seed | method | adapter 参数 | MMLU | GSM8K strict | GSM8K flex |", "| --- | --- | --- | --- | --- | --- |"]
    for s in SEED_DIRS:
        for run in RUNS:
            m = data[f"{run}_s{s}"]
            f = lambda v: f"{v:.4f}" if v is not None else "-"
            L.append(f"| {s} | {run} | {PARAMS[run]/1e6:.2f} M | {f(m['mmlu'])} | {f(m['gsm8k_strict'])} | {f(m['gsm8k_flex'])} |")
    L += ["", "## 均值 ± 标准差（n=3）", "",
          "| method | MMLU | GSM8K strict | GSM8K flex |", "| --- | --- | --- | --- |"]
    for run in RUNS:
        cells = []
        for mt in metrics:
            v = agg[run][mt]
            cells.append(f"{v['mean']:.4f} ± {v['std']:.4f}" if v["mean"] is not None else "-")
        L.append(f"| {run} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L += ["", "## 配对差（同种子 adaq − qlora）", "",
          "| metric | 42 | 2025 | 3407 | mean | 三种子同号? |", "| --- | --- | --- | --- | --- | --- |"]
    for mt in metrics:
        p = paired[mt]
        f = lambda v: f"{v:+.4f}" if v is not None else "-"
        same = "是(全正)" if p["all_positive"] else ("是(全负)" if p["all_negative"] else "否")
        L.append(f"| {mt} | {f(p['per_seed']['42'])} | {f(p['per_seed']['2025'])} | {f(p['per_seed']['3407'])} | {f(p['mean'])} | {same} |")
    L += ["", "## 判定规则（先写死，再看数）", "",
          "1. 若 **gsm8k_flex 的配对差三种子同号且均值 > 单次跑间噪声（4.5 点，来自 8B product/importance_only 同配置两次训练的经验值）**，才能写「14B 上 AdaQ 相对 QLoRA 有正向差异」。",
          "2. 若三种子符号不一致，或均值落在 ±4.5 点内，只能写「未观察到一致差异」。",
          "3. strict-match 一律不作为判据（Qwen-Instruct 格式不合规，strict 度量格式而非能力）。",
          "4. adaq 参数比 qlora 少 26.5%，任何比较都必须同时报参数量。", ""]
    open(OUT_MD, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
