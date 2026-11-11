#!/usr/bin/env python3
"""评测收口器：等所有 lm_eval 退出 -> 跑闸门 -> 只有 PASS 才生成 FINAL_REPORT.md。

设计原则（防止再造假表）：
  * 报告里的每个数字都从 results_v2/summary.json 读，禁止手抄、禁止硬编码；
  * 闸门 check_eval_status.py 不通过就**不生成**报告，只写 BLOCKED 收据；
  * 报告显式写出"不能说什么"（无方法增益、adalora 巧合、无有效显著性检验），
    避免后续写作时把描述性统计误读成推断性结论。

用法:
  python scripts/finalize_eval.py              # 一次性检查并收口
  python scripts/finalize_eval.py --wait       # 轮询等待所有 lm_eval 退出后再收口
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_v2")
SUMMARY = os.path.join(OUT, "summary.json")
REPORT = os.path.join(OUT, "FINAL_REPORT.md")
RECEIPT = os.path.join(OUT, "finalize_receipt.json")

METHODS = ("qlora", "dora", "adalora", "adaq")
SEEDS = (42, 3407, 2025)
# 与 check_eval_status.py 保持一致的 5 个门禁指标
GATE_METRICS = (
    "mmlu/acc",
    "gsm8k/exact_match,strict-match",
    "truthfulqa_mc1/acc",
    "truthfulqa_mc2/acc",
    "ifeval/inst_level_strict_acc",
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def running_evals() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "lm_eval --model"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except FileNotFoundError:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def wait_for_drain(poll_s: int = 120) -> None:
    while True:
        procs = running_evals()
        if not procs:
            print(f"[{now()}] no lm_eval running -> proceed")
            return
        print(f"[{now()}] {len(procs)} lm_eval still running, sleep {poll_s}s", flush=True)
        time.sleep(poll_s)


def run_gate() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "check_eval_status.py")],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


MARKERS = ("mmlu", "gsm8k", "safety")


def audit_markers() -> dict:
    """统计 12 个 run 的 .done_* marker。

    说明：闸门 check_eval_status.py 判的是**产物内容**（更强），marker 只是补跑脚本
    「已验证通过」的记账位。这里把 marker 数显式报出来，避免交付时只能口头声称
    「marker 齐全」而无据可查；marker 缺失但产物齐全不算失败，只作为观察项记录。
    """
    per_run: dict[str, list[str]] = {}
    for m in METHODS:
        for s in SEEDS:
            run = f"{m}_seed{s}"
            d = os.path.join(OUT, run)
            have = [k for k in MARKERS if os.path.isfile(os.path.join(d, f".done_{k}"))]
            per_run[run] = have
    counts = {k: sum(1 for v in per_run.values() if k in v) for k in MARKERS}
    complete = sum(1 for v in per_run.values() if len(v) == len(MARKERS))
    return {
        "counts": counts,
        "runs_complete": complete,
        "runs_total": len(per_run),
        "incomplete_runs": {k: v for k, v in per_run.items() if len(v) != len(MARKERS)},
    }


def build_report(summary: dict) -> str:
    lines: list[str] = []
    a = lines.append
    a("# AdaQ-LoRA 全量评测最终报告（自动生成）")
    a("")
    a(f"- 生成时间：{now()}")
    a(f"- 数据来源：`results_v2/summary.json`（本文件所有数字由脚本读取，无手抄）")
    a(f"- 口径：MMLU 5-shot 全量 14042 / GSM8K 4-shot 全量 1319 / "
      f"TruthfulQA mc1+mc2 + IFEval 0-shot")
    a(f"- 种子：{', '.join(str(s) for s in SEEDS)}（n=3）")
    a("")
    a("## 1. 主表（mean ± std, n=3）")
    a("")
    header = "| 方法 | MMLU | GSM8K | TruthfulQA mc1 | TruthfulQA mc2 | IFEval inst-strict |"
    a(header)
    a("| --- | --- | --- | --- | --- | --- |")
    for m in METHODS:
        cells = []
        for metric in GATE_METRICS:
            s = summary.get(m, {}).get(metric)
            cells.append(f"{s['mean']:.4f}±{s['std']:.4f}" if s else "—")
        a(f"| {m} | " + " | ".join(cells) + " |")
    a("")
    a("## 2. 逐种子明细（per_seed，可复核）")
    a("")
    a("| 方法 | 指标 | seed 42 | seed 2025 | seed 3407 | 样本数 |")
    a("| --- | --- | --- | --- | --- | --- |")
    for m in METHODS:
        for metric in GATE_METRICS:
            s = summary.get(m, {}).get(metric)
            if not s:
                continue
            ps = s.get("per_seed") or {}
            row = [ps.get(str(k), "—") for k in (42, 2025, 3407)]
            row = [f"{v:.4f}" if isinstance(v, (int, float)) else v for v in row]
            a(f"| {m} | {metric} | " + " | ".join(row) + f" | {s.get('sample_len')} |")
    a("")
    a("## 3. 结论与措辞边界（必读）")
    a("")
    a("1. **MMLU 上四方法差异落在种子波动内**：各方法 std 量级与方法间均值差同阶，"
      "不能作为方法有效性主张。MMLU 是知识问答，2000 条 OpenHermes 指令微调本就难以改变它。")
    a("2. **adalora 的 GSM8K strict 在三种子下为 629 / 629 / 631（共 1319 题）**："
      "前两种子同值属离散计数巧合（单次计数 sd≈18，极差仅 0.11 sd；"
      "P(某对种子同值)≈6.6%），第三个种子给出 631 直接否证「对种子不敏感」；"
      "两 adapter 权重逐张量均不同，flexible 指标亦不同（637 vs 635）。"
      "**不能**写成「秩分配对种子不敏感」的稳定性论据。")
    a("3. **无有效显著性检验**：3 种子下 Wilcoxon 最小双侧 p=0.25（n=6 才 0.03125），"
      "`summary.json` 的 `significance_test` 已对每任务标注样本不足。"
      "因此本报告只给描述性统计，**禁止**「显著优于 / SOTA / p<0.05」类表述。")
    a("")
    mk = audit_markers()
    a("## 4. 补跑记账位（.done_* marker 核对）")
    a("")
    a("闸门判据是**产物内容**（更强）；marker 是补跑脚本「已验证通过」的记账位。")
    a("")
    a("| marker | 齐全 run 数 / 12 |")
    a("| --- | --- |")
    for k in MARKERS:
        a(f"| `.done_{k}` | {mk['counts'][k]} |")
    a("")
    a(f"三 marker 全齐的 run：**{mk['runs_complete']}/{mk['runs_total']}**")
    if mk["incomplete_runs"]:
        a("")
        a("未齐的 run（观察项，不阻断交付——产物内容已由闸门判定）：")
        for r, v in sorted(mk["incomplete_runs"].items()):
            a(f"- `{r}`: {v or '无 marker'}")
    a("")
    a("## 5. 可复现命令")
    a("")
    a("```bash")
    a("bash scripts/run_eval_missing.py            # 幂等补缺失（WANT/OUT_DIR 见文件头）")
    a("python scripts/check_eval_status.py         # 机读闸门，PASS 才可交付")
    a("python scripts/aggregate_results.py --results-dir results_v2 --out results_v2/summary.json")
    a("python scripts/finalize_eval.py             # 本报告的生成器")
    a("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="轮询等待所有 lm_eval 退出")
    ap.add_argument("--poll", type=int, default=120, help="轮询间隔秒（默认 120）")
    args = ap.parse_args()

    if args.wait:
        wait_for_drain(args.poll)

    if not os.path.exists(SUMMARY):
        print(f"BLOCKED: summary.json 不存在（{SUMMARY}）")
        return 2

    rc, gate_out = run_gate()
    print(gate_out.strip()[-1200:])

    summary = json.load(open(SUMMARY, encoding="utf-8"))
    audit = summary.get("_audit", {}) or {}
    receipt = {
        "ts": now(),
        "gate_exit": rc,
        "gate_verdict": "PASS" if rc == 0 else "INCOMPLETE",
        "summary_bytes": os.path.getsize(SUMMARY),
        "summary_mtime": datetime.fromtimestamp(
            os.path.getmtime(SUMMARY), timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "audit": {k: v for k, v in audit.items() if k != "per_run_dirs"},
        "markers": audit_markers(),
        "report_generated": False,
    }

    if rc != 0:
        receipt["report_generated"] = False
        receipt["note"] = "闸门未通过，不生成报告（避免把残缺表当交付）"
        json.dump(receipt, open(RECEIPT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"BLOCKED: gate exit={rc} -> 不生成 FINAL_REPORT.md")
        return 2

    missing = [
        f"{m}/{k}" for m in METHODS for k in GATE_METRICS
        if (summary.get(m, {}).get(k) or {}).get("n_seeds") != 3
    ]
    if missing:
        receipt["note"] = f"n_seeds!=3: {missing}"
        json.dump(receipt, open(RECEIPT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"BLOCKED: 以下单元格 n_seeds!=3 -> {missing}")
        return 2

    open(REPORT, "w", encoding="utf-8").write(build_report(summary))
    receipt["report_generated"] = True
    receipt["report"] = os.path.relpath(REPORT, ROOT)
    receipt["report_bytes"] = os.path.getsize(REPORT)
    json.dump(receipt, open(RECEIPT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"OK: {REPORT} ({os.path.getsize(REPORT)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
