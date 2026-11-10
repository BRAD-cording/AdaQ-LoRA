#!/usr/bin/env python3
"""校验某个 run 的 results_*.json 是否真的包含目标任务的指标。

v2.0 缺陷：lm_eval 命令返回 0 但结果 JSON 可能只有另一个任务的指标
（例如 mmlu 因 429 失败、随后 gsm8k 成功共用 .done_main 把失败掩盖）。
本脚本在 touch marker 前做机读校验，杜绝"命令绿了但数据没到"。

用法: python verify_eval_artifact.py <run_dir> <marker>
exit 0 = 通过；1 = 缺失 / 无法解析
"""
import glob
import json
import os
import sys

PROBE = {
    "mmlu": ("mmlu",),
    "gsm8k": ("gsm8k",),
    "safety": ("truthfulqa_mc1", "truthfulqa_mc2", "ifeval"),
}


def main():
    if len(sys.argv) != 3:
        print("usage: verify_eval_artifact.py <run_dir> <marker>", file=sys.stderr)
        return 1
    run_dir, marker = sys.argv[1], sys.argv[2]
    need = PROBE.get(marker)
    if not need:
        print(f"unknown marker: {marker}", file=sys.stderr)
        return 1
    files = sorted(
        glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not files:
        print(f"no results_*.json under {run_dir}", file=sys.stderr)
        return 1
    # 补跑产物是独立的时间戳文件，不覆盖旧文件 -> 合并所有 json 的 results 键
    have = set()
    for f in files:
        try:
            with open(f) as fh:
                have |= set((json.load(fh).get("results") or {}).keys())
        except (OSError, json.JSONDecodeError):
            continue
    missing = [t for t in need if t not in have]
    if missing:
        print(
            f"missing {missing} in {run_dir} (latest {os.path.basename(files[-1])})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
