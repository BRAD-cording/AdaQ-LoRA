#!/usr/bin/env python3
"""AdaQ-LoRA 全流程执行脚本（end-to-end pipeline runner）。

按 SKILL.md 四支柱顺序执行：
  1. 理论审查（Pillar 1） — 已完成，报告在 docs/theory-review-report.md
  2. 预计算 + SOTA 对比准备（Pillar 2）
  3. 训练 + 评测（Pillar 2-3）
  4. 结果汇总（Pillar 4）

用法:
  python run_pipeline.py --step precompute    # 预计算秩分配
  python run_pipeline.py --step train          # 训练所有方法
  python run_pipeline.py --step merge          # 合并 adapter
  python run_pipeline.py --step eval           # 批量评测
  python run_pipeline.py --step aggregate      # 汇总结果
  python run_pipeline.py --step all            # 全流程
  python run_pipeline.py --step test           # 无 GPU 测试流程
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPTS = os.path.join(PROJECT_ROOT, "adaq-lora-research-pipeline", "scripts")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def run_cmd(cmd, desc=""):
    """运行命令并打印状态。"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} 完成 ({elapsed:.1f}s, exit={result.returncode})")
    return result.returncode == 0


def step_test():
    """无 GPU 环境下的测试流程。"""
    print("\n" + "=" * 60)
    print("  AdaQ-LoRA 无 GPU 测试流程")
    print("=" * 60)

    results = {}

    # 1. 预计算自测
    results["self_test"] = run_cmd(
        ["python3", os.path.join(PIPELINE_SCRIPTS, "adaq_precompute.py"), "--self-test"],
        "Step 1: 预计算分配逻辑自测"
    )

    # 2. 命题1' 数值验证
    results["prop1_verify"] = run_cmd(
        ["python3", os.path.join(SCRIPTS_DIR, "verify_proposition1_spectral_elevation.py")],
        "Step 2: 命题1'（平均谱抬升）数值验证"
    )

    # 3. 显存预算
    results["memory_budget"] = run_cmd(
        ["python3", os.path.join(SCRIPTS_DIR, "memory_budget.py"), "--interactive"],
        "Step 3: A10 显存预算扫描"
    )

    # 4. 生成合成 ranks
    results["synthetic_ranks"] = run_cmd(
        ["python3", os.path.join(SCRIPTS_DIR, "generate_synthetic_ranks.py"), "--budget", "512", "--out", "ranks.json"],
        "Step 4: 生成合成秩分配（无 GPU 替代）"
    )

    # 5. 生成消融配置
    results["ablation_configs"] = run_cmd(
        ["python3", os.path.join(SCRIPTS_DIR, "generate_ablation_configs.py"), "--base-ranks", "ranks.json", "--budget", "512"],
        "Step 5: 生成消融实验配置"
    )

    # 总结
    print("\n" + "=" * 60)
    print("  测试流程总结")
    print("=" * 60)
    all_pass = True
    for step, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {step}")
        if not ok:
            all_pass = False

    print(f"\n  {'所有测试通过 ✅' if all_pass else '部分测试失败 ❌，请检查上方日志'}")
    if all_pass:
        print("\n  下一步（需 GPU）:")
        print("    1. python run_pipeline.py --step precompute  # 真实预计算")
        print("    2. python run_pipeline.py --step train        # 训练")
        print("    3. python run_pipeline.py --step merge       # 合并")
        print("    4. python run_pipeline.py --step eval         # 评测")
        print("    5. python run_pipeline.py --step aggregate   # 汇总")
    return all_pass


def step_precompute():
    """预计算真实秩分配（需 GPU + bitsandbytes）。"""
    ok = run_cmd(
        ["python3", os.path.join(PIPELINE_SCRIPTS, "adaq_precompute.py"),
         "--model", "NousResearch/Meta-Llama-3-8B",
         "--budget", "512", "--r-min", "4", "--r-max", "128",
         "--out", "ranks.json"],
        "预计算: 逐层 NF4 量化误差 + 秩分配"
    )
    if ok:
        run_cmd(
            ["python3", os.path.join(SCRIPTS_DIR, "generate_ablation_configs.py"),
             "--base-ranks", "ranks.json", "--budget", "512"],
            "消融配置生成"
        )
    return ok


def step_train():
    """训练所有方法 × 种子。"""
    methods = ["qlora", "dora", "adalora", "adaq"]
    seeds = [42, 3407, 2025]
    all_ok = True
    for method in methods:
        for seed in seeds:
            ok = run_cmd(
                ["python3", os.path.join(SCRIPTS_DIR, "train.py"),
                 "--method", method, "--seed", str(seed),
                 "--budget", "512", "--base-rank", "8",
                 "--epochs", "3", "--dataset-size", "2000"],
                f"训练: {method} (seed={seed})"
            )
            if not ok:
                all_ok = False
    return all_ok


def step_merge():
    """合并所有 adapter 到 bf16（可选，默认跳过，评测时直接加载 adapter）。"""
    methods = ["qlora", "dora", "adalora", "adaq"]
    seeds = [42, 3407, 2025]
    all_ok = True
    for method in methods:
        for seed in seeds:
            adapter_dir = f"outputs/adapters/{method}_seed{seed}"
            if not os.path.exists(adapter_dir):
                print(f"  [SKIP] {method} seed={seed}: adapter not found")
                continue
            ok = run_cmd(
                ["python3", os.path.join(SCRIPTS_DIR, "merge_adapter.py"),
                 "--method", method, "--seed", str(seed)],
                f"合并: {method} (seed={seed})"
            )
            if not ok:
                all_ok = False
    return all_ok


def step_eval():
    """批量评测（使用 adapter 模式，无需 merge）。"""
    env = os.environ.copy()
    env["EVAL_MODE"] = "adapter"
    # 默认正式全量评测：禁止静默进入 debug 模式（--limit）。
    # 快速冒烟测试请显式设置 QUICK=1 与 QUICK_LIMIT。
    env["QUICK"] = env.get("QUICK", "0")
    env["QUICK_LIMIT"] = env.get("QUICK_LIMIT", "100")
    result = subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "run_eval_batch.sh")],
        cwd=PROJECT_ROOT, env=env
    )
    return result.returncode == 0


def step_aggregate():
    """汇总结果。"""
    return run_cmd(
        ["python3", os.path.join(SCRIPTS_DIR, "aggregate_results.py"),
         "--baseline", "qlora", "--out", "results/summary.json"],
        "汇总: 均值±std + Wilcoxon 显著性检验"
    )


def step_all():
    """全流程执行（跳过 merge，直接用 adapter 评测）。"""
    steps = [
        ("precompute", step_precompute),
        ("train", step_train),
        ("eval", step_eval),
        ("aggregate", step_aggregate),
    ]
    all_ok = True
    for name, func in steps:
        if not func():
            print(f"\n❌ Step '{name}' failed, stopping.")
            all_ok = False
            break
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="AdaQ-LoRA end-to-end pipeline")
    ap.add_argument("--step", required=True,
                    choices=["precompute", "train", "merge", "eval", "aggregate", "all", "test"])
    args = ap.parse_args()

    steps = {
        "precompute": step_precompute,
        "train": step_train,
        "merge": step_merge,
        "eval": step_eval,
        "aggregate": step_aggregate,
        "all": step_all,
        "test": step_test,
    }

    ok = steps[args.step]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
