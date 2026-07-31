#!/usr/bin/env python3
"""批量训练：4 方法 × 3 种子 × 3 epoch × 2000 样本。

按 SKILL.md 铁律2（公平先于领先）：所有方法共享同一数据、epoch、lr、量化配置。
唯一差异：秩分配方式。

用法: python batch_train.py [--epochs 3] [--dataset-size 2000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train.py")

METHODS = ["qlora", "dora", "adalora", "adaq"]
SEEDS = [42, 3407, 2025]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dataset-size", type=int, default=2000)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()

    total = len(args.methods) * len(args.seeds)
    done = 0
    failed = []
    start_all = time.time()

    print(f"=== Batch Training: {len(args.methods)} methods × {len(args.seeds)} seeds = {total} runs ===")
    print(f"    Epochs: {args.epochs}, Dataset size: {args.dataset_size}")
    print()

    for method in args.methods:
        for seed in args.seeds:
            done += 1
            output_dir = f"outputs/adapters/{method}_seed{seed}"
            stats_file = os.path.join(output_dir, "training_stats.json")

            # Skip if already trained with matching settings
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    stats = json.load(f)
                if stats.get("epochs", 0) >= args.epochs:
                    print(f"[{done}/{total}] SKIP {method}_seed{seed} (already trained)")
                    continue

            print(f"\n[{done}/{total}] Training {method} (seed={seed}, epochs={args.epochs})...")
            t0 = time.time()

            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--method", method,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--dataset-size", str(args.dataset_size),
                "--output-dir", output_dir,
            ]

            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)

            elapsed = time.time() - t0
            status = "✅" if result.returncode == 0 else "❌"
            print(f"  {status} {method}_seed{seed} done in {elapsed:.0f}s")

            if result.returncode != 0:
                failed.append(f"{method}_seed{seed}")

            # Record epochs in stats
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    stats = json.load(f)
                stats["epochs"] = args.epochs
                stats["dataset_size"] = args.dataset_size
                with open(stats_file, "w") as f:
                    json.dump(stats, f, indent=2)

    total_time = time.time() - start_all
    print(f"\n{'='*60}")
    print(f"Batch training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Total: {total}, Done: {done - len(failed)}, Failed: {len(failed)}")
    if failed:
        print(f"  Failed runs: {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
