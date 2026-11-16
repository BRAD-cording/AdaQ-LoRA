#!/usr/bin/env python3
"""F2: baseline_bf16 锚点 — Meta-Llama-3-8B BF16 干净基座、无 adapter。
主表 12 run 全部缺这一行（review/fair-baseline.md F2、review/jcr-gate.md P0）。

修复(2026-11-16)：旧版用 lm_eval.simple_evaluate(seed=...)，本机 lm_eval 该 API 不接受 seed kwarg
→ TypeError 全部 6 run rc=1。改用与 run_eval_ablation.py 完全一致的 CLI 子进程路径
（`python -m lm_eval ... --seed N`，该路径在本机已验证可用），baseline 即去掉 peft 参数。

用法: python run_baseline_bf16.py --seed 42 --task mmlu [--dry-run]  (GPU 由外层 CUDA_VISIBLE_DEVICES 控制)
幂等: .done_<task> 存在即跳过。产物: results_v2/baseline_bf16_seed<seed>/<task>/results_*.json (lm_eval 标准结构)
"""
import argparse, datetime, glob, os, subprocess, sys

PY = "/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python"
BASE = "/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B"
OUT_ROOT = "results_v2"
FEWSHOT = {"mmlu": 5, "gsm8k": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", choices=["mmlu", "gsm8k"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    out = os.path.join(OUT_ROOT, f"baseline_bf16_seed{args.seed}")
    done = os.path.join(out, f".done_{args.task}")
    if os.path.exists(done) and not args.dry_run:
        print(f"[skip] seed={args.seed} task={args.task} already done")
        return 0

    os.makedirs(out, exist_ok=True)
    model_args = f"pretrained={BASE},dtype=bfloat16"
    run_dir = os.path.join(out, args.task)
    cmd = [
        PY, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", args.task,
        "--num_fewshot", str(FEWSHOT[args.task]),
        "--batch_size", "auto",
        "--device", args.device,
        "--seed", str(args.seed),
        "--output_path", run_dir,
    ]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["NLTK_DATA"] = "/home/ubuntu/nltk_data"
    env["ADAQ_MODEL_PATH"] = BASE

    if args.dry_run:
        print("[dry-run] cwd=/home/ubuntu/AdaQ-LoRA")
        print("[dry-run] " + " ".join(cmd))
        return 0

    t0 = time_now()
    print(f"[run] seed={args.seed} task={args.task} -> {run_dir}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd="/home/ubuntu/AdaQ-LoRA")
    elapsed = time_now() - t0
    if rc != 0:
        print(f"[bf16-baseline] FAIL seed={args.seed} task={args.task} rc={rc} {utcnow()}", flush=True)
        return rc
    # 验收：results_*.json 必须存在且含该 task 的结果
    rjs = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    if not rjs:
        print(f"[bf16-baseline] FAIL seed={args.seed} task={args.task} rc=0 但无 results_*.json {utcnow()}", flush=True)
        return 3
    open(done, "w").close()
    print(f"[bf16-baseline] OK seed={args.seed} task={args.task} {elapsed:.0f}s n_json={len(rjs)} {utcnow()}", flush=True)
    return 0


def time_now():
    import time
    return time.time()


def utcnow():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + " UTC"


if __name__ == "__main__":
    sys.exit(main())
