#!/usr/bin/env python3
"""F2: baseline_bf16 锚点 — Meta-Llama-3-8B BF16 干净基座、无 adapter。
主表 12 run 全部缺这一行（review/fair-baseline.md F2、review/jcr-gate.md P0）。
用法: python run_baseline_bf16.py --seed 42 --task mmlu   (GPU 由外层 CUDA_VISIBLE_DEVICES 控制)
幂等: .done_<task> 存在即跳过。产物: results_v2/baseline_bf16_seed<seed>/results_<ts>.json
"""
import argparse, datetime, json, os, time

BASE = "/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B"
OUT_ROOT = "results_v2"
FEWSHOT = {"mmlu": 5, "gsm8k": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", choices=["mmlu", "gsm8k"], required=True)
    args = ap.parse_args()

    out = os.path.join(OUT_ROOT, f"baseline_bf16_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, f".done_{args.task}")
    if os.path.exists(done):
        print(f"[skip] seed={args.seed} task={args.task} already done")
        return

    import lm_eval  # 延迟 import，让 .done 快路径零开销

    model_args = f"pretrained={BASE},dtype=bfloat16"
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[args.task],
        num_fewshot=FEWSHOT[args.task],
        batch_size="auto",
        seed=args.seed,
    )
    elapsed = time.time() - t0
    payload = {
        "results": res.get("results", {}),
        "n-samples": res.get("n-samples", {}),
        "configs": res.get("configs", {}),
        "config": {
            "model": "hf",
            "model_args": model_args,
            "task": args.task,
            "num_fewshot": FEWSHOT[args.task],
            "eval_seed": args.seed,
            "batch_size": "auto",
            "elapsed_sec": round(elapsed, 1),
            "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
            "note": "F2 anchor: BF16 clean base, no adapter. eval seed swept 42/3407/2025.",
            "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        },
    }
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    path = os.path.join(out, f"results_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    open(done, "w").close()
    r = payload["results"].get(args.task, {})
    print(f"[done] seed={args.seed} task={args.task} {elapsed:.0f}s -> {path}")
    print(f"       metrics: {json.dumps({k: v for k, v in r.items() if isinstance(v, (int, float))}, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
