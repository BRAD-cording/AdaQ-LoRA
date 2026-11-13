#!/usr/bin/env python3
"""P0-E1/E2: 系统收益汇总（训练显存峰值 + adapter 参数量 + 可训练参数占比）。

不重跑训练，直接读已有产物：
  E1 训练显存峰值：outputs/adapters/<method>_seed42/training_stats.json 的 peak_vram_gb
  E2 adapter 参数量：outputs/adapters/<method>_seed42/adapter_model.safetensors 数参
  E2 可训练参数占比：adapter 参数量 / 基座总参数量（Llama-3-8B = 8,030,261,248）

诚实口径：AdaQ 预算守恒（512 rank），系统收益是「同预算下精度更高」，
不是「更低显存/延迟」。本脚本只记录事实，不虚构「更省显存」。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["qlora", "dora", "adalora", "adaq"]
BASE_PARAMS = 8_030_261_248  # Meta-Llama-3-8B 总参数量


def count_adapter_params(method):
    from safetensors import safe_open
    f = os.path.join(ROOT, "outputs", "adapters", f"{method}_seed42", "adapter_model.safetensors")
    if not os.path.isfile(f):
        return None
    total = 0
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            total += sf.get_tensor(k).numel()
    return total


def read_peak_vram(method):
    f = os.path.join(ROOT, "outputs", "adapters", f"{method}_seed42", "training_stats.json")
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        d = json.load(fh)
    return d.get("peak_vram_gb")


def main():
    rows = {}
    for m in METHODS:
        params = count_adapter_params(m)
        vram = read_peak_vram(m)
        ratio = (params / BASE_PARAMS) if params else None
        rows[m] = {
            "adapter_params": params,
            "peak_vram_gb": vram,
            "trainable_ratio": ratio,
        }
        print(f"{m:<10} params={params:>10,}  peak_vram={vram}GB  "
              f"trainable_ratio={ratio:.6f}" if ratio else f"{m}: adapter missing")

    out = os.path.join(ROOT, "results_v2", "system_metrics.json")
    with open(out, "w") as fh:
        json.dump({"base_params": BASE_PARAMS, "methods": rows}, fh, indent=2)
    print(f"\n系统收益汇总 -> {out}")


if __name__ == "__main__":
    main()
