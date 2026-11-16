#!/usr/bin/env python3
"""run_f18_rtn_baseline.py — F18：真 INT4-RTN-g128 PTQ 基线（无校准，CPU round-trip）。

背景：fair-baseline F18 指出 baseline_nf4 实为 bnb NF4+DQ（非 PTQ 文献的 INT4-RTN-g128），
命名已更正为 base_nf4dq_noadapter（b0d4dde）。本脚本补一个**真正的 INT4-RTN-g128** PTQ 锚点。

为何不用 auto-gptq：①sdist build 失败（build isolation 看不到 torch，需 --no-build-isolation +
CUDA 扩展编译，0.7.1 已归档、与 transformers 4.57.6 运行期大概率不兼容）；②auto-gptq 是 GPTQ
（校准-based，Hessian 误差补偿）工具，**纯 RTN（round-to-nearest，无校准）根本不需要它**。
INT4-RTN-g128 的标准定义 = 对称分组 round-to-nearest 量化-反量化 round-trip，纯 PyTorch 实现，
零依赖、完全可复现、比黑盒工具更 defensible（量化公式全显式）。

量化方案（对称 INT4，group=128，沿输入维分组，GPTQ 惯例）：
  每 group(128 个输入特征)：scale = max|w_g| / 7（对称，int4 正端 7）
  q = round(w / scale) 截断到 [-8, 7]；w' = q * scale（反量化回 bf16）
应用范围：所有 nn.Linear 权重，**排除 lm_head**（GPTQ/PTQ 文献惯例，lm_head 量化收益低且伤大）。
round-trip 后保存为普通 bf16 模型 → lm_eval 当 pretrained 评测（无 peft、无 load_in_4bit）。

用法：
  python run_f18_rtn_baseline.py --quantize          # CPU 量化 round-trip + 保存（无需 GPU）
  python run_f18_rtn_baseline.py --quantize --verify # 附带打印量化前后权重误差统计
"""
import argparse, os, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B"
OUT  = "/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B-int4rtn-g128"
GROUP = 128
INT4_MIN, INT4_MAX = -8, 7  # 对称 int4


def rtn_int4_g128_roundtrip(w: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """对称 INT4 group-128 round-to-nearest 量化-反量化 round-trip。w: [out, in]。"""
    out_f, in_f = w.shape
    orig_dtype = w.dtype
    wf = w.float()
    pad = (group - in_f % group) % group
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    wg = wf.view(out_f, -1, group)                          # [out, n_groups, group]
    scale = wg.abs().amax(dim=-1, keepdim=True) / float(INT4_MAX)   # 对称：max|w|/7
    scale = torch.clamp(scale, min=1e-8)
    q = torch.round(wg / scale).clamp(INT4_MIN, INT4_MAX)
    deq = (q * scale).view(out_f, -1)[:, :in_f]
    return deq.to(orig_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="CPU round-trip 量化 + 保存模型")
    ap.add_argument("--verify", action="store_true", help="打印量化前后权重误差统计")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not args.quantize:
        print("用 --quantize 运行"); return

    if os.path.exists(os.path.join(args.out, "config.json")):
        print(f"[f18] RTN 模型已存在 {args.out}，跳过量化（幂等）"); return

    print(f"[f18] 加载 {args.base}（CPU, bf16）...")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu")

    n_lin = 0; skipped = []; err_stats = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if "lm_head" in name:
                skipped.append(name); continue
            with torch.no_grad():
                w0 = mod.weight.data
                w1 = rtn_int4_g128_roundtrip(w0, GROUP)
                if args.verify:
                    rel = ((w1.float() - w0.float()).norm() / w0.float().norm().clamp(min=1e-8)).item()
                    err_stats.append((name, round(rel, 5)))
                mod.weight.copy_(w1)
            n_lin += 1
    print(f"[f18] round-trip {n_lin} 个 linear（对称 INT4 g128）；排除 lm_head={skipped}")
    if args.verify and err_stats:
        rels = [e[1] for e in err_stats]
        print(f"[f18] 权重相对误差 ‖W'−W‖/‖W‖：min={min(rels):.5f} max={max(rels):.5f} "
              f"mean={sum(rels)/len(rels):.5f}（INT4-RTN-g128 典型 ~0.01-0.03）")
        for nm, r in err_stats[:4]:
            print(f"    {nm}: {r}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "rtn_provenance.json"), "w") as f:
        json.dump({
            "method": "INT4-RTN symmetric group-128 round-to-nearest (no calibration)",
            "base": args.base, "n_linear_quantized": n_lin, "lm_head_skipped": skipped,
            "int4_range": [INT4_MIN, INT4_MAX], "group_size": GROUP,
            "scale": "max|w_group| / 7 (symmetric)", "roundtrip_dtype": "bf16",
            "weight_rel_err": ({"min": min([e[1] for e in err_stats]),
                                 "max": max([e[1] for e in err_stats]),
                                 "mean": sum([e[1] for e in err_stats])/len(err_stats)}
                                if err_stats else "not computed (pass --verify)"),
            "why_not_auto_gptq": "auto-gptq sdist build failed (build-isolation no torch) + archived "
                                 "+ incompatible w/ transformers 4.57.6; auto-gptq is GPTQ (calibration) "
                                 "not RTN; pure RTN = explicit symmetric round-trip (this script).",
        }, f, indent=2)
    print(f"[f18] RTN 模型已保存 -> {args.out}")


if __name__ == "__main__":
    main()
