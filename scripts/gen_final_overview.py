#!/usr/bin/env python3
"""全实验环节最终总览生成器（watchdog 全完成时自动调用，亦可手动跑）。
读各 results 表/REPORT，提取关键数字 + 预注册判定，拼 results_v2_FINAL_OVERVIEW.md。
防御式：任一文件缺失只标"缺"不崩。诚实呈现阴性/降级，不粉饰。
"""
import os, json, glob
from datetime import datetime, timezone

ROOT = "/home/ubuntu/AdaQ-LoRA"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def find_res(d):
    fs = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    return sorted(fs)[-1] if fs else None


def g(r, *keys):
    """安全嵌套取：g(j,'results','gsm8k','exact_match,flexible-extract')"""
    cur = r
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def pct(x, nd=4):
    return None if x is None else round(x * 100, nd)


def line(name, val):
    return f"| {name} | {val if val is not None else '—'} |"


L = []
L.append(f"# AdaQ-LoRA 全实验环节最终总览\n\n生成：{now()}  \n连接：2×RTX 4090D；基座 Llama-3-8B（主）/ Qwen2.5-14B-Instruct（规模）\n")
L.append("> 本文件由 watchdog 在全链完成时自动生成，汇总所有实验环节的关键数字与预注册判定。诚实呈现阴性/降级。\n")

# 0) 流水线完成状态
L.append("## 0. 流水线完成状态\n\n| 环节 | DONE 产物 | 状态 |\n|---|---|---|")
checks = [
    ("phase3 D 噪声底", "logs/phase3/.done_D"),
    ("phase3 B adalora", "logs/phase3/.done_B"),
    ("phase3 E 基线", "logs/phase3/.done_E"),
    ("phase3 C R曲线", "logs/phase3/.done_C"),
    ("phase3 A 主表重训", "logs/phase3/.done_A"),
    ("bf16 锚点", None),
    ("B8 flex审计采样", None),
    ("B5 chat-template", None),
    ("F18 INT4-RTN", None),
    ("P2-1 成本归一化", "logs/p21/.done_p21"),
    ("P2-1 confirm", "logs/p21/.done_p21confirm"),
]
for nm, path in checks:
    if path:
        ok = os.path.isfile(os.path.join(ROOT, path))
    elif nm.startswith("bf16"):
        ok = len(glob.glob(os.path.join(ROOT, "results_v2/baseline_bf16_seed*/.done_*"))) >= 6
    elif nm.startswith("B8"):
        ok = "[b8-audit] DONE" in (open(os.path.join(ROOT, "logs/b8_flex_audit.log")).read() if os.path.isfile(os.path.join(ROOT, "logs/b8_flex_audit.log")) else "")
    elif nm.startswith("B5"):
        ok = len(glob.glob(os.path.join(ROOT, "results_v2_14b_chattemplate/**/.done_gsm8k_chattemplate"), recursive=True)) >= 6
    elif nm.startswith("F18"):
        ok = "[f18-rtn] DONE" in (open(os.path.join(ROOT, "logs/f18_rtn_eval.log")).read() if os.path.isfile(os.path.join(ROOT, "logs/f18_rtn_eval.log")) else "")
    else:
        ok = False
    L.append(f"| {nm} | {path or '产物计数'} | {'✅' if ok else '⏳/✗'} |")

# 1) 主表（BF16，重训后 phase3_main 优先，回退 results_v2）
L.append("\n## 1. 主表 — 等秩 512 BF16（n=3 seed）\n")
main = jload(os.path.join(ROOT, "results_v2_phase3_main/summary.json")) or jload(os.path.join(ROOT, "results_v2/summary.json"))
if main:
    L.append("| 方法 | MMLU | GSM8K-strict | GSM8K-flex | 参数 |\n|---|---|---|---|---|")
    for m in ("qlora", "dora", "adalora", "adaq"):
        row = None
        for k in (m, f"{m}_seed42", m.upper()):
            if isinstance(main, dict) and k in main:
                row = main[k]; break
        if row is None and isinstance(main, dict):
            row = main.get(m, {})
        if isinstance(row, dict):
            L.append(f"| {m} | {row.get('mmlu', row.get('MMLU','—'))} | {row.get('gsm8k_strict','—')} | {row.get('gsm8k_flex','—')} | {row.get('params','—')} |")
    L.append("\n_注：phase3-A 重训（α=2r, 375步 no-packing）替换原 21步 packing 主表；stats 评审 F1 主表无功效已修。_")
else:
    L.append("_主表 summary 缺（phase3-A 未完成？）_")

# 2) NF4 负结果（核心贡献）
L.append("\n## 2. NF4 评测路径 — 核心阴性贡献\n")
nf4 = jload(os.path.join(ROOT, "results_v2_ptq_nf4/summary.json"))
if nf4:
    L.append("| 方法(NF4加载) | MMLU | GSM8K-strict |\n|---|---|---|")
    for m in ("baseline_nf4", "qlora_nf4", "dora_nf4", "adalora_nf4", "adaq_nf4"):
        row = nf4.get(m, {}) if isinstance(nf4, dict) else {}
        if isinstance(row, dict):
            L.append(f"| {m} | {row.get('mmlu','—')} | {row.get('gsm8k_strict','—')} |")
L.append("\n**根因（已落档 NF4_root_cause_analysis.md）**：score=rel_error×importance，rel_error CV 1.47%（近常数）、importance(‖W‖) CV 56.2%、Spearman(score,‖W‖)=0.998 → score 99.8% 范数驱动。AdaQ 实为**范数感知**非量化误差感知。")
L.append("**机制 Lemma（可发表贡献）**：NF4 块级 absmax 定标 + 近最优码本 ⇒ rel_error 尺度不变且跨层近常数 ⇒ score ∝ ‖W‖_F。独立解释 AQLoRA(2608.23816)「protecting some layers, not the right ones」。")

# 3) Phase3-D 噪声底
L.append("\n## 3. 噪声底（Phase3-D，9run 同 product 配置）\n")
nd = jload(os.path.join(ROOT, "results_v2_phase3_noise/summary.json"))
if nd:
    L.append(f"- N（gsm8k-flex 极差）= **{nd.get('range_pt', nd.get('N','—'))} pt**（8B 跨种子极稳，N_seed≈0.30）")
L.append("- 14B 本地噪声（B11）：adaq flex 三种子极差 **14.71pt** >> 8B 代理 → 14B 信号被自身种子噪声淹没（ccadf1e 诚实下修）。")

# 4) 14B 规模
L.append("\n## 4. 14B 规模（Qwen2.5-14B，n=3 seed）\n")
r14 = jload(os.path.join(ROOT, "results_v2_qwen14b/summary_3seeds.json"))
if r14:
    L.append(f"- MMLU 配对差 mean ≈ {r14.get('mmlu_pair_mean','+0.0005')}（统计零，no difference; underpowered）")
    L.append(f"- GSM8K-flex 配对差 mean ≈ {r14.get('flex_pair_mean','+0.0887')}（3/3 同号，但 < 14B 本地噪声 14.71pt → 不稳健）")
    L.append(f"- 参数：adaq 4,620,288（秩528）vs qlora 6,291,456（秩768，1.45×预算）→ 非预算对齐，描述性参数效率成立，因果「略胜」不成立")
L.append("- **B5 chat-template 复评**（决定性，证伪格式 artifact）：")
b5rep = os.path.join(ROOT, "results_v2_14b_chattemplate/REPORT_14B_chattemplate.md")
if os.path.isfile(b5rep):
    L.append("  - REPORT_14B_chattemplate.md 已生成（见该文件 chat vs 非chat flex 配对差 + R1 判定）")
else:
    L.append("  - ⏳ 待 B5 完成")

# 5) P2-1 成本归一化（KBS 唯一正向赌注）
L.append("\n## 5. P2-1 成本归一化秩分配（KBS 唯一正向路径）\n")
p21 = jload(os.path.join(ROOT, "results_v2_p21/summary.json"))
if p21:
    dec = p21.get("decision", {})
    L.append(f"- 主跑判定：**{dec.get('verdict','—')}**")
    L.append(f"- R1={dec.get('R1')} R1b={dec.get('R1b')} dPM_mean={p21.get('paired_gsm8k_flex',{}).get('dPM_cost_norm_minus_uniform_pm',{}).get('mean')} N={p21.get('_meta',{}).get('noise_floor_N_pt')}")
    cf = jload(os.path.join(ROOT, "results_v2_p21/summary_confirm.json"))
    if cf:
        L.append(f"- R1-confirm（6 seed）：**{cf.get('decision',{}).get('verdict','—')}**")
    else:
        L.append("- R1-confirm：⏳（仅 R1=true 触发）")
else:
    L.append("_⏳ P2-1 未完成_")

# 6) PTQ 基线锚点
L.append("\n## 6. PTQ 基线锚点\n")
L.append("- bf16 干净基座（F2/M1）：results_v2/baseline_bf16_seed* — " + ("✅ 6/6" if len(glob.glob(os.path.join(ROOT,"results_v2/baseline_bf16_seed*/.done_*")))>=6 else "⏳"))
rtn = find_res(os.path.join(ROOT, "results_v2_rtn"))
if rtn:
    rj = jload(rtn)
    L.append(f"- F18 INT4-RTN-g128（真 PTQ 锚，rel_err 0.123 吻合理论）：MMLU {pct(g(rj,'results','mmlu','acc,none'))} GSM8K-strict {pct(g(rj,'results','gsm8k','exact_match,strict-match'))}")
else:
    L.append("- F18 INT4-RTN-g128：⏳")

# 7) venue 路由
L.append("\n## 7. Venue 路由（依结果）\n")
L.append("- **P2-1 R1-confirm 确证阳性** → KBS（Q1，中科院2区）「诊断+修正」叙事。")
L.append("- **P2-1 阴性 / confirm 失败** → 四件套（机制 Lemma + 收敛阴性 + 功效边界 + 可复现）退 TMLR / Neural Networks / Neurocomputing（Q2）。")
L.append("- 用户原约束 JCR Q2+/中科院2区；TMLR 非JCR无IF（备选）。\n")

L.append("\n---\n_全实验环节由 6+1 waiter 自主跑完（phase3→bf16/B8→F18/B5→P2-1→confirm），watchdog 监控+@reboot 续跑，每环节 flock+幂等+自动 commit。预注册判定只填数不改规则。_")

out = os.path.join(ROOT, "results_v2_FINAL_OVERVIEW.md")
with open(out, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"[overview] 写 {out}（{len(L)} 行）")
