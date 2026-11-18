# AdaQ-LoRA 全实验环节最终总览

生成：2026-11-18 23:15:42 UTC  
连接：2×RTX 4090D；基座 Llama-3-8B（主）/ Qwen2.5-14B-Instruct（规模）

> 本文件由 watchdog 在全链完成时自动生成，汇总所有实验环节的关键数字与预注册判定。诚实呈现阴性/降级。

## 0. 流水线完成状态

| 环节 | DONE 产物 | 状态 |
|---|---|---|
| phase3 D 噪声底 | logs/phase3/.done_D | ✅ |
| phase3 B adalora | logs/phase3/.done_B | ✅ |
| phase3 E 基线 | logs/phase3/.done_E | ✅ |
| phase3 C R曲线 | logs/phase3/.done_C | ✅ |
| phase3 A 主表重训 | logs/phase3/.done_A | ✅ |
| bf16 锚点 | 产物计数 | ✅ |
| B8 flex审计采样 | 产物计数 | ✅ |
| B5 chat-template | 产物计数 | ✅ |
| F18 INT4-RTN | 产物计数 | ✅ |
| P2-1 成本归一化 | logs/p21/.done_p21 | ✅ |
| P2-1 confirm | logs/p21/.done_p21confirm | ⏳/✗ |

## 1. 主表 — 等秩 512 BF16（n=3 seed）

| 方法 | MMLU | GSM8K-strict | GSM8K-flex | 参数 |
|---|---|---|---|---|
| qlora | — | — | — | — |
| dora | — | — | — | — |
| adalora | — | — | — | — |
| adaq | — | — | — | — |

_注：phase3-A 重训（α=2r, 375步 no-packing）替换原 21步 packing 主表；stats 评审 F1 主表无功效已修。_

## 2. NF4 评测路径 — 核心阴性贡献

| 方法(NF4加载) | MMLU | GSM8K-strict |
|---|---|---|
| baseline_nf4 | 0.636376584532118 | — |
| qlora_nf4 | 0.6333855576128756 | — |
| dora_nf4 | 0.6343113516593077 | — |
| adalora_nf4 | 0.6356644352656317 | — |
| adaq_nf4 | 0.6335279874661729 | — |

**根因（已落档 NF4_root_cause_analysis.md）**：score=rel_error×importance，rel_error CV 1.47%（近常数）、importance(‖W‖) CV 56.2%、Spearman(score,‖W‖)=0.998 → score 99.8% 范数驱动。AdaQ 实为**范数感知**非量化误差感知。
**机制 Lemma（可发表贡献）**：NF4 块级 absmax 定标 + 近最优码本 ⇒ rel_error 尺度不变且跨层近常数 ⇒ score ∝ ‖W‖_F。独立解释 AQLoRA(2608.23816)「protecting some layers, not the right ones」。

## 3. 噪声底（Phase3-D，9run 同 product 配置）

- N（gsm8k-flex 极差）= **— pt**（8B 跨种子极稳，N_seed≈0.30）
- 14B 本地噪声（B11）：adaq flex 三种子极差 **14.71pt** >> 8B 代理 → 14B 信号被自身种子噪声淹没（ccadf1e 诚实下修）。

## 4. 14B 规模（Qwen2.5-14B，n=3 seed）

- MMLU 配对差 mean ≈ +0.0005（统计零，no difference; underpowered）
- GSM8K-flex 配对差 mean ≈ +0.0887（3/3 同号，但 < 14B 本地噪声 14.71pt → 不稳健）
- 参数：adaq 4,620,288（秩528）vs qlora 6,291,456（秩768，1.45×预算）→ 非预算对齐，描述性参数效率成立，因果「略胜」不成立
- **B5 chat-template 复评**（决定性，证伪格式 artifact）：
  - REPORT_14B_chattemplate.md 已生成（见该文件 chat vs 非chat flex 配对差 + R1 判定）

## 5. P2-1 成本归一化秩分配（KBS 唯一正向路径）

- 主跑判定：**R2（阴性）：cost_norm 未稳定超过 uniform_pm（参数匹配）→ 维持「机制+阴性+功效边界」四件套，退 TMLR/Neural Networks/Neurocomputing**
- R1=False R1b=False dPM_mean=0.8087 N=4.094
- R1-confirm：⏳（仅 R1=true 触发）

## 6. PTQ 基线锚点

- bf16 干净基座（F2/M1）：results_v2/baseline_bf16_seed* — ✅ 6/6
- F18 INT4-RTN-g128（真 PTQ 锚，rel_err 0.123 吻合理论）：MMLU 61.138 GSM8K-strict None

## 7. Venue 路由（依结果）

- **P2-1 R1-confirm 确证阳性** → KBS（Q1，中科院2区）「诊断+修正」叙事。
- **P2-1 阴性 / confirm 失败** → 四件套（机制 Lemma + 收敛阴性 + 功效边界 + 可复现）退 TMLR / Neural Networks / Neurocomputing（Q2）。
- 用户原约束 JCR Q2+/中科院2区；TMLR 非JCR无IF（备选）。


---
_全实验环节由 6+1 waiter 自主跑完（phase3→bf16/B8→F18/B5→P2-1→confirm），watchdog 监控+@reboot 续跑，每环节 flock+幂等+自动 commit。预注册判定只填数不改规则。_
