---
name: adaq-lora-research-pipeline
description: "基于 NVIDIA A10 24GB 单卡训练服务器的 AdaQ-LoRA（量化感知自适应秩分配）LLM 论文全周期执行管线，目标产出 Q2 及以上（中科院二区+/顶会）论文。三大支柱：(1) 理论正确性保障——审查量化 PEFT 数学推导（NF4 误差模型、Eckart-Young 最优秩分配、谱抬升定理、拉格朗日边际条件），逐条验证定理前提、不等式方向与启发式假设的表述边界，杜绝理论错误；(2) SOTA 对比与领先评测——围绕 QLoRA/DoRA/AdaLoRA 的 Gap 分析、参数公平对齐的基线设置、lm-evaluation-harness 权威评测（MMLU/GSM8K/TruthfulQA/IFEval）、多种子显著性检验；(3) A10 单卡训练效率优化——4-bit 量化训练显存预算、梯度检查点、序列打包、FlashAttention、Liger kernel、评测流水线，节约训练时间。当用户提出「AdaQ-LoRA」「A10 单卡 LLM 实验」「量化感知秩分配」「单卡 24GB 微调论文」「QLoRA/DoRA/AdaLoRA 对比实验」「审查我的秩分配理论/证明」「低秩预算分配」「lm-eval 评测设计」「Q2/一区/顶会论文实验规划」等请求时使用。"
---

# AdaQ-LoRA 研究管线（A10 单卡 → Q2+ 论文）

围绕一份核心资产：《A10 单卡 LLM 最小化实验论文落地执行手册 v3.0》（AdaQ-LoRA）。一切分析、代码与论文建议以该手册的算法框架为基座。

## 铁律（三条，不可违反）

1. **理论先于实验**：任何实验/写作输出前，自研算法的数学表述必须通过理论审查（见支柱一）。发现理论漏洞时，先修复或诚实降级表述（如"定理"→"命题/启发式"），再进入实验。
2. **公平先于领先**：对比 SOTA 必须参数预算对齐（同总秩或同显存），禁止 r=64 基线 vs R=512 自研的不对称对比。评测必须用标准工具 + 多种子。
3. **可复现先于速度**：训练优化不得破坏数值语义（如随意替换 kernel 导致结果漂移）；每项优化记录显存/耗时前后对比。

## 工作流总览

按用户请求所处阶段进入对应支柱；完整论文任务按 一 → 二 → 三 → 四 顺序推进。

### 支柱一：理论正确性审查（最高优先级）

读取 [references/theory-review.md](references/theory-review.md)，执行其中《理论审查清单》。核心动作：

- 逐条核验手册中的定理 1（谱抬升）、定理 2（边际条件）、定理 3（近似最优性）的前提、推导链与结论方向。
- 区分三个表述层级：**严格定理**（有完整证明）、**命题**（有证明概要但含假设）、**启发式**（如秩分配公式 r_l ∝ ‖E_l‖_F·I_l）。写作时必须按真实层级标注，不得升格。
- 输出《理论审查报告》：每条结论标注 ✅ 严格成立 / ⚠️ 需附加条件 / ❌ 有错误及修复方案。

已知高风险点（必须优先检查）：Weyl 不等式与随机矩阵启发式的混用、从边际条件到比例分配的跳跃、O(L/R) 界的证明缺失、NF4 零均值假设的适用边界。详见 theory-review.md §3。

### 支柱二：SOTA 对比与评测设计

读取 [references/sota-comparison.md](references/sota-comparison.md)，执行其中《公平对比协议》：

- Gap 论证：对照手册 §2.2 的 Gap-1~4 表，用 scholar 插件核查最新文献，确认 Gap 未被新工作填补（每次正式写作前重查一次）。
- 基线对齐：所有方法统一总参数量 Θ 或统一峰值显存；报告对齐计算过程。
- 评测：MMLU(5-shot) / GSM8K(4-shot) / TruthfulQA / IFEval，统一用 lm-evaluation-harness；≥3 个种子，报告均值±标准差，做显著性检验。
- 用 [scripts/run_benchmarks.sh](scripts/run_benchmarks.sh) 生成标准评测命令。

### 支柱三：A10 训练效率优化

读取 [references/training-optimization.md](references/training-optimization.md)。优化顺序（按性价比排序）：4-bit 双重量化 → bf16 计算 → FlashAttention-2 → 序列打包（packing）→ 梯度检查点 → Liger kernel → 梯度累积调优 → torch.compile。每步验证：loss 曲线一致 + 显存/耗时改善。目标：Llama-3-8B QLoRA 级训练峰值显存 ≤ 23GB。

### 支柱四：论文产出（Q2+）

- 叙事主线（手册 §6.1）：「量化与低秩适配的耦合被忽视」——Introduction 的第一段必须落在 Gap-1/Gap-2 上。
- 实验矩阵：手册 §四 的六个实验为完整版；时间不足时的最小集 = 实验一（方法对比）+ 实验二消融 1 + 实验三（秩预算）。
- Limitations 必须包含手册 §七 的五条，且每条配实验证据或理论说明——这是 Q2+ 审稿的硬性得分点。
- 期刊/会议定位、逐节写作：配合内置 research-writer 技能；文献核查用 scholar 插件。

## 脚本

- [scripts/adaq_precompute.py](scripts/adaq_precompute.py)：量化误差计算 + 秩分配（对应手册 §5.2，已修复释放顺序与参数对齐问题，支持模块类型分组方案 A 与逐层方案 B 的输出）。用法：`python3 adaq_precompute.py --model <hf_path> --budget 512 --r-min 4 --r-max 128 --out ranks.json`。无 GPU 环境下可用 `--self-test` 做合成数据自测。
- [scripts/run_benchmarks.sh](scripts/run_benchmarks.sh)：生成 lm-eval 标准评测命令（含 merge 后模型与 adapter 直接加载两种方式、输出目录规范）。

## 与其他技能的关系

- 选题调研、期刊匹配：用户技能 llm-compression-research（本技能是其 AdaQ-LoRA 特化执行版，冲突时以本技能的实验/理论细节为准）。
- 论文写作与润色：内置 research-writer；投稿前模拟审稿：内置 paper-review-coach。
