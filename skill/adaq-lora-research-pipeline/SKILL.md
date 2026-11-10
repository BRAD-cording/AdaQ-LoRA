---
name: adaq-lora-research-pipeline
description: "基于 2×NVIDIA RTX 4090（24GB×2，无 NVLink，原生 BF16）的 AdaQ-LoRA（NF4 量化感知自适应秩分配）LLM 论文全周期执行智能体，目标产出中科院二区+/顶会论文。核心方法：预计算逐层量化误差，以误差驱动 LoRA 秩预算分配，零额外显存开销。四大支柱：(1) 理论审查——审查量化 PEFT 数学推导（NF4 误差模型、Eckart-Young 最优秩分配、谱抬升定理、拉格朗日边际条件），只区分 Theorem/Proposition/Allocation Rule 三级表述；(2) SOTA 对比——QLoRA/DoRA/AdaLoRA 的 Gap 分析、Σr_l=512 参数预算对齐（口径 A：仅 q_proj+v_proj 共 64 位点）、lm-evaluation-harness 评测（MMLU 5-shot/GSM8K 4-shot/TruthfulQA 0-shot/IFEval 0-shot）、≥3 种子 + Wilcoxon 符号秩检验 + 逐任务 bootstrap；(3) 双卡训练优化——实验级并行优先、DDP 可选、NF4+double_quant→FA2→packing→grad_ckpt→Liger→paged_adamw_8bit 顺序、单卡 23GB 显存红线、统一 bf16；(4) 论文产出——Gap 叙事组织、Limitations ≥5 条逐条配证据。当用户提出「AdaQ-LoRA」「双卡 4090 LLM 实验」「量化感知秩分配」「24GB 微调论文」「QLoRA/DoRA/AdaLoRA 对比实验」「审查我的秩分配理论/证明」「低秩预算分配」「lm-eval 评测设计」「双卡调度/DDP」「Q2/一区/顶会论文实验规划」等请求时使用。"
---

# AdaQ-LoRA 研究管线（2×RTX 4090 双卡 → Q2+/顶会论文）

围绕一份核心资产：《A10 单卡 LLM 最小化实验论文落地执行手册 v3.0》（AdaQ-LoRA，硬件层已迁移至双卡 4090）。一切分析、代码与论文建议以该手册的算法框架为基座。

## 三条铁律（任何情况下不可违反）

1. **理论先于实验**：自研算法的数学表述必须先通过理论审查（见支柱一）；严禁混用启发式与严格证明，只区分 **Theorem / Proposition / Allocation Rule** 三级表述，写作时按真实层级标注，不得升格。
2. **公平先于领先**：对比 SOTA 必须参数预算对齐（**口径 A：所有方法 Σr_l=512，仅 q_proj+v_proj 共 64 位点**）；所有对比方法必须使用完全相同的硬件与并行策略，禁止不对称对比。
3. **可复现先于速度**：训练优化不得破坏数值语义；多卡并行不是放宽显存与数值校验的理由。每项优化记录显存/耗时前后对比。

## 双卡调度策略（核心增量）

- **默认：实验级并行**。GPU0/GPU1 各跑一个独立实验（如两种子、两方法），保持与单卡完全一致的数值语义，结果天然可比；实验矩阵 ~320h 的墙钟时间约减半。
- **可选：DDP 双卡训练**。仅当单次运行过慢时使用；生效 batch = 单卡 batch×2，必须在评测协议中显式记录，且所有方法一致；每 rank 独立设种子；启用前先通过 `numerical_invariance_check.py` 双卡模式验证。
- **显存红线按单卡 23GB 执行**（batch=4, seq=2048, grad_ckpt+packing），不得跨卡累加预算；配置变更后用 `memory_budget.py` 重新扫描。

## 工作流总览

按用户请求所处阶段进入对应支柱；完整论文任务按 一 → 二 → 三 → 四 顺序推进。

### 支柱一：理论正确性审查（最高优先级）

读取 [references/theory-review.md](references/theory-review.md)，执行其中《理论审查清单》，产出 `docs/theory-review-report.md`：

- 逐条核验定理 1（谱抬升）、定理 2（边际条件）、定理 3（近似最优性）的前提、推导链与结论方向。
- 已知高风险点（必须优先检查）：Weyl 不等式与随机矩阵启发式的混用、从边际条件到比例分配的跳跃、O(L/R) 界的证明缺失、NF4 零均值假设的适用边界。详见 theory-review.md §3。
- 输出《理论审查报告》：每条结论标注 ✅ 严格成立 / ⚠️ 需附加条件 / ❌ 有错误及修复方案。

### 支柱二：SOTA 对比与评测设计

读取 [references/sota-comparison.md](references/sota-comparison.md)，执行其中《公平对比协议》，产出 `sota-comparison-protocol.md`：

- Gap 论证：对照手册 §2.2 的 Gap-1~4 表，用 scholar 插件核查最新文献，确认 Gap 未被新工作填补（每次正式写作前重查一次）。
- 基线对齐：口径 A（Σr_l=512，仅 q_proj+v_proj 共 64 位点）；报告对齐计算过程。
- 评测：MMLU(5-shot) / GSM8K(4-shot) / TruthfulQA(0-shot) / IFEval(0-shot)，统一用 lm-evaluation-harness；≥3 个种子 + Wilcoxon 符号秩检验 + 逐任务 bootstrap，报告均值±标准差。种子按双卡并行排队。
- 用 [scripts/run_benchmarks.sh](scripts/run_benchmarks.sh) 生成标准评测命令。

### 支柱三：双卡训练效率优化

读取 [references/training-optimization.md](references/training-optimization.md)。单卡内按性价比顺序应用：**NF4+double_quant → FA2 → packing → grad_ckpt → Liger → paged_adamw_8bit**。4090 原生 BF16，统一用 bf16 而非 fp16。每步验证：loss 曲线一致 + 显存/耗时改善。目标：单卡峰值显存 ≤ 23GB。

### 支柱四：论文产出（Q2+/顶会）

- 叙事主线（手册 §6.1）：围绕「量化与低秩适配的耦合被忽视」组织 Gap 叙事——Introduction 第一段必须落在 Gap-1/Gap-2 上。维护 `paper-framework.md` 与 `paper-outline.md`。
- 实验矩阵：手册 §四 的六个实验为完整版；时间不足时的最小集 = 实验一（方法对比）+ 实验二消融 1 + 实验三（秩预算）。
- Limitations ≥5 条且逐条配证据（实验证据或理论说明）——这是 Q2+ 审稿的硬性得分点。
- 期刊/会议定位、逐节写作：配合内置 research-writer 技能；文献核查用 scholar 插件。

## 工作规范

- 动手前先查 `configs/experiment_matrix.json` 与训练超参；无 GPU 时仅运行自检脚本（`--self-test`、合成秩、显存扫描）。
- 任何代码/超参改动先自检是否触碰铁律；实验结论必须附带 **GPU 编号、并行策略、数据路径与复现命令**。
- 产出统一写入 `docs/` 与 `results/`，确保报告可追溯到脚本与配置。

## 脚本

- [scripts/adaq_precompute.py](scripts/adaq_precompute.py)：量化误差计算 + 秩分配（对应手册 §5.2，支持模块类型分组方案 A 与逐层方案 B 的输出）。用法：`python3 adaq_precompute.py --model <hf_path> --budget 512 --r-min 4 --r-max 128 --out ranks.json`。无 GPU 环境下可用 `--self-test` 做合成数据自测。
- [scripts/run_benchmarks.sh](scripts/run_benchmarks.sh)：生成 lm-eval 标准评测命令（含 merge 后模型与 adapter 直接加载两种方式、输出目录规范）。

## 与其他技能的关系

- 选题调研、期刊匹配：用户技能 llm-compression-research（本技能是其 AdaQ-LoRA 特化执行版，冲突时以本技能的实验/理论细节为准）。
- 论文写作与润色：内置 research-writer；投稿前模拟审稿：内置 paper-review-coach。
