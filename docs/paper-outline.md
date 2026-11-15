# AdaQ-LoRA 论文逐节大纲

> 配合 paper-framework.md 使用；每节标注内容要点、理论层级和图表要求。
> 配合内置 research-writer 技能进行实际写作；文献核查用 scholar 插件。

---

## Title

**Norm-Aware Rank Budgeting for LoRA Fine-Tuning on NF4-Quantized LLMs: Mechanism, Negative Result, and Detectable-Effect Boundaries**
> [2026-11-16 评审裁定改题] 原题 *AdaQ-LoRA: Quantization-Aware Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning of LLMs*——「Quantization-Aware」名不副实（score 99.8% 范数驱动）、「Parameter-Efficient」等秩下反号（+11.5%）。

（备选：Error-Driven Rank Budgeting for 4-bit Quantized LoRA Fine-Tuning）

## Abstract (~200 words)

- 背景：4-bit 量化 + LoRA 是单卡微调大模型的标配
- 问题：均匀秩分配忽视量化误差的层间异质性
- 方法：预计算 NF4 逐层误差 → 误差驱动秩分配
- 理论：严格扰动界 + 谱抬升命题 + 代理质量诊断
- 实验：[填充实验后] 4 benchmark × 3 seed × 4 方法，公平对齐
- 结论 [2026-11-16 评审裁定撤回]：等秩下无参数效率优势（adaq +11.5% 参数、adalora 臂 2× 预算未剪枝）。结论重构=阴性结果 + 结构性机制解释 + 可检出效应边界。

## §1 Introduction (~1 page)

### 结构（第一段必须落在 Gap-1/Gap-2 上）

- **¶1**: LLM 微调的量化 + 低秩范式（QLoRA 开创）。4-bit 基座 + LoRA 适配器 = 单卡可训练。但现有方法对所有层均匀分配秩——**Gap-1: 量化误差的层间异质性被忽视**。
- **¶2**: 量化误差补偿的已有工作仅做初始化补偿（LoftQ）或梯度幅度缩放（QA-LoRA），未做秩容量分配——**Gap-2**。
- **¶3**: AdaLoRA 的自适应秩在 4-bit 基座上重要性分数不可靠——**Gap-3**。NF4 对不同模块类型的量化误差可差数倍——**Gap-4**。
- **¶4** [重构]：原「核心洞察」（量化误差可作秩分配信号）被证伪——ε_l 近常数是 NF4 块定标的结构性后果，score 实为范数驱动。重构为：Lemma ‖E_l‖_F=ε_l‖W_l‖_F 且 ε≈const ⇒ 误差分配可证明退化为范数分配；给出该机制 + 收敛阴性 + 功效边界（噪声底为区间 [0.9,11.5] 点，非单点 4.5）。
- **¶5**: 贡献总结（3 点：方法、理论、实验）。
- **图 1**: 方法概览图（预计算 ε_l → 分配 r_l → QLoRA 训练）。

## §2 Related Work (~1.5 pages)

### §2.1 量化微调
- QLoRA: NF4 + double quant + paged optimizer → 单卡 4-bit 训练
- LoftQ: 初始化补偿（用 SVD 近似量化误差做初始化）→ 区分：补偿初始化 vs 分配容量
- QA-LoRA: 梯度幅度缩放 → 区分：缩放 vs 分配
- LQ-LoRA, IR-QLoRA: 其他量化 PEFT 变体
- **边界声明**：本文框架适用于量化冻结的 QLoRA 范式；当量化参数联合训练时，E_l 变为可学习，"已知偏移"假设不成立（引 future work）

### §2.2 自适应秩 PEFT
- AdaLoRA: 训练中动态调整秩，基于重要性分数
- Gap-3: 4-bit 基座上重要性分数不可靠
- DyLoRA, FLoRA: 其他动态秩方法

### §2.3 低秩逼近与量化理论
- Eckart-Young-Mirsky 定理
- Weyl 奇异值扰动界
- 与本文理论的关系

## §3 Method (~3 pages)

### §3.1 预备知识
- NF4 分块量化（块大小 64，absmax 归一化）
- LoRA: W + ΔW = W + BA，B∈ℝ^{d×r}, A∈ℝ^{r×k}
- 4-bit QLoRA: W_q = Q₄(W)，训练在 bf16 计算精度上进行

### §3.2 量化误差模型
- **定义**：E_l = Q₄(W_l) − W_l = Ŵ_l − W_l
- **假设 H0**: E_l 近似零均值（附实测直方图）— A1 ✅
- **经验值**: ε_l = ‖E_l‖_F/‖W_l‖_F，实测分布（附表 min/median/max）— A2 ⚠️需实测
- **假设 H_block**: 块内同分布、块间独立 — A3 ⚠️

### §3.3 理论分析

**定理 1（双向扰动界，严格）** ✅ Theorem:
> |σ_i(ΔW*−E) − σ_i(ΔW*)| ≤ σ_1(E) ≤ ‖E‖_F
> (Weyl 奇异值扰动定理，无额外假设)

**命题 1'（平均谱抬升）** ⚠️ Proposition:
> 假设 H_isotropic: E 近似各向同性随机扰动
> 则 𝔼‖ΔW*−E‖_F² = ‖ΔW*‖_F² + ‖E‖_F²
> erank(M) 随 ‖E‖_F 增大而增大
> (配数值验证图 — 已生成 ✅)

**命题 2（边际条件）** ⚠️ Proposition:
> 假设 H1: 各层谱幂律衰减且指数相近
> 连续松弛下 KKT 条件: σ²_{l,r_l+1} = λ
> 推导链: H1 → 边际条件 → r*_l ∝ c_l^{2/α} → 代理 ‖E_l‖_F·I_l

**分配规则（Allocation Rule）** ⚠️ Heuristic:
> r_l ∝ ‖E_l‖_F · I_l (I_l = ‖W_l‖_F)
> 定位: 在 H1 下对最优解的可计算代理

**命题 3a（取整损失）** ⚠️ Proposition:
> 连续解取整损失 ≤ O(L·σ̄²/R)，R ≫ L 时可忽略

**命题 3b（代理质量）** ⚠️ Empirical:
> Spearman ρ(score, σ²_{r+1}) ≥ 0.7 (实测: ρ=0.9836 ✅)

### §3.4 算法
- 预计算流程（adaq_precompute.py）
  1. FP16 模型 → CPU 逐层取出
  2. 4-bit 模型 → GPU 逐层反量化（dequantize_4bit）
  3. 计算 ‖E_l‖_F, ‖W_l‖_F → score
  4. 迭代水位法分配（截断后归一化，总秩守恒）
  5. 方案 A 分组（PEFT rank_pattern）
- 复杂度: O(L·d·k)，<5 min

### §3.5 与 QLoRA 的差异
- 唯一差异: 秩分配方式（非均匀 vs 均匀）
- 训练流程、量化配置、优化器完全对齐

## §4 Experiments (~3 pages)

### §4.1 设置
- 模型: Llama-3-8B (主), Mistral-7B, Qwen2-7B (泛化)
- 数据: OpenHermes-2.5 (2k 子集)
- 训练: 3 epoch, lr=2e-4, batch=16, seq=2048, NF4+double quant
- 评测: lm-evaluation-harness, MMLU(5-shot)/GSM8K(4-shot)/TruthfulQA(0-shot)/IFEval(0-shot)
- 种子: 42/3407/2025
- 统一超参表（见 configs/training_params.yaml）

### §4.2 主实验（实验一）
- 表 1: 方法 × 4 benchmark 主表
- 加粗最优, 下划线次优
- 脚注: 秩预算 R=512, 种子 N=3, 口径 A, lm-eval 版本

### §4.3 消融（实验二）
- 表 2: 5 种分配策略 × 2 benchmark
- 对应: error_only / importance_only / product / uniform / equal_weight
- 说明: 消融为代理有效性的直接证据

### §4.4 秩预算分析（实验三）
- 图 2: R-性能曲线 (R ∈ {128, 256, 512, 768, 1024})
- 分析: AdaQ 在各预算下 vs QLoRA 的趋势 [未执行——R 曲线 R∈{128,256,512,768,1024} 0 产物，投稿前 P0 补做项，见 review/fair-baseline.md F17-M4]

### §4.5 效率分析
- 表 3: 峰值显存 / 每 epoch 时长 / 预计算耗时 / 推理吞吐
- 图 3: 显存-性能散点

### §4.6 跨模型泛化（实验五，可选）
- 表 4: Llama-3-8B / Mistral-7B / Qwen2-7B

### §4.7 统计报告（当前为描述性，非推断性）
- 主表只报 `mean ± std (n=3)` + 每格 `per_seed` 明细（来自 summary.json，不手抄）
- MMLU 逐任务 bootstrap 可作为**探索性**分析，需注明 57 任务多重比较未校正
- **不写** Wilcoxon：3 种子下最小双侧 p=0.25，达不到 p<0.05，
  详见 `docs/sota-comparison-protocol.md` §显著性检验 的功效硬约束
- 若审稿要求显著性：扩到 ≥6 种子重跑，或改报效应量 + 置信区间并明确标注探索性

## §5 Analysis (~1 page)

- **图 4**: 逐层 ε_l 分布 + 对应 r_l 分配条形图（核心证据图）
- 讨论: 高 ε 层获得更高秩 → 验证方法直觉
- 不同模型的 ε_l 分布对比 → 跨模型一致性

## §6 Limitations (~0.5 page)

1. L1: 分配代理的任务无关性（方案 A 使用 ‖W_l‖_F）
2. L2: 幂律衰减假设 H1（实际谱衰减可能不同）
3. L3: NF4 量化冻结假设（联合训练量化参数时框架失效）
4. L4: A10 单卡约束（未验证 70B+ 模型）
5. L5: 分配截断影响（r_min/r_max 截断）

> 每条配实验证据或理论说明 — Q2+ 审稿硬性得分点

## §7 Conclusion & Future Work (~0.3 page)

- 总结: 量化误差驱动的秩分配，零额外开销
- Future work: 量化参数联合训练、任务感知重要性度量、更大模型验证

---

## 图表清单

| 编号 | 内容 | 来源 |
|:---|:---|:---|
| 图1 | 方法概览 | 新制 |
| 图2 | R-性能曲线 | 实验三 |
| 图3 | 显存-性能散点 | 效率分析 |
| 图4 | 逐层 ε_l + r_l 分布 | adaq_precompute 输出 |
| 图5 | erank vs ‖E‖_F | 命题1' 验证（已生成 ✅） |
| 表1 | 主对比表 | 实验一 |
| 表2 | 消融表 | 实验二 |
| 表3 | 效率表 | 训练 profile |
| 表4 | 跨模型表 | 实验五（可选） |
