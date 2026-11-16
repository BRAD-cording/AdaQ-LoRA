# AdaQ-LoRA: 范数感知自适应秩分配 LLM 微调研究项目

围绕 2×NVIDIA RTX 4090D 24GB 训练服务器（双卡），目标产出 Q2+（中科院二区+/顶会）论文。

**核心方法**：AdaQ-LoRA（范数感知自适应秩分配）—— 预计算 NF4 逐层量化误差与权重范数，以 score=rel_error×‖W‖_F 分配 LoRA 秩预算。实测该 score 99.8% 由权重范数驱动（rel_error 跨层近常数，系 NF4 per-block absmax 定标的结构性后果），故方法本质为范数/重要性感知而非量化误差感知；主要结论为阴性：等秩预算下未观察到一致优于 QLoRA 的精度增益（详见 docs/claim-revision-norm-aware.md）。

## 三条铁律（不可违反）

1. **理论先于实验**：自研算法的数学表述必须通过理论审查
2. **公平先于领先**：对比 SOTA 必须参数预算对齐，禁止不对称对比
3. **可复现先于速度**：训练优化不得破坏数值语义

## 项目结构

```
AdaQ-LoRA/
├── adaq-lora-research-pipeline/    # 研究管线（技能包）
│   ├── SKILL.md                      # 管线定义与四支柱工作流
│   ├── references/                   # 三大支柱参考文档
│   │   ├── theory-review.md          #   理论审查清单与高风险点
│   │   ├── sota-comparison.md        #   SOTA 对比与公平评测协议
│   │   └── training-optimization.md  #   RTX 4090D 训练效率优化
│   └── scripts/                      # 管线核心脚本
│       ├── adaq_precompute.py        #   预计算：量化误差 + 秩分配
│       └── run_benchmarks.sh         #   标准评测命令生成
│
├── docs/                            # 项目文档
│   ├── theory-review-report.md       #   理论审查报告（Pillar 1 产出）
│   ├── sota-comparison-protocol.md   #   SOTA 对比协议（Pillar 2 产出）
│   ├── training-optimization-plan.md #   训练优化计划（Pillar 3 产出）
│   ├── paper-framework.md            #   论文框架（Pillar 4 产出）
│   ├── paper-outline.md              #   论文逐节大纲
│   └── figures/                      #   图表数据
│       └── erank_vs_error.json        #     命题1' 数值验证数据
│
├── scripts/                         # 项目脚本
│   ├── train.py                      #   统一训练（4 方法）
│   ├── merge_adapter.py             #   合并 adapter 到 bf16
│   ├── run_eval_batch.sh             #   批量评测
│   ├── aggregate_results.py         #   结果汇总 + Wilcoxon 检验
│   ├── generate_ablation_configs.py #   消融配置生成
│   ├── generate_synthetic_ranks.py  #   合成秩（无 GPU 测试）
│   ├── verify_proposition1_spectral_elevation.py  # 命题1' 验证
│   ├── memory_budget.py             #   显存预算计算器
│   ├── numerical_invariance_check.py #  数值语义不变性检查
│   └── profile_training.py          #   训练 profiling
│
├── configs/                         # 配置
│   ├── training_params.yaml          #   统一训练超参
│   ├── experiment_matrix.json        #   实验矩阵（6 实验）
│   └── ablation_ranks/              #   消融分配配置
│       ├── ranks_error_only.json
│       ├── ranks_importance_only.json
│       ├── ranks_product.json
│       ├── ranks_uniform.json
│       └── ranks_equal_weight.json
│
├── data/                            # 评测数据
│   ├── GSM8K/                        #   GSM8K 数据集
│   └── MMLU/                         #   MMLU 数据集（57 科目）
│
├── outputs/                         # 训练输出
│   ├── adapters/                     #   LoRA adapter
│   └── merged/                       #   合并后模型
│
├── results/                         # 评测结果
├── logs/                            # 训练日志
├── run_pipeline.py                  # 全流程执行入口
├── requirements.txt                 # 依赖
└── ranks.json                       # 预计算秩分配（合成版，需替换为真实）
```

## 快速开始

### 无 GPU 测试（验证流程完整性）

```bash
python run_pipeline.py --step test
```

验证内容：分配逻辑自测 → 命题1' 验证 → 显存预算 → 合成秩生成 → 消融配置

### 完整流程（需 GPU + 依赖）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 预计算秩分配（真实模型）
python run_pipeline.py --step precompute

# 3. 训练所有方法
python run_pipeline.py --step train

# 4. 合并 adapter
python run_pipeline.py --step merge

# 5. 批量评测
python run_pipeline.py --step eval

# 6. 汇总结果
python run_pipeline.py --step aggregate

# 或一步执行全流程
python run_pipeline.py --step all
```

## 四支柱工作流

### 支柱一：理论正确性审查（最高优先级）

**产出**：[docs/theory-review-report.md](docs/theory-review-report.md)

**关键结论**：
- 定理1（谱抬升）：❌ Weyl 与随机启发式混用 → 修复为 Theorem 1（严格扰动界）+ Proposition 1'（期望谱抬升）
- 定理3（近似最优界）：❌ O(L/R) 无证明 → 修复为 Proposition 3a（取整损失）+ 3b（代理质量，Spearman ρ=0.9836）
- 分配公式 r_l ∝ ‖E_l‖_F·I_l：⚠️ 降级为 Allocation Rule（启发式）

### 支柱二：SOTA 对比与评测设计

**产出**：[docs/sota-comparison-protocol.md](docs/sota-comparison-protocol.md)

**关键设计**：
- 口径 A（总秩预算对齐）：所有方法 Σ r_l = 512，仅 q_proj+v_proj = 64 位点 → r̄=8
- 评测：MMLU(5-shot)/GSM8K(4-shot)/TruthfulQA(0-shot)/IFEval(0-shot)
- 统计：≥3 种子，Wilcoxon 符号秩检验，逐任务 bootstrap

### 支柱三：RTX 4090D 训练效率优化

**产出**：[docs/training-optimization-plan.md](docs/training-optimization-plan.md)

**关键配置**：
- batch=4, seq=2048, grad_ckpt + packing → 峰值 ~21-22GB（23GB 预算内）
- 优化顺序：NF4+double_quant → FA2 → packing → grad_ckpt → Liger → paged_adamw_8bit
- 预计算 <5 min（逐层释放，不同时驻留两份模型）

### 支柱四：论文产出（Q2+）

**产出**：[docs/paper-framework.md](docs/paper-framework.md) + [docs/paper-outline.md](docs/paper-outline.md)

**叙事主线**：「量化与低秩适配的耦合被忽视」——Introduction 第一段落 Gap-1/Gap-2
**Limitations**：5 条，每条配实验证据/理论说明
**目标**：ICLR/NeurIPS（首选），KBS/ESWA（备选）

## 核心脚本

| 脚本 | 用途 | 无 GPU? |
|:---|:---|:---:|
| `adaq_precompute.py --self-test` | 分配逻辑自测 | ✅ |
| `verify_proposition1_spectral_elevation.py` | 命题1' 数值验证 | ✅ |
| `memory_budget.py --interactive` | 显存预算扫描 | ✅ |
| `generate_synthetic_ranks.py` | 合成秩（测试用） | ✅ |
| `generate_ablation_configs.py` | 消融配置生成 | ✅ |
| `adaq_precompute.py --model ...` | 真实预计算 | ❌ |
| `train.py` | 训练 | ❌ |
| `merge_adapter.py` | 合并 | ❌ |
| `run_eval_batch.sh` | 评测 | ❌ |
| `aggregate_results.py` | 汇总 | ✅ |

## 实验矩阵

| 实验 | 内容 | 优先级 | 机时 |
|:---|:---|:---:|:---:|
| 一 | 方法对比（4 方法×3 种子） | 必须 | ~90h |
| 二 | 消融（5 种分配策略） | 必须 | ~60h |
| 三 | 秩预算扫描（R=128~1024） | 必须 | ~50h |
| 四 | 目标模块扩展 | 可选 | ~20h |
| 五 | 跨模型泛化 | 可选 | ~80h |
| 六 | 8-bit 鲁棒性 | 可选 | ~20h |

总计 ~320h，分布在 10 周内执行。

## 引用

如使用本框架，请引用：

```bibtex
@software{adaq-lora,
  title={AdaQ-LoRA: Norm-Aware Adaptive Rank Budgeting for LoRA Fine-Tuning on NF4-Quantized LLMs},
  year={2026},
  note={Research pipeline for dual RTX 4090D LLM fine-tuning}
}
```
