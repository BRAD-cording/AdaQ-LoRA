# SOTA 对比与权威评测协议（AdaQ-LoRA）

> 依据：adaq-lora-research-pipeline/references/sota-comparison.md
> 状态：理论审查通过后的执行文档

---

## 1. Gap 论证维护

### 四个 Gap（手册 §2.2 立论根基）

| Gap | 描述 | 检索验证状态 |
|:---|:---|:---|
| Gap-1 | 量化与低秩适配的耦合被忽视——现有 PEFT 方法不根据量化误差分配秩 | 待 scholar 核查 |
| Gap-2 | 4-bit 量化误差补偿仅做初始化（LoftQ）或幅度缩放（QA-LoRA），未做秩分配 | 待 scholar 核查 |
| Gap-3 | AdaLoRA 的重要性分数在量化基座上不可靠，缺乏量化感知的重要性度量 | 待 scholar 核查 |
| Gap-4 | 层间异质性（NF4 对不同模块的误差差异）未被利用为分配信号 | 待 scholar 核查 |

### 检索协议

- **检索词**：`quantization-aware rank allocation LoRA`、`adaptive rank quantized LLM fine-tuning`、`low-rank budget allocation 4-bit`、`NF4 error compensation adapter`
- **重点核查（2024+）**：LoftQ、QA-LoRA、LQ-LoRA、IR-QLoRA、AQLM 系、QuaRot 系
- **每次正式写作/投稿前重查一次**，附检索日期
- **表述要求**：禁止"所有 PEFT 方法都未解决"这类过宽断言；写"在截至投稿的公开工作中，未见将逐层 4-bit 量化误差纳入秩预算分配的方法"，并附检索日期

### 重叠工作区分策略

若出现部分重叠工作，立即调整叙事：

| 重叠类型 | 叙事调整 |
|:---|:---|
| 补偿幅度/初始化（如 LoftQ） | "它们补偿初始化，我们分配容量" |
| 幅度缩放（如 QA-LoRA） | "它们缩放梯度幅度，我们分配秩预算" |
| 量化感知训练（如 QAT 系） | "我们冻结量化参数，它们联合训练" |

---

## 2. 公平对比协议（参数对齐铁律）

### 选定口径：A — 总秩预算对齐（推荐，叙事最干净）

**核心原则**：所有方法的总秩预算 R 相同，唯一差异是分配方式。

### 参数对齐计算（Llama-3-8B 为基准模型）

```
Llama-3-8B 架构：32 层 × 7 模块类型 = 224 个适配位点
模块类型：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

口径 A 方案：
  总预算 R = 512
  基线（QLoRA/DoRA）：均匀秩 r̄ = R / N_sites
  AdaQ-LoRA：非均匀分配 Σ r_l = 512

  问题：224 位点 → r̄ = 512/224 ≈ 2.3（太小，退化）

  解决：限制适配位点子集，仅 q_proj + v_proj = 64 位点
  → r̄ = 512/64 = 8（基线均匀秩）
  → AdaQ：Σ r_l = 512（非均匀，r_l ∈ [4, 128]）
```

### 统一训练超参

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 量化 | NF4 + double quant | `bnb_4bit_use_double_quant=True` |
| 计算精度 | bf16 | `bnb_4bit_compute_dtype=bfloat16` |
| LoRA alpha | α = 2r | 所有方法统一 |
| LoRA dropout | 0.05 | 所有方法统一 |
| 学习率 | 扫描 {1e-4, 2e-4, 5e-4, 1e-3} | 同一范围 |
| Epoch 数 | 3 | 所有方法统一 |
| 训练数据 | OpenHermes 2k 子集 | 所有方法统一 |
| 序列长度 | 2048 | 所有方法统一 |
| 有效 batch | 16 (micro-batch 4 × grad-accum 4) | 所有方法统一 |

### 禁止事项

- ❌ r=64 基线 vs R=512 自研（8 倍预算差）
- ❌ 基线用官方默认超参而自研调参
- ❌ 不同方法用不同数据/epoch/quantization 配置

---

## 3. 基线实现要点

| 基线 | 实现 | 关键配置 | 易错点 |
|:---|:---|:---|:---|
| QLoRA | PEFT `LoraConfig` + bitsandbytes NF4 | `r=8, alpha=16, dropout=0.05, target_modules=[q_proj,v_proj]` | 必须开 `bnb_4bit_use_double_quant=True` |
| DoRA | PEFT `LoraConfig(use_dora=True)` | 同 QLoRA 配置 + `use_dora=True` | DoRA 多幅度向量，参数略多——报告时注明 |
| AdaLoRA | PEFT `AdaLoraConfig` | 全局预算对齐到 R=512 | 4-bit 基座上重要性分数不可靠正是 Gap-3，按官方默认调度复现；[2026-11-16 评审裁定撤回]「结果差是预期证据」属预承诺结论。实测三种子 lora_E 秩恒 16、sum-of-ranks 1024=2× 预算（tinit=20>max_steps=21 剪枝未触发、token_acc .4990→.4955 未收敛），该臂不能支撑任何关于 AdaLoRA 的结论；Gap-3 已撤回 |
| AdaQ-LoRA | `adaq_precompute.py` 输出秩 + 方案 A 分组 | `budget=512, r_min=4, r_max=128` | 与 QLoRA 的唯一差异只能是秩分配，其余逐行对齐 |

### AdaQ-LoRA 接入 PEFT 的方式

```python
from peft import LoraConfig

# 从 adaq_precompute.py 输出的 ranks.json 读取方案 A 的分模块秩
import json
ranks = json.load(open("ranks.json"))
type_ranks = ranks["per_module_type_ranks (方案A, PEFT 直接使用)"]

# PEFT 原生支持 per-module-type 的不同秩
config = LoraConfig(
    r=8,  # 默认值，被以下覆盖
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    rank_pattern=type_ranks,  # {"q_proj": 12, "v_proj": 6, ...}
)
```

---

## 4. 评测协议

### 工具与版本

- **工具**：`lm-evaluation-harness`（EleutherAI）
- 版本号写进论文（`pip show lm-eval` 或 git commit hash）

### 任务矩阵

| 任务 | 设置 | 指标 | 备注 |
|:---|:---|:---|:---|
| MMLU | 5-shot | acc | 57 任务汇总 |
| GSM8K | 4-shot | strict-match acc | 对数值精度敏感；[2026-11-16 评审裁定撤回「优势主战场」预承诺] 本实验未观察到超噪声底的方法差异；strict-match 对 Instruct 基座失效（14B 实测 .18/.13 vs 官方≈0.94），永不作判据，flexible-extract 为主判据且须人工审计 |
| TruthfulQA | 0-shot | MC1/MC2 | 幻觉检测 |
| IFEval | 0-shot | prompt-level acc | 指令遵循 |

### 执行规范

- 用 `scripts/run_benchmarks.sh` 生成命令
- **混合 shot 数问题**：MMLU=5-shot, GSM8K=4-shot, TruthfulQA/IFEval=0-shot，须分批运行
- batch_size = 4（A10 bf16 推理）
- 每模型固定输出到 `results/<method>_<seed>/<task>.json`
- **原始 json 全部保留，论文数值只能从 json 汇总生成，禁止手抄**

### Merge 策略

训练完即 merge（4-bit 反量化 → bf16，merge 后 ~15GB，单卡可推理）：

```bash
# 方式1: merge 后评测（推荐，稳定）
python -m lm_eval --model hf --model_args pretrained=./merged/<method>_seed<seed>,dtype=bfloat16 --tasks mmlu --num_fewshot 5 --batch_size 4 --device cuda:0

# 方式2: adapter 直接加载（省 merge 时间）
python -m lm_eval --model hf --model_args pretrained=<base>,peft=<adapter_path>,dtype=bfloat16 --tasks mmlu --num_fewshot 5 --batch_size 4 --device cuda:0
```

---

## 5. 统计严谨性

### 多种子

- 训练种子：42 / 3407 / 2025（≥3 个）
- 报告 mean ± std
- lm-eval 对 GSM8K 等任务输出 stderr，一并报告

### 显著性检验

| 检验 | 适用场景 | 方法 | 样本量下限 |
|:---|:---|:---|:---|
| 逐任务 bootstrap | MMLU 57 任务 | AdaQ vs 最强基线，1000 次 bootstrap | 57 任务（够） |
| Wilcoxon 符号秩检验 | 跨种子 | AdaQ vs 最强基线，按 seed 配对 | **≥6 对**（精确检验下限） |
| 报告 | p 值 | p < 0.05 为显著 | — |

> **⚠️ 功效硬约束（2026-11-11 实测确认）**
> 当前实验矩阵固定 3 种子（42/3407/2025）。Wilcoxon 符号秩检验在 n=3 时
> **最小可能双侧 p = 0.25**，无论数据多干净都不可能达到 p<0.05——这不是数据问题，
> 是检验本身的下限。`scripts/aggregate_results.py` 已内置该检查：配对 <6 时
> `significance_test` 对每个任务输出 `{"note": "仅 N 对配对值；Wilcoxon 需 >=6 对才有效"}`。
>
> 因此论文在当前 3 种子设置下**只能报告描述性统计** `mean ± std (n=3)` 与每格
> `per_seed` 明细，**禁止**出现「显著」「p<0.05」「显著优于」「SOTA」等推断性表述。
> 要写显著性结论，必须先扩到 ≥6 种子（或改用不依赖配对秩的逐任务 bootstrap 并
> 明确其假设），这是实验设计变更，不是分析技巧。

### 预期值处理

- 手册 §6.4 的预期数值表只能写进实验设计文档
- **绝不允许出现在论文里**
- 论文数值以实验为准
- 若优势不及预期，转向"零额外开销下的参数效率 Pareto 前沿"叙事

---

## 6. 结果呈现规范

### 主表

方法 × 4 benchmark，加粗最优、下划线次优。

| Method | Budget | Params (M) | MMLU↑ | GSM8K↑ | TruthfulQA(MC1)↑ | IFEval↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| QLoRA (r=8) | 512 | 3.41（实测 3,407,872；Params/QLoRA=1.000） | — | — | — | — |
| DoRA (r=8) | 512 | 3.57（实测 3,571,712；1.048×） | — | — | — | — |
| AdaLoRA | 512→实测1024（剪枝未触发） | 6.82（实测 6,816,768；2.000×） | — | — | — | — |
| **AdaQ-LoRA** | **512** | **3.80（实测 3,801,088；1.115×）** | — | — | — | — |

† DoRA 多幅度向量，参数略多。

### 必配图

1. **逐层 ε_l 分布与对应 r_l 分配条形图**（方法图的核心证据）
2. **秩预算-性能曲线**（实验三：R ∈ {128, 256, 512, 768, 1024}）
3. **显存-性能散点**（效率叙事）

### 效率表

| Method | Peak VRAM (GB) | Time/epoch (h) | Precompute (min) | Inference (tok/s) |
|:---|:---:|:---:|:---:|:---:|
| QLoRA | — | — | 0 | — |
| AdaQ-LoRA | — | — | <5 | — |

指标：`torch.cuda.max_memory_allocated()`、每 epoch 训练时长、预计算耗时、推理吞吐。

### 脚注

脚注写明：秩预算 R、种子数 N=3、对齐口径（A）、lm-eval 版本。
