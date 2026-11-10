# A10 单卡训练效率优化计划（AdaQ-LoRA）

> 依据：adaq-lora-research-pipeline/references/training-optimization.md
> 硬件约束：NVIDIA A10 24GB 单卡
> 目标：Llama-3-8B QLoRA 级训练峰值显存 ≤ 23GB

---

## 1. 显存预算模型（Llama-3-8B，4-bit QLoRA 级训练）

| 组件 | 估算 | 说明 |
|:---|:---|:---|
| 4-bit 基座权重 | ~5.5 GB | NF4 + double quant |
| LoRA 参数（bf16） | 0.2–0.4 GB | R=512 全模块 |
| AdamW 优化器状态（仅 LoRA） | 2× LoRA 参数 | paged_adamw_8bit 再减半 |
| 梯度（LoRA） | ≈ LoRA 参数 | |
| 激活 | **10–14 GB（大头）** | 随 batch×seq 线性增长，梯度检查点后降至 ~1/3 |
| 预算上限 | ≤ 23 GB | 留 1GB 碎片余量 |

**结论**：显存瓶颈在激活，优化重心是 batch×seq 配置 + 梯度检查点 + packing。

> 使用 `scripts/memory_budget.py --interactive` 可交互式估算不同配置的显存。

---

## 2. 优化清单（按性价比排序，逐项验证）

**铁律3（可复现先于速度）**：每项优化启用后跑 100 步对照：loss 曲线重合（相对偏差 <1%）+ 记录显存/耗时。用 `scripts/numerical_invariance_check.py` 自动化。

| 序号 | 优化项 | 配置 | 预期收益 | 风险 | 验证方法 |
|:---:|:---|:---|:---|:---|:---|
| 1 | NF4 + 双重量化 + bf16 compute | `bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16` | 标配 | 无 | 基线，不验证 |
| 2 | FlashAttention-2 | `attn_implementation="flash_attention_2"` | 省 VRAM + 提速 20-30% | A10 Ampere 可能不兼容 | 失败退 `sdpa`，差距 <10% |
| 3 | 序列打包（packing） | `SFTConfig(packing=True, max_length=2048)` | 吞吐 1.5-2× | 跨样本注意力泄漏 | FA2 + position_ids 重置（TRL ≥0.9 默认） |
| 4 | 梯度检查点 | `gradient_checkpointing=True, use_reentrant=False` | 激活 VRAM 降至 1/3 | 时间 +30-40% | batch×seq 翻倍后净吞吐为正 |
| 5 | Liger kernel | `apply_liger_kernel_to_llama()` | 省 1-2GB + 提速 ~10% | 融合 CE 是显存大头（vocab 128k） | loss 曲线一致 |
| 6 | paged_adamw_8bit | `optim="paged_adamw_8bit"` | 优化器 VRAM 减半 | 无 | 标准操作 |
| 7 | torch.compile | `torch.compile(model, mode="reduce-overhead")` | 提速 | 兼容性不稳定 | 若 loss 漂移或报错直接放弃 |

### 典型目标配置（Llama-3-8B，seq 2048）

```
micro-batch 4 + grad-accum 4 (有效 batch 16) + 检查点 + packing
→ 峰值 ~21-22GB，约 2.5-3.5 it/s（实测为准）
```

---

## 3. 吞吐与调度

- **50 步 profile**：tokens/s、显存峰值、数据加载占比（`dataloader_num_workers=4` 起）
  - 用 `scripts/profile_training.py` 自动完成
- **lr 扫描**：用 1/10 数据 + 1 epoch 快速完成（约 2-3h），禁止在全量实验上扫参
- **checkpoint**：只存 LoRA adapter（几十 MB），`save_only_model=True`
- **断点续训**：A10 单卡环境下长实验被抢占/断电是常态，断点续训脚本常备

---

## 4. 评测与流水线省时

- 评测排队：训练完即 merge → 用 `scripts/run_eval_batch.sh` 批量跑
- MMLU 与 GSM8K 先行（主表数据），TruthfulQA/IFEval 后补
- **消融实验只用 GSM8K + MMLU 两个最敏感指标**，全量 4 任务留给主表与最终模型
  - 消融组数多（实验二约 10 组），此策略可省 ~40% 评测时间
- 跨模型实验复用同一套秩分配脚本与评测脚本

---

## 5. 预计算阶段（AdaQ 特有，手册 §5.2 修正）

### 关键修正（相对手册示例）

1. **内存安全**：FP16 模型放 CPU 逐层取出，4-bit 模型在 GPU；逐层反量化计算 ‖Ŵ−W‖_F，算完一层释放一层——**不同时驻留两份完整模型**
2. **反量化路径一致**：用 bitsandbytes 的 `dequantize_4bit`，而非手写 NF4 查表
3. **结果开源**：ranks.json 随代码开源，审稿人可复现
4. **自测**：`python3 adaq_precompute.py --self-test` 在合成数据上验证分配逻辑
   - 已验证 ✅：总秩守恒、边界约束、高误差层优先、单调性、Spearman ρ=0.9836

### 执行命令

```bash
# 预计算（需 GPU + bitsandbytes）
python3 adaq-lora-research-pipeline/scripts/adaq_precompute.py \
    --model meta-llama/Llama-3-8B \
    --budget 512 \
    --r-min 4 \
    --r-max 128 \
    --out ranks.json

# 自测（无需 GPU）
python3 adaq-lora-research-pipeline/scripts/adaq_precompute.py --self-test
```

---

## 6. 数值语义不变性检查（不可省略）

**铁律3 的执行手段**：每项优化启用后，同一批 100 条数据、固定种子跑 100 步，与基线配置对比 loss 序列，相对偏差 >1% 即回退并记录。

使用 `scripts/numerical_invariance_check.py`：

```bash
# 基线配置（无优化）
python scripts/numerical_invariance_check.py --config baseline --steps 100

# 启用某优化后
python scripts/numerical_invariance_check.py --config with_flash_attn --steps 100

# 自动对比
python scripts/numerical_invariance_check.py --compare baseline with_flash_attn
```

---

## 7. 17 周时间预算（手册 §八算力核算）

基准：2k 条 OpenHermes、seq 2048、Llama-3-8B

| 阶段 | 训练次数估算 | 训练+评测机时 | 周数 |
|:---|:---|:---|:---|
| 跑通+调试（W3-4） | ~5 次短跑 | ~20h | 2 |
| 实验一 4 方法×3 种子 | 12 次 | ~90h | 2（夜间连跑） |
| 实验二消融 ~10 组 | 10 次 | ~60h（只测 2 benchmark） | 2 |
| 实验三/四 扫描 ~7 组 | 7 次 | ~50h | 2 |
| 实验五 3 模型 | 9 次（含基线） | ~80h | 2 |
| 8-bit 鲁棒性 | 3 次 | ~20h | 1 |

**总计 ~320h 机时 ≈ 13-14 天满负荷，分布在 10 周内执行（含失败重跑余量 ~30%）**

### 周次分解

| 周 | 阶段 | 产出 |
|:---|:---|:---|
| W1-2 | 环境搭建 + 预计算 | ranks.json, 环境就绪 |
| W3-4 | 跑通+调试 | 训练流程通, 首个模型训完 |
| W5-6 | 实验一（主对比） | 主表数据 |
| W7-8 | 实验二（消融） | 消融表数据 |
| W9-10 | 实验三/四（扫描+扩展） | 秩预算曲线, 模块扩展 |
| W11-12 | 实验五/六（泛化+鲁棒） | 跨模型结果 |
| W13-17 | 论文写作+补实验 | 论文初稿→终稿 |
