# A10 单卡训练效率优化（24GB 显存约束）

## 目录
1. 显存预算模型
2. 优化清单（按性价比排序，逐项验证）
3. 吞吐与调度
4. 评测与流水线省时
5. 预计算阶段（AdaQ 特有）
6. 数值语义不变性检查
7. 17 周时间预算

---

## 1. 显存预算模型（Llama-3-8B，4-bit QLoRA 级训练）

| 组件 | 估算 | 说明 |
| :--- | :--- | :--- |
| 4-bit 基座权重 | ~5.5 GB | NF4 + double quant |
| LoRA 参数（bf16） | R=512 全模块约 0.2–0.4 GB | 秩分配不改变总量级 |
| AdamW 优化器状态（仅 LoRA） | 2× LoRA 参数 | paged_adamw_8bit 再减半 |
| 梯度（LoRA） | ≈ LoRA 参数 | |
| 激活 | **大头，10–14 GB** | 随 batch×seq 线性增长，梯度检查点后降至 ~1/3 |
| 预算上限 | ≤ 23 GB | 留 1GB 碎片余量 |

结论：显存瓶颈在激活，优化重心是 batch×seq 配置 + 梯度检查点 + packing。

## 2. 优化清单（按性价比排序）

按序启用，每项启用后跑 100 步对照：loss 曲线重合（相对偏差 <1%）+ 记录显存/耗时。

1. **NF4 + 双重量化 + bf16 compute**（标配，非选项）：`bnb_4bit_use_double_quant=True`，`bnb_4bit_compute_dtype=bfloat16`。
2. **FlashAttention-2**：`attn_implementation="flash_attention_2"`。A10 支持；省显存且提速 20–30%。注意：A10 为 Ampere 架构，FA2 支持 bf16/fp16，不支持滑窗以外的特化——若装不上，退 `sdpa`，差距 <10%。
3. **序列打包（packing）**：TRL `SFTTrainer(packing=True, max_length=2048)`。短样本居多的指令数据（OpenHermes 均值 <1k token）吞吐可提升 1.5–2×。风险：跨样本注意力泄漏——用 `attn_implementation="flash_attention_2"` + position_ids 重置（TRL ≥0.9 默认处理）；论文中声明 packing 策略。
4. **梯度检查点**：`gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}`。激活显存降至 ~1/3，时间 +30–40%；用它换取 batch×seq 翻倍后净吞吐仍为正。
5. **Liger kernel**：`from liger_kernel.transformers import apply_liger_kernel_to_llama; apply_liger_kernel_to_llama()`（训练前调用）。融合 RMSNorm/RoPE/SwiGLU/CrossEntropy，省 1–2GB + 提速 ~10%。交叉熵融合是显存大头（vocab 128k）。
6. **优化器**：`optim="paged_adamw_8bit"`；单卡 24GB 不必上 adafactor。
7. **torch.compile**：最后尝试，`torch.compile(model, mode="reduce-overhead")`；与 4-bit + LoRA 组合兼容性不稳定，编译耗时本身 10–20 min，只在长训练（>10h）时值得。若 loss 漂移或报错直接放弃，不纠缠。

典型目标配置（Llama-3-8B，seq 2048）：micro-batch 4 + grad-accum 4（有效 batch 16）+ 检查点 + packing → 峰值 ~21–22GB，约 2.5–3.5 it/s 量级（实测为准）。

## 3. 吞吐与调度

- 先跑 50 步 profile：tokens/s、显存峰值、数据加载占比（`dataloader_num_workers=4` 起）。
- lr 扫描用 1/10 数据 + 1 epoch 快速完成（约 2–3h），禁止在全量实验上扫参。
- checkpoint 只存 LoRA adapter（几十 MB），`save_only_model=True`，省磁盘与时间。
- 断点续训脚本常备；A10 单卡环境下长实验被抢占/断电是常态。

## 4. 评测与流水线省时

- 评测排队：训练完即 merge（4-bit 反量化 → bf16，merge 后 ~15GB，单卡可推理），用 scripts/run_benchmarks.sh 批量跑；MMLU 与 GSM8K 先行（主表数据），TruthfulQA/IFEval 后补。
- 消融实验只用 GSM8K + MMLU 两个最敏感指标，全量 4 任务留给主表与最终模型——消融组数多（手册实验二约 10 组），此策略可省 ~40% 评测时间。
- 跨模型实验（实验五）复用同一套秩分配脚本与评测脚本，只换模型名。

## 5. 预计算阶段（AdaQ 特有，手册 §5.2 的修正）

- FP16 模型放 CPU（`device_map="cpu"`）逐层取出，4-bit 模型在 GPU；逐层反量化计算 ‖Ŵ−W‖_F，算完一层释放一层——**不要**同时驻留两份完整模型（手册示例整体加载 FP16 再删，8B 模型约 16GB 内存，机器内存紧张时会 OOM）。
- 反量化路径必须与训练一致：用 bitsandbytes 的 `dequantize_4bit`，而非手写 NF4 查表，避免预计算的 E_l 与训练实际的量化权重不一致。
- 预计算结果（ranks.json）随代码开源，审稿人可复现。
- 用 `python3 adaq_precompute.py --self-test` 在合成数据上验证分配逻辑（幂律谱合成 → 检查高 ε 层获得更高秩、总量守恒、截断后重新归一化）。

## 6. 数值语义不变性检查（不可省略）

每项优化启用后：同一批 100 条数据、固定种子跑 100 步，与基线配置对比 loss 序列，相对偏差 >1% 即回退该项并记录。这条是铁律 3 的执行手段——优化省的时间不能以实验结论可信度为代价。

## 7. 17 周时间预算（手册 §八 的算力核算）

以 2k 条 OpenHermes、seq 2048、Llama-3-8B 为基准：单次训练（3 epoch）约 4–6h；评测一轮约 3–4h。

| 阶段 | 训练次数估算 | 训练+评测机时 | 周数 |
| :--- | :--- | :--- | :--- |
| 跑通+调试（W3–4） | ~5 次短跑 | ~20h | 2 |
| 实验一 4 方法×3 种子 | 12 次 | ~90h | 2（可夜间连跑） |
| 实验二消融 ~10 组 | 10 次 | ~60h（只测 2 benchmark） | 2 |
| 实验三/四 扫描 ~7 组 | 7 次 | ~50h | 2 |
| 实验五 3 模型 | 9 次（含基线） | ~80h | 2 |
| 8-bit 鲁棒性 | 3 次 | ~20h | 1 |

总计 ~320h 机时 ≈ 13–14 天满负荷，分布在 10 周内执行（含失败重跑余量 ~30%），与手册 §6.3 路线图兼容。W11 之后的写作期可穿插补实验。
