# SOTA 对比与权威评测协议

## 目录
1. Gap 论证维护（防被抢先/防已解决）
2. 公平对比协议（参数对齐铁律）
3. 基线实现要点
4. 评测协议（lm-evaluation-harness）
5. 统计严谨性
6. 结果呈现规范

---

## 1. Gap 论证维护

手册 §2.2 的四个 Gap 是论文的立论根基，每次正式写作/投稿前必须用 scholar 插件重查一次：

- 检索词建议：`quantization-aware rank allocation LoRA`、`adaptive rank quantized LLM fine-tuning`、`low-rank budget allocation 4-bit`、`NF4 error compensation adapter`。
- 重点核查 2024 年后工作：LoftQ、QA-LoRA、LQ-LoRA、IR-QLoRA、AQLM 系、QuaRot 系是否已做"量化误差驱动的秩分配"。若出现部分重叠工作：立即调整叙事为"它们补偿幅度/初始化，我们分配容量"，并在 Related Work 中明确区分。
- Gap 表述必须与所引文献严格对应，禁止"所有 PEFT 方法都未解决"这类过宽断言——写成"在截至投稿的公开工作中，未见将逐层 4-bit 量化误差纳入秩预算分配的方法"，并附检索日期。

## 2. 公平对比协议（参数对齐铁律）

手册实验一中"QLoRA r=64 vs AdaQ R=512"存在对齐歧义，执行前必须统一口径。两种合法口径，选一种并贯穿全文：

**口径 A：总秩预算对齐（推荐，叙事最干净）**
- AdaQ-LoRA：Σ_l r_l = R = 512。
- QLoRA/DoRA：均匀秩 r̄，使 Σ_l r̄ = 512（32 层 × 7 模块 = 224 个适配位点 → r̄ ≈ 2.3，太小会退化，故需声明适配位点子集，如仅 q/v_proj → 64 位点 → r̄ = 8）。
- AdaLoRA：官方全局预算参数对齐到同一 R。

**口径 B：峰值显存对齐**
- 各方法在 A10 上以不 OOM 的最大秩运行，报告各自显存与性能。适合"同硬件预算"叙事，但参数效率故事变弱。

禁止事项：
- 禁止 r=64 基线 vs R=512 自研（8 倍预算差），审稿人必提。
- 禁止基线用官方默认超参而自研调参。所有方法共享：同一数据、同一 epoch 数、同一 lr 扫描范围、同一量化配置（NF4 + double quant + bf16 compute）。
- LoRA alpha 统一（如 α=2r），dropout 统一 0.05。

## 3. 基线实现要点

| 基线 | 实现 | 易错点 |
| :--- | :--- | :--- |
| QLoRA | PEFT LoraConfig + bitsandbytes NF4 | 必须开 `bnb_4bit_use_double_quant=True` 与论文一致 |
| DoRA | PEFT `use_dora=True` | DoRA 多一个幅度向量，参数略多——报告时注明 |
| AdaLoRA | PEFT AdaLoraConfig | 4-bit 基座上重要性分数不可靠正是 Gap-3，按官方实现跑即可，结果差是预期证据，不要"修"它 |
| AdaQ-LoRA | scripts/adaq_precompute.py 输出秩 + 方案 A 分组 | 与 QLoRA 的唯一差异只能是秩分配，其余逐行对齐 |

## 4. 评测协议

工具：`lm-evaluation-harness`（EleutherAI），版本号写进论文。

| 任务 | 设置 | 指标 | 备注 |
| :--- | :--- | :--- | :--- |
| MMLU | 5-shot | acc | 57 任务汇总 |
| GSM8K | 4-shot（与手册一致） | strict-match acc | 对数值精度最敏感，是 AdaQ 优势主战场 |
| TruthfulQA | 0-shot | MC1/MC2 | 幻觉 |
| IFEval | 0-shot | prompt-level acc | 指令遵循 |
| ARC-C / HellaSwag（可选补强） | 0-shot | acc_norm | 常识 |

执行规范：
- 用 [../scripts/run_benchmarks.sh](../scripts/run_benchmarks.sh) 生成命令；adapter 需 merge 到 4-bit 反量化后的 bf16 底座再评测，或直接用 `--model_args pretrained=base,peft=adapter_path`。
- batch_size 在 A10 上 4-bit 推理建议 4–8；记录每任务耗时（评测也是时间预算大头，GSM8K 生成式评测约 1–2h/模型）。
- 每个模型固定输出到 `results/<method>_<seed>/<task>.json`，原始 json 全部保留，论文数值只能从 json 汇总生成，禁止手抄。

## 5. 统计严谨性

- 训练 ≥3 种子（如 42/3407/2025），报告 mean ± std。评测侧：lm-eval 对 GSM8K 等给出 stderr，一并报告。
- 主对比（AdaQ vs 最强基线）做配对显著性检验：MMLU 用逐任务 bootstrap，跨种子用 Wilcoxon 符号秩检验，报告 p 值。
- 手册 §6.4 的预期数值表只能写进实验设计文档，**绝不允许**出现在论文里；论文数值以实验为准，若优势不及预期，转向强调"零额外开销下的参数效率 Pareto 前沿"叙事，而不是改数据。

## 6. 结果呈现规范

- 主表：方法 × 4 benchmark，加粗最优、下划线次优；脚注写明秩预算、种子数。
- 必配图：①逐层 ε_l 分布与对应 r_l 分配条形图（方法图的核心证据）；②秩预算-性能曲线（实验三）；③显存-性能散点（效率叙事）。
- 效率表：峰值显存（torch.cuda.max_memory_allocated）、每 epoch 训练时长、预计算耗时（<5 min 要实测）、推理吞吐。
