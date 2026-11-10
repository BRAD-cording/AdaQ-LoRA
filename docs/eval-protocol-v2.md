# 评测口径统一方案 v2（Evaluability Protocol）

> 起因：2026-11-10 接入审计发现——历史结果分散在 `/mnt` 与 `/home` 两套环境，
> MMLU 样本数 1425（每科 25）与 5700（每科 100）混用，跨方法对比无效；
> `aggregate_results.py` 在 lm-eval 0.4.13 下产出空表。
>
> 2026-11-11 二次审计发现 v2.0 驱动的 marker 缺陷（见 §5），已升级 v2.1 并补跑。

## 1. 统一口径硬约束

| 维度 | v2 规则 | 落地位置 |
|:---|:---|:---|
| 评测样本 | MMLU 全量 14042；GSM8K 全量 1319；**禁止 `--limit` 混入主表** | `run_eval_unified.sh`（LIMIT 留空） |
| Few-shot | MMLU=5, GSM8K=4, TruthfulQA/IFEval=0 | 同上 TASKSETS 定义 |
| 基座 | `models/Meta-Llama-3-8B`（本地 30G） | `eval_worker.sh` |
| 适配器加载 | bf16 基座 + PEFT adapter 直接加载（**不是** 4-bit 基座） | `run_eval_unified.sh` |
| 网络 | 默认 `HF_HUB_OFFLINE=1`（数据集已缓存，绕开 hf-mirror 429）；首次下载用 `HF_ENDPOINT=https://hf-mirror.com` + `OFFLINE=0` | 同上 |
| 输出目录 | 一次口径 = 一个 OUT_DIR；禁止跨目录合并 | `aggregate_results.py --results-dir` |
| 任务标记 | **每任务独立 marker**：`.done_mmlu` / `.done_gsm8k` / `.done_safety`；touch 前必须机读校验产物 | `verify_eval_artifact.py` |

> ⚠️ 适配器加载是 **bf16 基座**。若论文正文写"4-bit 合并后推理"，需另跑 `EVAL_MODE=merge` 一组，
> 并在方法节明确"两种推理路径都报告"。当前主表口径 = bf16 + adapter。

## 2. 结果目录约定（v2.1）

```
results_v2/            # 统一样本数、bf16+adapter、全量（论文主表）
  <method>_seed<seed>/
    eval_mmlu.log          # MMLU 独立日志
    eval_gsm8k.log         # GSM8K 独立日志
    eval_safety.log        # TruthfulQA + IFEval 独立日志
    results_*.json         # lm-eval 原始输出（每次运行一个时间戳文件，互不覆盖）
    .done_mmlu             # 每任务独立完成标记
    .done_gsm8k
    .done_safety
  summary.json             # 唯一论文数值来源
```

`aggregate_results.py` 按 `(method, seed, task)` 合并所有 `results_*.json`，
同键取 mtime 最新，因此补跑产物是**新增文件**而非覆盖，天然安全。

## 3. 阶段状态

| 阶段 | 内容 | 状态 |
|:---|:---|:---:|
| P0-1 | 修 aggregate 指标键 + 口径审计 | ✅ |
| P0-2 | 全量 MMLU 14042 / GSM8K 1319 跑完 12 run（补齐缺失任务） | 🔄 `run_eval_missing.py` 补跑中 |
| P0-3 | 落成 `results_v2/summary.json` 且口径审计 exit 0 | 🔄 待补跑完成 |
| P1 | git 首次入库 + .gitignore | ✅ |
| P2 | TruthfulQA(mc1/mc2) + IFEval 0-shot，12 run | 🔄 与补跑同批 |
| P2 | 双卡并行评测驱动 | ✅ v2.1 |

## 4. 复现命令

```bash
# 全量单卡（论文主表）
OUT_DIR=results_v2 LIMIT= bash scripts/run_eval_unified.sh

# 双卡（seed 42 在 GPU0，seed 3407 在 GPU1）
OUT_DIR=results_v2 bash scripts/eval_worker.sh 0 "qlora dora adalora adaq" "42"
OUT_DIR=results_v2 bash scripts/eval_worker.sh 1 "qlora dora adalora adaq" "3407"

# 补齐缺失任务（推荐：自动扫描已有产物，双卡并行，跑完自动聚合）
OUT_DIR=results_v2 WANT=mmlu,gsm8k,safety GPUS=0,1 \
  python scripts/run_eval_missing.py

# 汇总（退出码 2 = 口径不一致，必须处理）
python scripts/aggregate_results.py --results-dir results_v2 --out results_v2/summary.json
```

## 5. 缺陷复盘：v2.0 为什么产出了一张假表

**现象**：v2.0 跑完 12 个 run、watchdog 报告 `all done`、`.done_main` 全齐，
但 `aggregate_results.py` 的主表里 qlora 的 GSM8K 是 `—`，各方法 `n_seeds` 参差 1~3。

**根因链**（逐条有证据）：

1. `run_eval_unified.sh` v2.0 第 58 行：
   `main) SETS=("mmlu|5|main" "gsm8k|4|main")` —— 两个任务**共用 marker `main`**。
   第 108 行第一个任务成功后 `touch .done_main`，第 89 行第二个任务开头
   看到 `.done_main` 直接 `SKIP`。于是每个 run 只跑了一个任务。
2. 更隐蔽：MMLU 因 hf-mirror 返回 **429 Too Many Requests**
   （`https://hf-mirror.com/api/datasets/cais/mmlu/tree/.../international_law`）
   加载失败，但脚本不 `break`，循环继续跑 GSM8K；GSM8K 成功反过来
   `touch .done_main`，**把 MMLU 的失败永久掩盖**。
3. `watch_eval.sh` 只认 `.done_main`，看到文件就认为该 run 完整，不再补跑。
4. `aggregate_results.py` 的口径审计只检查 `sample_len` 是否跨方法一致，
   **没有检查"每个 run 是否具备全部任务"**，所以 exit 0 但表是残缺的。
   12 个 run 中 8 个只有 MMLU、4 个只有 GSM8K，没有任何一个同时具备两项。

**修复（v2.1）**：

- 每任务独立 marker（`.done_mmlu` / `.done_gsm8k` / `.done_safety`）。
- `touch` marker 前调用 `verify_eval_artifact.py` 机读校验该任务指标确实出现在
  `results_*.json` 中，命令返回 0 不等于产物到齐。
- 默认 `HF_HUB_OFFLINE=1`，mmlu/gsm8k/truthfulqa/ifeval 数据集全部本地缓存，
  彻底绕开 429；`OFFLINE=0` 仅在首次下载数据集时使用。
- 任一任务失败记入 `FAILED` 并 `exit 2`，不静默继续、不掩盖。
- `run_eval_missing.py` 按"实际产物里有哪些 task"生成缺失清单，不认旧
  `.done_main`，补跑产物为新增时间戳文件，由聚合器合并。

**同类排查**：`results_v2/` 下任何 run 若 `n_seeds` 不等于 3，先跑
`python scripts/verify_eval_artifact.py results_v2/<run> <marker>` 定位，
不要直接看 `.done_*` 标记。

## 6. 已知结论（基于全量数据，供写作校准措辞）

- 四种方法在 **MMLU** 上差异落在种子内部波动范围内（约 0.653–0.655，
  方法间差 ≤0.001）。MMLU 是知识问答，2000 条 OpenHermes 指令微调动不了它，
  **不能作为"方法有效性"的主证据**；应把主证据放到指令敏感任务上。
- **adalora 两个种子 GSM8K strict 都是 629/1319 = 0.4769**，但 flexible-extract
  分别是 637/1319 与 635/1319，且两个 adapter 权重 192 个张量无一相同
  （max abs diff 0.144）。结论：**离散计数巧合**，不是权重相同、不是 bug，
  也不足以写成"秩分配对种子不敏感"的稳定性论据。样本量 n=1319、p≈0.477
  时该整数重合属正常范围。
- `results_v2/summary.json` 的 `_audit` 必须 `mismatch_task_count == 0`，
  且每个方法每个任务的 `n_seeds == 3`，才能进入论文表。
