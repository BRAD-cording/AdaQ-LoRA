# AdaQ-LoRA 投稿前评审 —— 公平基线与 SOTA 对比

> 评审角色：公平基线与 SOTA 对比专家（预算对齐 / 超参对称 / 相关工作 delta / 「看似公平实不公平」设计识别）
> 评审日期：2026-11-15
> 证据原则：本文所有数字均为本次评审在远端 `/home/ubuntu/AdaQ-LoRA` **实读文件 / 实跑脚本**取得，附文件路径与行号。未实测的一律标注为「估计」或「待作者澄清」，不作为结论依据。GPU 训练进程未被触碰，全程只读。

---

## 0. 总判定

# **Reject（当前形态）→ 大修后可降至 Major Revision**

理由一句话：**本项目的对照实验在「参数预算」「评测基座口径」「基线收敛条件」三个维度上同时不对齐，且这三处不对齐的方向都不利于得出可信结论；而已实测到的方法效应（≤0.001 MMLU）比同配置重跑的噪声底（11.5 GSM8K 点）小两个数量级。**

这不是「结果不够好」的问题，是**当前实验装置无法产生任何可用于支撑或否证主张的数值**。项目内部文档（`docs/claim-revision-norm-aware.md`）已诚实记录了阴性结论，方向正确、态度可嘉；但阴性结论同样需要有效装置——**用一个坏装置得到的阴性结果，不能证明「方法无效」，只能证明「这次测量无效」**。

### 0.1 实读 / 实跑证据清单

| 类别 | 路径 | 取证方式 |
|:--|:--|:--|
| 主表产物 | `results_v2/FINAL_REPORT.md`、`results_v2/summary.json`、`results_v2/**/results_*.json`（12 run） | 全文读 + 解析 `config.model_args` |
| PTQ/NF4 表 | `results_v2_ptq_nf4/summary.json`、`results_v2_ptq_nf4/*/results_*.json`（10 文件） | 全文读 + 解析 `model_args` / `sample_len` |
| 14B 表 | `results_v2_qwen14b/summary.json`、`results_v2_qwen14b_seed2025/`、`results_v2_qwen14b_seed3407/`（不存在） | 全文读 + 目录核查 |
| 消融 | `results_v2_ablation/`、`results_v2_ablation_multiseed/summary.json`、`results_v2_ablation_schemeA_s{3407,2025}/` | 解析全部 `results_*.json` |
| Adapter 真值 | `outputs/adapters/*/{adapter_config.json,adapter_model.safetensors,checkpoint-*/trainer_state.json}`、`outputs/adapters_qwen14b/*`、`outputs/ablation_adapters*/**` | **直接解析 safetensors 二进制头**（8B 长度前缀 + JSON header），逐张量取 shape/dtype/字节哈希，不加载模型、不占显存 |
| 秩分配真值 | `ranks.json`、`ranks_qwen14b.json`、`configs/ablation_ranks/*.json` | 解析 `per_layer_ranks` / `layer_stats`，**独立重算** CV、Spearman、预算守恒 |
| 训练/评测代码 | `scripts/train.py`、`scripts/run_eval_unified.sh`、`scripts/run_eval_ptq_nf4.py`、`scripts/run_eval_b4_qwen14b.py`、`scripts/aggregate_b4_3seeds.py`、`scripts/run_b4_qwen14b.sh`、`run_pipeline.py` | 逐行读 |
| 协议文档 | `docs/sota-comparison-protocol.md`、`docs/eval-protocol-v2.md`、`docs/paper-framework.md`、`docs/paper-outline.md`、`docs/claim-revision-norm-aware.md`、`docs/theory-review-report.md` | 全文读 |
| 系统指标 | `results_v2/system_metrics.json`、`outputs/**/training_stats.json`（18 份） | 全文读 |

### 0.2 与同侪评审的关系

`docs/review/` 下已存在 `jcr-gate.md`、`rank-budget.md`、`nf4-spectral.md`。本文与 `rank-budget.md` §1.3/§1.4/§1.6/§2.1–2.3 在「方案A 破坏预算守恒」「秩预算≠参数预算」「α/r 混杂」「六策略塌缩为四配置」上**独立收敛到同一结论**（我用 safetensors 二进制头 + `trainer_state.json` 取证，路径不同、结论一致，互为交叉验证）。本文**新增**的部分集中在 §2.2（AdaLoRA 全程未剪枝的 checkpoint 级铁证 + Gap-3 因果主张无证据）、§3.3（两份协议文档互相矛盾且推荐路径在代码中不存在）、§3.5–3.7（指标政策事后切换 + 14B strict 口径失效）、§4（缺失基线按致命度排序）、§5.2–5.3（intro 前提被自家 `ranks.json` 证伪到小数点后四位）。

**一处需要与同侪核对的分歧**：`nf4-spectral.md:20,39` 记「error_only md5 ≡ uniform（8940e8dbaf65）」。我实测 `configs/ablation_ranks/ranks_error_only.json` = `19b80ce915cd3560ac50a70726ad645a`、`ranks_uniform.json` = `7b3f73af8ed789604aca09c2e0adef2b`，**两份 config JSON 的 md5 不同**。塌缩是真的，但发生在**训练产物层**而非 config 文件层：`outputs/ablation_adapters/{error_only,uniform}/adapter_config.json` 的 `rank_pattern` 逐键比对 **0/64 层不同**（均 all-8），`outputs/ablation_adapters_schemeA_*/{error_only,uniform}` 均为 `{q_proj:8, v_proj:8}`。建议三份评审统一表述为「**配置文件不同、有效秩图相同**」，否则 rebuttal 时会被作者以 md5 不同为由反驳。

---

## 1. 问题①：基线与基准（尤其 RTN）是否绝对正确？

### 1.1 任务前提先被证伪：本项目**没有**用 bnb-fp4，用的是 nf4

评审任务书假设「PTQ 基线用 bitsandbytes fp4（`BitsAndBytesConfig` 默认 `quant_type`）作 RTN 近亲」。**该前提不成立**，实测两处均为 `nf4`：

- `scripts/run_eval_ptq_nf4.py:110-122`，`model_args_str()` 显式拼接：
  `load_in_4bit=True, bnb_4bit_quant_type=nf4, bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True, dtype=bfloat16`
- `scripts/train.py:146-151`，`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`，四方法共用同一构造。
- **运行时产物核对（10/10 文件）**：`results_v2_ptq_nf4/*/results_*.json` 的 `config.model_args` 全部含 `'bnb_4bit_quant_type':'nf4'`、`'bnb_4bit_use_double_quant':True`。

结论：**代码与产物一致，没有踩「bnb 默认 fp4」这个坑**。这一点是干净的，应当在论文实验节明写 `nf4` 而非泛指「4-bit」，以免审稿人默认怀疑。

### 1.2 那么 `baseline_nf4` 是不是 RTN？—— 是 RTN，但**不是文献语境的 RTN**，命名必须改

**可辩护的部分**：bnb 的 4-bit 路径**没有任何校准/误差补偿搜索**——每 64 个权重一块，用块内 absmax 定标，每个值吸附到码本中最近的一项。这在「round-to-nearest」的字面意义上**就是 RTN**。所以 `baseline_nf4` 作为「NF4 量化基座、不挂 adapter」的**内部控制**是正确且必要的，它干净地隔离了 adapter 的贡献。

**不可辩护的部分（命名与参照系）**：

1. PTQ 文献里「RTN」约定俗成指 **均匀 INT4 RTN**（sym/asym，group size 128，最近整数）。NF4 是 Dettmers et al. (2023) 的**非均匀、对 N(0,1) 分位最优**的 4-bit 码本。把 NF4 结果标为「RTN baseline」而不加限定，量化方向的审稿人会直接判为类别错误。
2. `bnb_4bit_use_double_quant=True` 对 scale 再做一次 8-bit 量化（省约 0.373 bit/param），**改变了误差分布**。所以这一行的准确身份是「**NF4 + Double Quantization，round-to-nearest，无校准**」，即 **QLoRA 的基座量化器本身**。
3. 它**不是** PTQ SOTA 参照系。论文若声称「量化误差层间异质」，参照系必须包含**真正做误差补偿**的方法（GPTQ 的逐列 Cholesky/Hessian 补偿、AWQ 的激活感知逐通道缩放），否则「误差」这一概念在论文里没有对手。

**精确改法**（见 §6 修复项 F1）：`results_v2_ptq_nf4/summary.json` 的键 `baseline_nf4` → `base_nf4dq_noadapter`；论文表格行名 → 「Llama-3-8B, NF4+DQ (RTN, no calibration), no adapter」；若要保留一个「RTN」字样的行，另加 `rtn_int4_g128`（均匀 INT4，g=128），两者并列。

**若坚持留在 bnb 生态内做最朴素的 RTN**：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False)`——这才是 bnb 的默认 `quant_type`，也是最不加修饰的非均匀 4-bit 最近舍入。这一行成本极低（一次评测，无需训练），建议直接补上，用来堵住「你连 RTN 都没跑」这句最廉价的攻击。

### 1.3 `baseline_nf4` 只有 n=1，而主表**完全没有基座锚点**

- `scripts/run_eval_ptq_nf4.py:48-54` 的 `CONFIGS` 只含 `*_seed42`，**PTQ/NF4 表全部 n=1**，无种子重复、无 std。而主表是 n=3。跨表统计强度不对称。
- **更严重**：`results_v2/summary.json` 顶层键实测为 `['adalora','adaq','dora','qlora','_audit','significance_test']` —— **8B BF16 主表没有任何基座行**（无 `baseline_bf16`、无 no-adapter 锚点）。全库检索 `baseline_bf16|bf16 基线|无 adapter|纯基座` 在 `docs/`、`scripts/`、`results_v2/` 中 **0 命中**。
- 后果：主表四方法并列在 .6533–.6543（MMLU 极差 0.0010），**读者无法判断这是「四个方法一样好」还是「四个方法一样没用」**。而 NF4 表给出了强提示——四个微调臂的 MMLU **全部低于** `baseline_nf4`：

| 行 | MMLU (acc) | GSM8K (exact_match, strict-match) |
|:--|--:|--:|
| `baseline_nf4`（无 adapter） | **0.636376584532118** | 0.41243366186504926 |
| `adalora_nf4` | 0.6356644352656317 | 0.4184988627748294 |
| `dora_nf4` | 0.6343113516593077 | 0.4230477634571645 |
| `adaq_nf4` | 0.6335279874661729 | **0.40636846095526913（五行最差）** |
| `qlora_nf4` | 0.6333855576128756 | 0.41243366186504926 |

  即 **NF4 口径下，2000 条 OpenHermes × 3 epoch 的微调让 MMLU 下降了 0.07–0.30 点**。BF16 主表极可能出现同向结果，却因为没有锚点行而**被完全隐藏**。这是「看似公平实不公平」的典型形态：四臂对称、种子对称、任务对称，唯独缺了那个能证伪「微调有益」的对照。

  项目自己的 `docs/eval-protocol-v2.md` §6 已写下「MMLU 是知识问答，2000 条 OpenHermes 指令微调动不动它，**不能作为『方法有效性』的主证据**」——但 `results_v2/FINAL_REPORT.md` 的主表仍以 MMLU 为首列，14B 的头条也是 MMLU 持平。**内部写作校准规则未被执行**。

### 1.4 `qlora_nf4` 与 `baseline_nf4` 的 GSM8K 逐位同值：判定为计数巧合，但必须留证

实测两个指标同时逐位相同：

- strict-match：两者均 `0.41243366186504926`；stderr 分别为 `0.013559628790941445` 与 `0.013559628790941447`（**末两位不同 ⇒ 两次独立计算，不是文件复制**）。
- flexible-extract：两者均 `0.41925701288855194`。

判定依据：

1. `0.41243366186504926 × 1319 = 544.0`、`0.41925701288855194 × 1319 = 553.0` —— GSM8K 是**离散计数**指标，1319 题下每 1 题 = 0.000758。两模型答对数恰好同为 544 / 553 是完全可能的。
2. stderr 末位不同**排除了产物复制**（若复制则连 stderr 也逐位相同）。
3. **同一行的 MMLU 不同**（`qlora_nf4` .6333855576128756 vs `baseline_nf4` .636376584532118），MMLU 是 14042 题的连续型 acc，adapter 确已加载并改变了输出 ⇒ **加载链路正常**。

**但仍有一处必须查清的三向碰撞**：数值 `0.41470811220621684`（= 547/1319）在**三个互不相干的实验**中同时出现——
`results_v2_ptq_nf4/adaq_nf4` 的 **flexible-extract**、`results_v2_ablation_schemeA_s3407/equal_weight` 的 **strict-match**、`results_v2_ablation/random`（逐层方案B）的 **strict-match**。
三者的 adapter 不同（q12/v4 vs q10/v4 vs 逐层随机）、评测基座不同（NF4 vs BF16 vs BF16）、指标不同（flex vs strict vs strict）。**我倾向于巧合**（547 落在这一带的概率不低），但三次精确碰撞已越过「顺手解释掉」的阈值。修复项 F2 要求对这三份产物跑 `scripts/verify_eval_artifact.py` 并保留各自的 per-sample 预测文件做去重比对——这是本项目已经存在、且 `docs/eval-protocol-v2.md` §5 证明**曾经救过一次假表**的工具，应当复用。

> `docs/eval-protocol-v2.md` §5 记录：v2.0 曾因两个任务共用 `.done_main` marker + MMLU 遭遇 hf-mirror 429 未 break，产出了一张「watchdog 报 all done、聚合器 exit 0、但 12 个 run 无一同时具备 MMLU 与 GSM8K」的**假表**。**这说明本 harness 有产出可信假阳性的既往史**，因此本次评审对所有「看起来太干净」的同值都要求留证，不是过度谨慎。

---

## 2. 问题②：基线集 QLoRA / DoRA / AdaLoRA 是否公平？

### 2.1 参数预算：**秩预算对齐 ≠ 参数预算对齐**，实测跨度 2.000×

直接解析 `adapter_model.safetensors` 二进制头（8 字节长度前缀 + JSON header），逐张量累加元素数，seed42：

| 方法 | 路径 | 张量数 | **实测元素数** | vs QLoRA | `trainable_ratio`（`results_v2/system_metrics.json`） | `peak_vram_gb` |
|:--|:--|--:|--:|--:|--:|--:|
| qlora | `outputs/adapters/qlora_seed42` | 128 | **3,407,872** | 1.000× | 4.2438e-4 | 14.80 |
| dora | `outputs/adapters/dora_seed42` | 192 | **3,571,712** | 1.048× | 4.4478e-4 | 15.01 |
| **adaq** | `outputs/adapters/adaq_seed42` | 128 | **3,801,088** | **1.115×** | 4.7335e-4 | 14.81 |
| **adalora** | `outputs/adapters/adalora_seed42` | 192 | **6,816,768** | **2.000×** | 8.4888e-4 | 14.84 |

手算复核（Llama-3-8B：`q_proj` 4096×4096，`v_proj` 4096×1024，GQA 8 KV heads，32 层）：
- qlora/层 = 8×(4096+4096) + 8×(4096+1024) = 65,536 + 40,960 = 106,496 → ×32 = **3,407,872** ✓
- adaq/层 = 12×8192 + 4×5120 = 98,304 + 20,480 = 118,784 → ×32 = **3,801,088** ✓
- dora = qlora + 64 个 `dora_scale` 向量 = 3,407,872 + 32×(4096+1024) = 3,407,872 + 163,840 = **3,571,712** ✓

**关键机制（GQA 放大效应）**：`ranks.json` 的 `per_module_type_ranks` = `{q_proj:12, v_proj:4}`，sum-of-ranks = (12+4)×32 = **512，与 QLoRA 的 8×2×32 = 512 完全相等，delta = 0**。但 **`q_proj` 的输出维是 4096，`v_proj` 只有 1024**——把秩从 v 挪到 q，同样的 sum-of-ranks 换来 **+11.5% 的参数量**。也就是说：

> **`docs/sota-comparison-protocol.md` §2 选定的「口径 A —— 总秩预算对齐」在 GQA 模型上不是公平口径。它对齐的是一个与参数量、与计算量都不成比例的中间量。adaq 在 8B 上白拿 11.5% 参数，方向恰好有利于 adaq。**

而 §6 的主表模板把四个方法的 `Params (M)` 列**全部写成 2.7**。实测是 3.41 / 3.57 / 3.80 / **6.82**，跨度 2.000×。**若按模板填表，主表将直接印出一个假的预算对齐声明。**

**另注（显存不能替代参数对齐）**：adalora 参数翻倍，`peak_vram_gb` 只从 14.80 涨到 14.84（+0.27%），因为峰值显存被 8B 冻结基座 + 激活主导。所以「四方法显存都≈14.8GB，可见预算对齐」这类论证**无效**，`docs/paper-framework.md` §4.5 计划的「显存-性能散点图」在当前数据下会是一条水平线，不能用作 Pareto 论证。

### 2.2 AdaLoRA：**2.000× 预算 + 自适应机制全程未启动 + 未收敛**（本轮评审最硬的新证据）

`outputs/adapters/adalora_*/adapter_config.json` 实测配置（三种子完全一致）：
```
peft_type=ADALORA  r=8  init_r=16  target_r=8  total_step=200
tinit=20  tfinal=60  deltaT=5  beta1=0.3  beta2=0.3  orth_reg_weight=0.1
lora_alpha=16  rank_pattern=null  peft_version=0.19.1
```
来源：`scripts/train.py:30-53`，**全部硬编码**，`total_step=200` 是一个与真实训练长度无关的常数。

`outputs/adapters/adalora_seed*/checkpoint-*/trainer_state.json` 实测 `max_steps = 21`。

**逐 checkpoint 解析 `lora_E` 张量形状（`lora_E` 的第 0 维即该模块的当前秩）**：

| run | checkpoints | `lora_E` 秩集合 | `total_elems` | `global_step/max_steps` |
|:--|:--|:--|--:|:--|
| `adalora_seed42` | 2, 7, 14, 21 | **`{16}`（全部 64 个）** | **6,816,768（恒定）** | 2/2 → 7/21 → 14/21 → 21/21 |
| `adalora_seed3407` | 7, 14, 21 | **`{16}`** | **6,816,768（恒定）** | 7/21 → 14/21 → 21/21 |
| `adalora_seed2025` | 7, 14, 21 | **`{16}`** | **6,816,768（恒定）** | 7/21 → 14/21 → 21/21 |

**⇒ `target_r=8` 的剪枝从未发生一次。秩恒等于 `init_r=16`。**

原因是纯算术的，与 4-bit、与重要性分数、与基座都无关：
剪枝窗口是 `[tinit, tfinal] = [20, 60]`，计划总长 `total_step=200`；实际训练在 `max_steps=21` 结束。**窗口只可能被推进 1 步（第 20→21 步），而 `deltaT=5` 意味着每 5 步才剪一次 —— 一次都不会触发。**

**训练健康度（`checkpoint-21/trainer_state.json` 的 `log_history`，step 20 处，四方法 × 三种子）**：

| 方法 | loss 起→终 | `mean_token_accuracy` 起→终 | `grad_norm` 起→终 |
|:--|:--|:--|:--|
| qlora | 2.005 → **1.481** | .5331 → **.6433** | 0.96 → 1.23 |
| dora | 2.008 → **1.490** | .5346 → **.6419** | 1.02 → 1.17 |
| adaq | 2.010 → **1.497** | .5348 → **.6423** | 1.32 → 1.55 |
| **adalora** | **3.096 → 2.937** | **.4990 → .4955** | **0.0884 → 0.0938** |

AdaLoRA 的初始 loss 高 1.1、终态 token accuracy 低 **14.5 个百分点**、`grad_norm` 小 **10–18×**，且 21 步内 token accuracy **不升反降**。**它没有在学习。** 这与「秩 16 + α 16 ⇒ α/r = 1.0（其余方法为 2.0）」「SVD 三元组 `lora_E/lora_A/lora_B` 参数化下梯度被 `orth_reg_weight=0.1` 的正交惩罚进一步压制」是一致的，但**无论机理是哪一条，结论都一样：这一行不是 AdaLoRA 的公平测量结果。**

**因此 `results_v2/FINAL_REPORT.md` 主表里的 `adalora .6538±.0012 / .4774±.0009` 应当被理解为：「一个参数量 2× 于对照、自适应机制从未运行、且在 21 步内未收敛的 SVD-LoRA 变体」的成绩，而不是 AdaLoRA 的成绩。**

### 2.3 由 2.2 直接推出：**Gap-3 的因果主张在本库中零证据**

- `docs/paper-framework.md:32-33` §2.2：「**Gap-3：AdaLoRA 在 4-bit 基座上重要性分数不可靠**」
- `docs/sota-comparison-protocol.md` §3 基线实现要点表 AdaLoRA 行：「全局预算对齐到 R=512」+ 易错点「**4-bit 基座上重要性分数不可靠正是 Gap-3，按官方实现跑，结果差是预期证据**」

两条都被 §2.2 的实测否证：

1. **「重要性分数不可靠」无从谈起——重要性分数从未参与任何决策。** `lora_E` 秩恒为 16，说明 AdaLoRA 的重要性打分即使算了，也没有影响过任何一个模块的秩。用一个机制未启动的 run 去论证「该机制在 4-bit 下不可靠」，是把 harness 缺陷当成了科学发现。
2. **「全局预算对齐到 R=512」是协议自己写下的实现要求，而它没被满足**：AdaLoRA 实跑 sum-of-ranks = 16×2×32 = **1024 = 2.000× R=512**，参数 6,816,768 = 2.000× QLoRA。**协议要求被违反，且没有任何检查发现它**（原因见 §2.7）。
3. **「结果差是预期证据」这句话本身就是评审要识别的「看似公平实不公平」设计的教科书样本**：协议在跑实验**之前**就把基线的失败预登记为支持自身 Gap 的证据。这使「公平对齐」的承诺在逻辑上被作废——**任何结果都已被预先解释**，基线臂失去了证伪能力。`docs/sota-comparison-protocol.md` §4 任务矩阵里「GSM8K | 4-shot | strict-match acc | 对数值精度最敏感，**AdaQ 优势主战场**」是同一模式的第二处实例。

> **协议可以预设假设，但不能预设证据的解释权。** 正确的写法是：先声明「若 AdaLoRA 在本预算下表现不佳，候选解释有 (a) 4-bit 基座下重要性分数失真、(b) 剪枝调度未适配短训练、(c) α/r 失配」，再用 artifact 区分三者。本项目跳过了区分步骤，而 artifact 明确指向 (b)。

### 2.4 超参对称性逐项核查

**对称的部分（实测通过，应保留并明写）**：

| 维度 | 证据 | 判定 |
|:--|:--|:--:|
| 学习率调度 | `outputs/adapters/*/checkpoint-21/trainer_state.json` 的 `log_history`：step10 lr = **1.3090e-4**、step20 lr = **4.8943e-6**，**四方法 × 三种子共 12 份逐位相同** | ✅ 对称 |
| 峰值 lr | 14B `checkpoint-255/trainer_state.json` step10 lr = **1.9999e-4** ≈ 2e-4 | ✅ 符合设定 |
| 量化配置 | `scripts/train.py:146-151` 单一 `BitsAndBytesConfig`，四方法共用 | ✅ 对称 |
| 数据 / epoch / 种子 | 12 份 `training_stats.json` 均 `dataset_size=2000, epochs=3`；`trainer_state.json` 均 `epoch=3.0, num_train_epochs=3`；种子 42/3407/2025 全覆盖 | ✅ 对称 |
| 优化器 / 精度 / 调度器 | `scripts/train.py:206-231` 单一 `SFTConfig`（`paged_adamw_8bit`, `bf16=True`, `cosine`, `warmup_ratio=0.03`, `max_grad_norm=1.0`），四方法共用 | ✅ 对称 |
| few-shot | `scripts/run_eval_unified.sh:78` `SETS=("mmlu\|5\|mmlu" "gsm8k\|4\|gsm8k")`；日志实测 `num_fewshot of mmlu_* from None to 5`（57 子任务全部）、`num_fewshot of gsm8k from 5 to 4` | ✅ 对称 |
| 样本量 | 主表/PTQ 表/14B 表全部 `mmlu sample_len=14042`、`gsm8k n-samples {original:1319, effective:1319}`；`run_eval_unified.sh:40` `LIMIT` 默认空 | ✅ 无 `--limit` 污染 |

**不对称的部分（三处，均为实测）**：

**(a) α/r 混杂 —— adaq 的有效学习率缩放与所有基线不同**

`outputs/adapters/adaq_seed{42,3407,2025}/adapter_config.json` 实测：`rank_pattern={"q_proj":12,"v_proj":4}`、**`alpha_pattern={}`（空）**、`lora_alpha=16`。

| 方法 | q_proj α/r | v_proj α/r |
|:--|--:|--:|
| qlora / dora | 16/8 = **2.000** | 16/8 = **2.000** |
| **adaq** | 16/12 = **1.333** | 16/4 = **4.000** |
| adalora | 16/16 = **1.000** | 16/16 = **1.000** |

LoRA 的前向缩放是 `(α/r)·BA`。所以 adaq 相对 qlora 在 q_proj 上把更新**缩小到 0.667×**、在 v_proj 上**放大到 2.000×**。**这不是秩分配的差异，这是一组未被声明的、逐模块类型的有效学习率差异**，它与待检验的处理（秩分配）完全混杂。

代码层面的成因明确：`scripts/train.py:87-89`
```python
if rank_scheme == "per_module_type":
    alpha_pattern = None   # 注释：严格复刻主实验早期行为：只用 rank_pattern，不写 alpha_pattern
```
而**同一文件 L103-104 的注释自陈**：「非均匀秩下 α 须随 r 缩放……与 qlora/dora 的 α=2r（**铁律2：公平先于领先**）保持一致」。`docs/sota-comparison-protocol.md` §2 统一超参表也写明「LoRA alpha **α=2r 所有方法统一**」。

**⇒ 线上 12 份 adaq adapter 的 `alpha_pattern={}` 证明主表走的是 `per_module_type` 分支，直接违反 train.py 自己写下的「铁律2」与协议 §2 的统一超参表。** `train.py:75-79` 的 `auto`/`per_layer` 分支本来会设 `alpha_pattern={k: 2*v}`（正确做法），但主表没用它。

旁证（实测权重幅度）：`outputs/adapters/qlora_seed42` 的 `max|w| = 0.1250`，`outputs/adapters/adaq_seed42` 的 `max|w| = 0.2471`，**1.98×**，与 v_proj 上 α/r 从 2.0 变 4.0 的 2× 放大高度吻合。

**(b) 序列长度 / 有效 batch 跨表不对称**

| 表 | 证据 | max_length | micro-bs | grad-accum | 有效 bs | `max_steps` | step/epoch | 总 tokens |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 8B 主表 | `checkpoint-21/trainer_state.json` `train_batch_size=4`；`num_tokens@step20 = 593,446` ⇒ 29,672 tok/step ÷ 16 seq = **1,854 tok/seq** | **2048**（+packing） | 4 | 4 | 16 | **21** | 6.76 | ≈623K |
| 8B PTQ/NF4 表 | 复用主表 adapter，同一批 | 2048 | 4 | 4 | 16 | 21 | 6.76 | ≈623K |
| **14B 表** | `scripts/run_b4_qwen14b.sh:35-44` 显式 `--max-length 512 --batch-size 2 --grad-accum 8`；`checkpoint-255/trainer_state.json` `train_batch_size=2`、step10@epoch 0.1185 ⇒ 84.4 step/epoch | **512** | 2 | 8 | 16 | **255** | 84.4 | ≈2.07M |

成因：`run_pipeline.py:131-134` 训练调用只传 `--budget 512 --base-rank 8 --epochs 3 --dataset-size 2000`，**不传 `--max-length/--batch-size/--grad-accum`**，于是 8B 落到 `scripts/train.py:122-124` 的 argparse 默认 `2048/4/4`；而 14B 走的是手写 shell，显式给了 `512/2/8`。

**⇒ 8B 与 14B 相差 12× 优化步数、3.3× 训练 tokens。两张表不可互相印证，也不可合并叙述。** 注意 `docs/sota-comparison-protocol.md` §2 的统一超参表写的是「序列长度 **2048**」「有效 batch 16 (micro-batch **4** × grad-accum **4**)」——**8B 与协议一致，是评审任务书里的「max-len 512/batch 2/grad-accum 8」描述对应的是 14B**。论文必须为两张表分别列出各自的超参表，不能共用一行。

**(c) 学习率扫描承诺未兑现（但方向上仍可辩护）**

`docs/sota-comparison-protocol.md` §2 写「学习率 扫描 {1e-4, 2e-4, 5e-4, 1e-3} 同一范围」。实测**所有 run 均为单一 lr=2e-4，无任何扫描产物**（无 `results*lr*` 目录）。

这一条**不构成不公平**——单一 lr 施加于全部方法，恰恰是对称的，且比「扫描后各取最优」更难被指控为选择性调参。**但它与协议文本不符，必须在论文中明写**：「所有方法共用 lr=2e-4，未做逐方法调参」，并把 `sota-comparison-protocol.md` §2 的扫描承诺删除或标注为未执行。否则一旦审稿人索取扫描数据，会发现协议与产物不一致，牵连其余所有对称性声明的可信度。

### 2.5 21 步：**主表的统计功效为零，四方法打平是必然结果而非发现**

8B 主表 `max_steps = 21`（`epoch 1.48 @ step10`、`epoch 2.96 @ step20` ⇒ 6.76 step/epoch × 3）。2000 条 OpenHermes 在 packing=True、max_length=2048 下被压成每 epoch 约 6.76 个序列包。

在 21 步、warmup_ratio=0.03（≈0.6 步）的设置下：
- 四方法 MMLU 落在 **.6533 / .6543 / .6538 / .6540**，极差 **0.0010**，而 `FINAL_REPORT.md` 自报的种子 std 为 .0006–.0015。**方法间差异 ≤ 种子内波动。**
- `FINAL_REPORT.md` §3 已正确指出：3 种子 Wilcoxon 最小双侧 p = 0.25，**不存在有效显著性检验**。
- 我补充一个更直接的量化：§3.3 将证明主表存在 **train-NF4 / eval-BF16 的基座错配**，其相对 Frobenius 误差实测为 **0.0926–0.1227**（`ranks.json` / `ranks_qwen14b.json` 的 `layer_stats.rel_error`）。而待测的处理效应（秩分配）在 MMLU 上是 0.001 量级、在参数预算上是 11.5%。**基座错配比处理效应大约高一个数量级**，主表的四路打平是这个装置的预期输出，不携带关于秩分配的信息。

**结论：8B 主表在当前形态下不能作为「方法有效性」或「方法无效性」的证据，只能作为「装置不足」的证据。** 修复项 F3（提高步数到 AdaLoRA 调度可运行的量级）与 F5（统一评测基座）是所有其他修复的前置。

### 2.6 14B：**「同 budget 512」被证伪，两臂都不在 512，且 QLoRA 多拿 45% 秩预算**

`outputs/adapters_qwen14b/*/adapter_config.json` 全 6 份实测（Qwen2.5-14B：48 层，`q_proj` 5120×5120，`v_proj` 5120×1024）：

| run | `rank_pattern` | **sum-of-ranks** | vs 协议 R=512 | **params** | vs qlora |
|:--|:--|--:|--:|--:|--:|
| `adaq_seed{42,3407,2025}` | `{q_proj:7, v_proj:4}` | **528** | **+16（+3.1%）** | **4,620,288** | **0.734×** |
| `qlora_seed{42,3407,2025}` | `{}`（r=8） | **768** | **+256（+50.0%）** | **6,291,456** | 1.000× |

手算复核：adaq/层 = 7×(5120+5120) + 4×(5120+1024) = 71,680 + 24,576 = 96,256 → ×48 = **4,620,288** ✓；qlora/层 = 8×10240 + 8×6144 = 81,920 + 49,152 = 131,072 → ×48 = **6,291,456** ✓。

三重问题：

1. **`scripts/run_b4_qwen14b.sh:2` 的注释「同预算 512 rank」是假的**，两臂分别是 528 与 768。
2. **`training_stats.json` 里 6 份 14B 记录全部写着 `"budget": 512`**，其中至少 qlora 的 3 份是 768。原因是 `--budget` 是个死参数（§2.7），标签从未被校验。
3. **`docs/sota-comparison-protocol.md` 的禁止事项明确写着「❌ r=64 基线 vs R=512 自研（8 倍预算差）」**——14B 上是 768 vs 528（1.45×），**方向与禁止事项相同（基线拿更多预算），只是幅度小**。协议自己的红线被自己的 14B 实验越过。

**对叙述的影响（必须精确改述）**：`docs/claim-revision-norm-aware.md` §4 给出的措辞上限是「AdaQ 以少 26.5% 的 adapter 参数取得与 QLoRA 持平的 MMLU」。**26.5% 是参数比（1 − 0.734），而秩预算比是 528/768 = 0.688，即少 31.2% 的 sum-of-ranks。** 两个数字都对，但含义不同，论文必须同时给出并说明 GQA 造成的差异（§2.1）。**更重要：这一比较不是预算对齐比较，因此不能用「同预算下 AdaQ 更省」的句式**；诚实的句式是「在 QLoRA 获得 1.45× 秩预算的条件下，AdaQ 以 0.734× 的参数取得统计上不可区分的 MMLU（.78429 vs .78372，差 +0.00057；seed2025 复现 +0.00036）」——这实际上是一个**参数效率**的观察，不是**性能优势**的观察。

### 2.7 根因：`--budget` 是死参数，预算守恒从未被机器校验

`scripts/train.py:117` 定义 `--budget`，全文件仅两处使用：
- `L241`：`print` 出来
- `L266`：写入 `training_stats.json`

**它从不进入 `build_lora_config`（L30-108）。** 所以：

- 18 份 `training_stats.json` 里的 `"budget": 512` 是**未经校验的标签**，其中 14B 的 6 份为假（实测 528 / 768）。
- 8B adaq 恰好在 512（`per_module_type_ranks` 的 implied sum = 4×32 + 12×32 = 512，delta 0）是**巧合而非保证**。
- AdaLoRA 的 1024 从未被任何断言拦下。
- 消融的方案A 臂 `equal_weight {q10,v4}` = **448**、`random {q7,v8}` = **480**（实测 `outputs/ablation_adapters_schemeA_*/`，三种子均如此），**同样不在 512**，而这两臂的结果被与 512 预算的臂放在同一张表里比较。

**这是本项目所有预算不对齐问题的单一根因，也是修复成本最低的一项**（见 F4）。

### 2.8 消融矩阵：六臂实为四配置，且**预算对齐的那一版结论更不利**

**方案A（塌缩，`outputs/ablation_adapters_schemeA_*`）实测 `rank_pattern`**：

| 策略 | `rank_pattern` | sum-of-ranks | 等价于 |
|:--|:--|--:|:--|
| `error_only` | `{q_proj:8, v_proj:8}` | 512 | **= QLoRA 主表配置** |
| `uniform` | `{q_proj:8, v_proj:8}` | 512 | **= QLoRA 主表配置（与上行完全相同）** |
| `importance_only` | `{q_proj:12, v_proj:4}` | 512 | **= adaq 主表配置** |
| `product` | `{q_proj:12, v_proj:4}` | 512 | **= adaq 主表配置（与上行完全相同）** |
| `equal_weight` | `{q_proj:10, v_proj:4}` | **448** | 独有，但**预算 −12.5%** |
| `random` | `{q_proj:7, v_proj:8}` | **480** | 独有，但**预算 −6.25%** |

**方案B（逐层，`outputs/ablation_adapters`）实测逐键比对**（64 个模块位点）：

| 配对 | 不同的层数 | sum-of-ranks |
|:--|--:|:--|
| **`error_only` vs `uniform`** | **0 / 64** | 512 vs 512 |
| **`importance_only` vs `product`** | **2 / 64** | 512 vs 512 |
| `equal_weight` vs `product` | 39 / 64 | 512 vs 512 |
| `random` vs `product` | 50 / 64 | 512 vs 512 |
| `error_only` vs `product` | 64 / 64 | 512 vs 512 |

各臂 distinct 秩值数：`error_only`=1、`uniform`=1、`importance_only`=6、`product`=6、`random`=11、`equal_weight`=14。**方案B 六臂预算全部守恒在 512** ✓（这是方案B 相对方案A 的一个实质优点，应在修复中保留）。

**⇒ 两个版本都塌缩为约四种有效配置。所谓「误差信号 vs 重要性信号 vs 乘积」的消融，在配置层面不存在：**
- `error_only` ≡ `uniform`：因为 `rel_error` 的层间 CV 只有 1.47%（§5.2），近乎常数 ⇒ 只按误差分配等价于均匀分配。**这个塌缩本身就是方法前提的反证，是有价值的发现**，但它意味着「error_only 臂」不提供任何关于误差信号的独立信息。
- `importance_only` ≡ `product`：因为 `Spearman(importance, score) = 0.9979`（§5.2），乘积的排序 99.8% 由范数决定 ⇒ 两臂分配几乎相同。**同理，「乘积优于单信号」这一命题在本装置下不可检验。**

**方案A s3407 实测 GSM8K strict-match（`results_v2_ablation_multiseed/summary.json`，`metric="exact_match,strict-match"`，`sample_len=1319`）**：

| 策略 | GSM8K strict | 备注 |
|:--|--:|:--|
| `product` | **0.42532221379833207** | 配置 q12/v4 |
| `equal_weight` | 0.41470811220621684 | 配置 q10/v4（预算 448） |
| `random` | 0.40561031084154664 | 配置 q7/v8（预算 480） |
| `error_only` | 0.3934799090219864 | 配置 q8/v8 |
| `uniform` | 0.38286580742987114 | 配置 q8/v8（**与上行同配置**） |
| `importance_only` | **0.3100833965125095** | 配置 q12/v4（**与 product 同配置**） |

**⇒ 同配置、同种子（3407）的两次独立训练，GSM8K strict 相差 `0.42532 − 0.31008 = 0.11524`，即 11.52 个百分点（152/1319 题）。** 这是本项目**目前唯一可用的、直接测得的跑间噪声底**。

对照 `docs/claim-revision-norm-aware.md` §5 采用的噪声底「同配置跑间噪声 **4.5 点**」——**实测值是 11.5 点，被低估 2.6×**。文档据此得出的「六策略 overall 极差 2.7 点 < 噪声底 4.5 点」结论方向正确（仍是阴性），但**余量被严重高估**：真实余量是 2.7 点 vs 11.5 点，即**信号比噪声小 4.3×**，阴性程度远比文档描述的更彻底。

同配置对 `error_only` vs `uniform`（均 q8/v8）差 `0.39348 − 0.38287 = 1.06` 点，量级小得多 ⇒ **噪声底在不同配置对上不均匀，单点估计不可用**；必须用 ≥3 组同配置重复对来估计（修复项 F8）。

**方案B（预算守恒）的实测结果反而更不利于本方法**（`results_v2_ablation/`，评测基座同主表为 BF16 + adapter）：

| 策略 | MMLU acc | GSM8K strict | GSM8K flexible |
|:--|--:|--:|--:|
| **`uniform`** | **0.6480558325024925（最高）** | 0.4450341167551175 | 0.5185746777862017 |
| `importance_only` | 0.6464891041162227 | 0.38741470811220624 | 0.5049279757391963 |
| `random` | 0.6464178891895741 | 0.41470811220621684 | 0.5125094768764216 |
| `equal_weight` | 0.6459905996296824 | 0.45564821834723274 | 0.5276724791508719 |
| `product` | 0.6458481697763852 | 0.3949962092494314 | 0.5026535253980288 |
| **`error_only`** | 0.6436405070502778 | **0.4746019711902957（最高）** | 0.5246398786959818 |

**在预算严格守恒（六臂均 512）的这一版里：MMLU 最高的是 `uniform`（= QLoRA 配置），GSM8K strict 最高的是 `error_only`（≡ `uniform`，秩图 0/64 层不同），而 `product`（本方法）在 strict 上排第 5/6，比 `uniform` 低 4.96 点、比 `error_only` 低 7.96 点。**

**⇒ 项目当前对外引用的消融数字来自方案A（预算不守恒、且 `equal_weight`/`random` 两臂被少给 12.5%/6.25% 预算的那一版），而没有引用预算守恒的方案B。方案B 的结论对方法更不利。这构成选择性报告的风险**，无论是否有意。修复项 F9 要求：以方案B 为主消融（预算守恒），方案A 降为附录并明确标注其预算破缺。

### 2.9 其余公平性瑕疵（实测，影响较小但需披露）

1. **Adapter 保存 dtype 不一致**：`outputs/adapters/*` 实测 `dtype={'BF16':128}`；`outputs/ablation_adapters*/*` 实测 `dtype={'F32':128}`。两者经 lm_eval 载入后均转 bf16，数值影响可忽略，但说明主表与消融走了**不同的保存代码路径**，应在 artifact 说明中披露。（附：全部四个受检 adapter `non_finite = 0`，无 NaN/Inf；`max|w|` 分别 qlora 0.1250 / adaq 0.2471 / product-方案A 0.1678 / product-方案B 0.1878。）
2. **8B 主表存在中断-resume 痕迹**：`adalora_seed42` 有 `checkpoint-2`（其余为 7/14/21），且 `save_total_limit=3` 却留存 4 个 checkpoint。`scripts/train.py:244-254` 会自动从最大 `checkpoint-*` 续训。**AdaLoRA 的 PEFT 内部 step 计数与 SVD 三元组状态在 resume 后是否恢复，本库无任何验证**。在 §2.2 已证明剪枝从未发生的前提下这不改变结论，但 resume 逻辑对 AdaLoRA 的安全性必须在修复后重新验证。
3. **14B seed42 的 `peak_vram_gb = 6.67`**，而同方法 seed2025/3407 为 17.6/17.62。14B 的 NF4 基座本身约需 9GB，**6.67GB 在物理上不可能覆盖一次完整的 14B 训练**。⇒ seed42 这一份 `training_stats.json` 的显存数字不可信（可能是 resume 后半程的峰值，或采集时机错误）。由于 `docs/paper-framework.md` §4.5 计划用显存做 Pareto 图，这一异常必须先解释。

---

## 3. 问题③：SOTA 复现性是否完全正确？

### 3.1 同一 harness ✅

全部表都走 `lm_eval --model hf`：
- 主表：`scripts/run_eval_unified.sh:104` `model_args="pretrained=${BASE_MODEL},peft=${adapter},dtype=bfloat16"`，`:121` `--num_fewshot "${shots}"`，`:38` `BASE_MODEL="${PROJECT_ROOT}/models/Meta-Llama-3-8B"`
- PTQ/NF4 表：`scripts/run_eval_ptq_nf4.py:139-149` `python -m lm_eval --model hf --tasks {t} --num_fewshot {shots} --batch_size auto --device cuda:0 --seed 42`
- 14B 表：`scripts/run_eval_b4_qwen14b.py:13` `SPECS=[("mmlu",5),("gsm8k",4)]`、`:26` model_args 含 nf4、`:28` `--seed 42`

`--seed 42`、`--batch_size auto`、离线环境变量（`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`）跨表一致。**这一项通过。**

### 3.2 few-shot 一致 ✅（但需补全量核查）

`results_v2/adaq_seed42/*.log` 与 `results_v2_ptq_nf4/adaq_nf4/*.log` 实测：`num_fewshot of mmlu_* from None to 5`（57 个子任务全部）、`num_fewshot of gsm8k from 5 to 4`、`truthfulqa_mc1/mc2` 与 `ifeval` 均 `has been set to 0 … Manual configuration will be ignored`。

**mmlu 5-shot / gsm8k 4-shot 在三张表间一致。** 局限：本次只逐字核查了 `adaq` 的两个目录，其余 22 个 run 的日志未逐条 grep（修复项 F10 要求全量 sweep 并写入 `_audit`）。

注：`config.num_fewshot` 在 `results_*.json` 顶层记录为 `None`——lm_eval 把 CLI override 存在 task config 内，不回填顶层字段。**因此聚合器无法从 JSON 校验 few-shot，只能靠日志**，这是一个可复现性缺口。

### 3.3 **致命：8B 主表是「NF4 训练 / BF16 评测」的口径错配，且两份协议文档互相矛盾，被推荐的那条路径在代码里不存在**

**实测事实链**：

1. **训练侧用 NF4**：`scripts/train.py:146-151` `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`，随后 `L160` `prepare_model_for_kbit_training`。四方法一致。
2. **评测侧用 BF16 干净基座**：`results_v2/**/results_*.json` **全部 12 个 run** 的 `config.model_args` 实测只有三项——`{'pretrained':'/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B', 'peft':'<adapter>', 'dtype':'bfloat16'}`。**无 `load_in_4bit`、无 `bnb_4bit_quant_type`、无 merge 路径。**
3. **这是有意为之，不是笔误**：`docs/eval-protocol-v2.md` §1 统一口径硬约束表明写「适配器加载 | **bf16 基座 + PEFT adapter 直接加载（不是 4-bit 基座）** | `run_eval_unified.sh`」，并附 ⚠️「适配器加载是 **bf16 基座**。若论文正文写『4-bit 合并后推理』，需另跑 `EVAL_MODE=merge` 一组……当前主表口径 = bf16 + adapter。」
4. **但另一份协议文档要求相反**：`docs/sota-comparison-protocol.md` §4 Merge 策略写「训练完即 merge（4-bit 反量化 → bf16）……**方式1: merge 后评测（推荐，稳定）**」，方式2 = `pretrained=<base>,peft=<adapter>,dtype=bfloat16`。
5. **实际走的是方式2**，而方式2 的文档字符串本身就漏了 `load_in_4bit`——所以「方式2」在文档里就是一个不完整/不正确的命令模板。
6. **`scripts/merge_adapter.py` 存在，但从未用于 `results_v2`**（`model_args` 中 `pretrained` 指向原始基座目录，非 merge 输出目录）。
7. **`EVAL_MODE=merge` 分支在 `scripts/run_eval_unified.sh` 中不存在**：grep `EVAL_MODE|load_in_4bit|merge` 于该脚本，只有 `L104` 那一行硬编码的 `model_args`，无任何条件分支。**`eval-protocol-v2.md` §1 承诺「需另跑 EVAL_MODE=merge 一组」，而该开关从未被实现。**

**为什么这是致命的（三条独立的理由）**：

- **(i) 它测量的是一个不存在的系统。** QLoRA 的语义是「在 NF4 量化基座上做低秩适配，推理时基座仍是 4-bit」。把 NF4 上训练出的 `BA` 加到干净 BF16 基座上，得到的既不是 QLoRA 的部署形态，也不是标准 BF16 LoRA（后者会全程在 BF16 基座上训练）。**它是一个第三种东西，文献中没有对应数值，因此主表无法与任何已发表的 QLoRA/DoRA/AdaLoRA 数字对照。**
- **(ii) 错配量级压过处理效应。** `ranks.json` 的 `layer_stats.rel_error` 实测 min **0.092642** / max **0.100753**（8B），`ranks_qwen14b.json` min **0.092305** / max **0.122659**（14B）。即训练基座与评测基座相差约 **9.3%–12.3% 的相对 Frobenius 误差**。而待测的秩分配效应：MMLU 上 0.001 量级，参数预算上 11.5%。（严格地说，把两者放在同一尺度需要 `‖BA‖_F / ‖W‖_F`，**该量在库中从未被测量** —— 这本身是修复项 F6 的内容。）以 `max|w|` 实测 0.125（qlora）估计，rank-8 更新的相对量级在 1e-2 附近，**即基座错配比处理效应约高一个数量级**（此为估计，非实测）。
- **(iii) 它与 PTQ 表不可比，而论文需要两表互证。** PTQ/NF4 表（`run_eval_ptq_nf4.py:110-122`）显式带 `load_in_4bit=True, nf4, double_quant`，**是正确的 NF4 口径**。于是同一个 `qlora_seed42` adapter 在两张表里的基座不同：主表 MMLU **.6533**（BF16 基座）、PTQ 表 **.6333855576128756**（NF4 基座），**相差 2.0 点**。这 2.0 点**纯粹来自基座口径**，与方法无关。**把这两张表放进同一篇论文而不显式隔离基座变量，读者会把它误读为方法效应或规模效应。**

**唯一的自洽表**：14B。`results_v2_qwen14b/*/results_*.json` 的 `config.model_args` 实测含 `load_in_4bit:true, bnb_4bit_quant_type:"nf4", double_quant:true, dtype:"bfloat16", peft=...`，训练侧 `run_b4_qwen14b.sh` 同样走 `train.py:146-151` 的 NF4。**14B 是全项目唯一 train/eval 口径一致的表**——这从侧面说明 8B 的 BF16 评测是路径依赖的遗留，而非经过论证的设计。

**修复方向（F5，二选一，不可含糊）**：
- **方案 5A（推荐，与 `sota-comparison-protocol.md` §4 方式1 一致）**：用 `scripts/merge_adapter.py` 把每个 adapter 反量化-合并进其**训练时的 NF4 基座**，产出 merged bf16 权重，评测 `pretrained=<merged_dir>,dtype=bfloat16`。这既保持推理确定性，又保留「基座是量化过的」这一 QLoRA 语义。**需在 `run_eval_unified.sh` 中真正实现 `EVAL_MODE=merge` 分支。**
- **方案 5B（成本更低）**：在 `run_eval_unified.sh:104` 的 `model_args` 中加入 `load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16`，与 `run_eval_ptq_nf4.py:110-122` 逐字对齐。
- **无论哪种，主表必须重跑，且必须新增 `baseline_bf16`（或 `base_nf4dq_noadapter`）锚点行**（F2）。
- **并在 `docs/eval-protocol-v2.md` §1 与 `docs/sota-comparison-protocol.md` §4 之间做一次性裁决**，删除被否决的那条，避免第三份文档继续引用矛盾口径。

### 3.4 跨表可比性总结

| 表 | 训练基座 | 评测基座 | max_len | 有效 bs | `max_steps` | n_seeds | 有锚点行 | 自洽 |
|:--|:--|:--|--:|--:|--:|--:|:--:|:--:|
| 8B 主表 `results_v2` | NF4+DQ | **BF16 干净** | 2048 | 16 | **21** | 3 | **无** | ❌ |
| 8B PTQ `results_v2_ptq_nf4` | （复用主表 adapter） | NF4+DQ | 2048 | 16 | 21 | **1** | 有（`baseline_nf4`） | ✅ |
| 8B 消融 `results_v2_ablation*` | NF4+DQ | **BF16 干净** | 2048 | 16 | 21 | 1–2 | 无 | ❌ |
| 14B `results_v2_qwen14b` | NF4+DQ | NF4+DQ | **512** | 16 | **255** | **1**（gsm8k）/ 2（mmlu） | **无** | ✅ 口径自洽，但预算与种子不齐 |

**⇒ 三张表三套协议。任何跨表的数值比较（例如「8B .654 → 14B .784 说明规模可扩展」）在当前 artifact 下都不成立。** `docs/claim-revision-norm-aware.md` §4 已正确禁止「具备规模可扩展性」的表述，应保留该禁令。

### 3.5 指标政策自相矛盾 = 事后切换指标（reviewer 会读作 cherry-picking）

三处互相冲突的政策声明，全部实测：

1. `docs/sota-comparison-protocol.md` §4 任务矩阵：「GSM8K | 4-shot | **strict-match acc** | 对数值精度最敏感，AdaQ 优势主战场」→ **strict 是预注册的主指标**
2. `scripts/run_eval_ptq_nf4.py:191`：gsm8k 取 `exact_match,strict-match` → **PTQ 表用 strict**
3. `results_v2/FINAL_REPORT.md` 主表 GSM8K 列 = strict → **8B 主表用 strict**
4. **`scripts/aggregate_b4_3seeds.py:100`**：预写结论「**3. strict-match 一律不作为判据**（Qwen-Instruct 格式不合规，strict 度量格式而非能力）」→ **14B 表改用 flexible**
5. `results_v2_qwen14b/summary.json` 的 `_meta.notes`：「strict-match 对 Qwen-Instruct 格式不友好，flexible-extract 更可信」

**⇒ 8B 用 strict，14B 宣布 strict 一律不可用并改报 flexible。而 14B 的头条结论（adaq flex .6588 vs qlora .5254，+13.3 点）恰好只在 flexible 下成立——strict 下是 .1827 vs .1304（+5.2 点，且两者绝对值都低到不可用）。**

这不必然是不诚实的（`_meta.notes` 已披露理由，`claim-revision-norm-aware.md` §4 也把它列为 Limitation），**但当前的组织方式会让审稿人自行得出「指标被事后选择」的结论**，因为：预注册文档（§4 任务矩阵）说 strict，聚合脚本（L100）说 strict 一律不用，两者都在库里，且**没有任何一份文档记录这个决定是何时、基于什么做出的**。

**另一处同源问题**：`docs/claim-revision-norm-aware.md` §1-E3 报消融用 **strict**（product .4230），§5 报同一批消融用 **flexible**（product .5129）。**同一实验、两种指标、两处数字，文中未标注口径差异**，读者会当成两套矛盾数据。

**修复（F7）**：在 `docs/eval-protocol-v2.md` 新增 §7「GSM8K 指标政策」，一次性写清：(a) 预注册主指标 = strict-match；(b) 同时无条件报告 flexible-extract；(c) 对 chat/instruct 基座（14B），strict 因格式不匹配而失去效度，**给出证据**（strict .1827/.1304 对官方 Qwen2.5-14B-Instruct GSM8K ≈0.94 的量级差），故 14B 以 flexible 为解释指标、strict 仍须列出；(d) 该决定在跑 14B 评测**之前**还是之后做出——如实写。所有表格两列并列，任何单列叙述都视为违规。

### 3.6 14B strict 口径失效 ⇒ flexible 的 +13.3 点很可能主要测格式而非数学能力

`results_v2_qwen14b/summary.json` 实测（seed42）：

| run | params | MMLU acc | GSM8K strict | strict stderr | GSM8K flexible | flex stderr |
|:--|--:|--:|--:|--:|--:|--:|
| adaq | 4,620,288 | 0.7842899871813132 | 0.18271417740712662 | .010644 | **0.6588324488248674** | .013059 |
| qlora | 6,291,456 | 0.7837202677681242 | 0.13040181956027294 | .009276 | **0.5253980288097043** | .013755 |

**Qwen2.5-14B-Instruct 的官方 GSM8K 约 0.94。这里 strict 只有 .18/.13。** 原因是 harness 用的是 **4-shot raw-completion**（`run_eval_b4_qwen14b.py:13` `SPECS=[("mmlu",5),("gsm8k",4)]`，无 chat template），而基座是 **Instruct** 模型——它被 RLHF 训练成期待 `<|im_start|>...<|im_end|>` 结构，收到裸 4-shot 续写提示时输出格式高度不稳定。

**⇒ 在这个口径下，strict↔flexible 的差（adaq .18→.66，qlora .13→.53）度量的主要是「答案是否被正则抽取到」，即输出格式，而不是数学能力。而 flexible 下 adaq 比 qlora 高 13.3 点，这个差同样可能主要由格式漂移差异驱动。**

支持这一解读的旁证：`docs/claim-revision-norm-aware.md` §4 自己写下「绝对水位偏低提示小样本 SFT 格式漂移与能力回退」。2000 条 `### Human: / ### Assistant:` 格式（`train.py:177-201`）的 SFT 施加在一个已经 chat-tuned 的 Instruct 模型上，**必然造成格式覆盖**，而两个臂被覆盖的程度不同就足以产生 13 点的 flexible 差。

**修复（F11）**：14B 的 GSM8K 必须补一组 **chat-template + 0-shot（或 5-shot chat）** 的评测（lm_eval 的 `gsm8k` 配 `hf` 模型加 `--apply_chat_template`，或 `minerva_math`/`mgsm` 的 chat 变体）。**在补上之前，14B 的 GSM8K 差值不得出现在 abstract / intro / conclusion 的任何位置**，只能作为附录中带上「口径失效」警示的描述性数字。

### 3.7 14B 的种子覆盖与产物完整性：**「3 seeds」未兑现**

- `results_v2_qwen14b/`：adaq + qlora，**mmlu + gsm8k 齐全**，seed42。
- `results_v2_qwen14b_seed2025/`：**只有 `.done_mmlu`，无 gsm8k**。实测 mmlu adaq `0.7840051274747187` / qlora `0.7836490528414756`（差 **+0.00036**，与 seed42 的 **+0.00057** 同为微量，方向一致 ⇒ **MMLU 的「持平」结论在两个种子上稳定**）。
- `results_v2_qwen14b_seed3407/`：**目录不存在**。
- `scripts/aggregate_b4_3seeds.py:6-8` 定义 `SEED_DIRS={42:..., 2025:..., 3407:...}`，`:10-11` 应输出 `results_v2_qwen14b/summary_3seeds.json` 与 `REPORT_14B.md`。**两者均未生成。**
- **adapter 侧三种子齐全**（§2.6 表中 `adaq_seed{42,3407,2025}` 与 `qlora_seed{42,3407,2025}` 六份 `adapter_config.json` 全部存在）⇒ **训练已完成，只缺评测**。
- 附带缺陷：`scripts/run_eval_b4_qwen14b.py:24` `adapter = os.path.join(ADAPTER_DIR, f"{run}_seed42")` **硬编码 seed42**，该脚本结构上无法评测其他种子（多种子走了另一个 `run_eval_b4_qwen14b_seeds.py`）。

**⇒ 14B GSM8K 的头条 +13.3 点建立在 n=1 上，而 §2.8 已实测同配置重跑噪声底达 11.5 点（8B 口径）。** 14B 的 255 步比 8B 的 21 步收敛得多，噪声底应更小，但**没有任何 14B 的同配置重复测量来估计它**。在拿到 seed3407 + seed2025 的 gsm8k 之前，这个 +13.3 点不具备可报告性。

**这是全项目修复性价比最高的一项**：adapter 已训好，只需跑 4 次 gsm8k 评测（2 方法 × 2 种子），即可把 14B 从 n=1 升到 n=3。见 F12。

### 3.8 溯源链断裂：无法证明训练基座与评测基座是同一份权重

- 全部 18 份 `adapter_config.json` 的 `base_model_name_or_path`：
  - 8B（12 份）= `/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B` —— **该路径在本机 NOT PRESENT**（实测）
  - 14B（6 份）= `/home/ubuntu/AdaQ-LoRA/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master`（注意 `models/models` 双层）
- 评测实际使用 `/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B`（`run_eval_unified.sh:38`；`run_eval_ptq_nf4.py` 的 `model_args` 同）。该目录实测：`config.json` = `architectures ['LlamaForCausalLM'], hidden 4096, layers 32, kv_heads 8, heads 32, torch_dtype bfloat16`，4 个 safetensors 分片（4,976,698,672 + 4,999,802,720 + 4,915,916,176 + 1,168,138,808 字节），含 `original/`、`README.md`(36,584B)、`USE_POLICY.md` ⇒ **是 Meta-Llama-3-8B base（非 Instruct）**，与主表的 5-shot MMLU / 4-shot GSM8K raw-completion 口径匹配 ✓。
- **全库检索 `sha256|checksum|md5|model_hash` 于 `scripts/`、`docs/`、`results_v2/*.json`、`results_v2/*.md`：唯一命中在 `docs/review/nf4-spectral.md`（同侪评审文本），项目自身的代码与产物中 0 命中。**

**⇒ 没有任何机器可读的证据把「训练时的基座」与「评测时的基座」绑定在一起。** adapter 里记的是一个不存在的路径。在 §3.3 已经证明基座口径是本项目最敏感变量的前提下，**基座身份的不可验证是一个 P0 级可复现性缺陷**——审稿人无法排除「训练用了另一份权重」的可能，作者也无法自证清白。

**修复（F13）**：新增 `scripts/record_provenance.py`，在每次训练与评测启动时把 (a) 基座目录每个 `*.safetensors` 分片的 sha256、(b) `config.json` 的 sha256、(c) `adapter_config.json` 的 sha256、(d) `git rev-parse HEAD` + `git status --porcelain` 的 sha256、(e) `pip freeze` 的 sha256，写入该 run 目录下的 `provenance.json`；`scripts/aggregate_results.py` 的 `_audit` 增加「所有 run 的基座 sha256 必须一致，否则 exit 2」。

---

## 4. 问题④：Q2 审稿人必索而缺失的基线

**全库目录扫描实测**：`ls -d results*` 得到 11 个目录（`results`, `results_v2`, `results_v2_ablation`, `results_v2_ablation_multiseed`, `results_v2_ablation_schemeA`, `..._s2025`, `..._s3407`, `results_v2_ppl`, `results_v2_ptq_nf4`, `results_v2_qwen14b`, `results_v2_qwen14b_seed2025`）。
`find . -maxdepth 2 -type d -name "*budget*" -o -name "*R128/256/768/1024*" -o -name "*loftq*" -o -name "*gptq*" -o -name "*awq*" -o -name "*hqq*" -o -name "*baseline_bf16*"` → **0 命中**。
`grep -rni "GPTQ|AWQ|HQQ|\bRTN\b|round-to-nearest|SpQR|SqueezeLLM|OmniQuant|QuaRot|LoftQ|QA-LoRA|LQ-LoRA|IR-QLoRA"` 于 `docs/ scripts/ *.md *.py` → **LoftQ/QA-LoRA/LQ-LoRA/IR-QLoRA/AQLM/QuaRot 仅出现在 `sota-comparison-protocol.md:15,22,32,33`、`paper-framework.md:18,32,33,114`、`paper-outline.md:28,38,39,40`、`theory-review-report.md:146,148` 的散文里，无任何对应实验产物。GPTQ / AWQ / HQQ / RTN / SpQR / SqueezeLLM / OmniQuant / QuaRot 在 `docs/`、`scripts/` 中 0 命中**（唯一命中在 `docs/review/*.md`，即同侪评审文本）。

按致命度排序：

### P0 —— 缺任何一个都足以被拒

| # | 缺失项 | 为什么致命 | 成本 |
|:--|:--|:--|:--|
| **M1** | **`baseline_bf16` / 无 adapter 锚点行（8B 主表）** | 主表四臂打平在 .6533–.6543，**没有锚点就无法区分「四方法同样好」与「四方法同样无用」**。NF4 表已显示四臂 MMLU 全部低于 `baseline_nf4`（.6334–.6357 vs .6364），BF16 侧极可能同向却被隐藏。审稿人第一个问题就是这一行。 | **极低**：1 次评测，无需训练 |
| **M2** | **LoftQ** | `paper-framework.md:32-33` 与 `sota-comparison-protocol.md:32` 把新颖性**明确押在与 LoftQ 的区分上**（「LoftQ 补偿初始化，我们分配容量」）。**这是一个针对具名方法的差异性主张，而该方法从未被运行。** 更糟：LoftQ 的初始化用的正是**量化误差的 SVD**——而本项目的核心实测结论是「量化误差项对 score 的排序贡献只有 0.144 的 Spearman」（§5.2）。**若 LoftQ 的误差驱动初始化同样无效，那本文的「诊断价值」也一起被削弱；若它有效，本文的方法主张被直接推翻。两种情况都必须跑。** | 中：PEFT 已内置 `LoftQConfig`，1 臂 × 3 种子 |
| **M3** | **参数匹配的 QLoRA 臂** | §2.1 实测 adaq = **1.115×** qlora 参数（3,801,088 vs 3,407,872），源于 GQA 下把秩从 `v_proj`(out=1024) 挪到 `q_proj`(out=4096)。**当前任何「adaq ≥ qlora」的结论都与「adaq 多 11.5% 参数」完全混杂。** 需补一臂参数量对齐到 ≈3.80M 的均匀 QLoRA（例如 `q_proj:v_proj = 6:10` 或 `r≈8.9` 的非整数解 → 用 `rank_pattern={q_proj:9,v_proj:9}` 得 (9×8192+9×5120)×32 = 3,833,856，误差 +0.9%）。 | 中：1 臂 × 3 种子 |
| **M4** | **实验三：秩预算-性能曲线（R ∈ {128,256,512,768,1024}）** | `paper-framework.md:73` 明写「**最小集 = 实验一 + 实验二消融1 + 实验三**」，§4.4 列为「秩预算分析」；`sota-comparison-protocol.md` §6 把它列为**必配图**。**这是项目自己声明的最小可发表集的一部分，而它 0 产物。** 更关键：在 §2.8 已证明「六策略塌缩为四配置、uniform 反而最好」之后，**R-曲线是唯一还能产出正面结论的实验**——若在某个 R 上非均匀分配确实优于均匀，那就是论文的主结果；若在所有 R 上都不优于，阴性结论就有了强得多的支撑（覆盖 5 个预算点而非 1 个）。**这是把当前困境转化为论文的最直接路径。** | 高：5 个 R × 2 策略（uniform vs product）× 3 种子 = 30 run |

### P1 —— 强烈建议，缺失会被点名但不必然致命

| # | 缺失项 | 理由 |
|:--|:--|:--|
| **M5** | **GPTQ（int4, g=128）** | 论文的核心概念是「量化误差」。GPTQ 是**显式做逐列误差补偿**（Hessian/Cholesky 更新）的方法，是「量化误差」这一概念在 PTQ 文献里的主要处理者。审稿人的必然提问：「既然误差可以被 GPTQ 直接补偿掉，为什么还要用秩分配去间接补偿？」**当前库里没有任何东西可以回答。** |
| **M6** | **AWQ（int4, activation-aware）** | 同 M5。AWQ 用**激活幅度**做逐通道缩放，这与本项目的 `importance = ‖W‖_F`（**权重**幅度）是不同的信号源。由于 §5.2 已证明本方法实质是「范数感知」，AWQ 是「激活感知」的最近邻，**两者的对比是本文定位所必需的**。 |
| **M7** | **真 INT4-RTN（均匀，g=128）** | §1.2 已论证：`baseline_nf4` 是 NF4+DQ 的 RTN，**不是文献语境的 RTN**。补一行均匀 INT4-RTN 成本极低（纯 PTQ，无训练），可以彻底堵住「你连 RTN 都没跑」这句最廉价的攻击，同时把 NF4 码本本身的效果隔离出来。 |
| **M8** | **AdaLoRA 重跑（修 `total_step` 与 `init_r`）** | §2.2 已证明当前 AdaLoRA 臂在 rank 16 = 2.000× 预算下、剪枝机制全程未启动、21 步未收敛。**这一行必须重跑或删除。** 重跑要求：`total_step = max_steps`（而非硬编码 200）、`tinit`/`tfinal` 按实际步数缩放（例如 `tinit = 0.1×max_steps`, `tfinal = 0.5×max_steps`）、并在 trainer callback 中断言 `lora_E` 秩在训练结束时确已降到 `target_r`。**若选择不重跑，则必须从主表删除 AdaLoRA 行，并撤回 Gap-3 的全部表述**（§2.3）——保留一个机制未启动的臂并据此论证「该机制在 4-bit 下不可靠」，是比缺失基线严重得多的问题。 |

### P2 —— 加分项，缺失不致命

| # | 缺失项 | 理由 |
|:--|:--|:--|
| M9 | HQQ | 免校准、半二次分裂，是「更简单的 training-free 4-bit」攻击面 |
| M10 | QA-LoRA | `paper-framework.md:33` 已划界（「缩放梯度 vs 分配秩」），划界主张应有实验支撑 |
| M11 | SpQR / OmniQuant / QuaRot | `sota-comparison-protocol.md:22` 自列的重点核查清单，目前状态全是「待 scholar 核查」 |
| M12 | DyLoRA / FLoRA | `paper-framework.md:34` §2.2 列为自适应秩 PEFT 同族，动态秩方法的直接对照 |
| M13 | 一个「按架构静态设定 q:v = 3:1」的平凡对照 | **这是 §5.2 的发现直接要求的**：既然 score 的排序 99.8% 由 `‖W‖_F` 决定，而 `‖W‖_F` 在 transformer 中主要由**模块形状**决定，那么「q_proj 给 3× 于 v_proj 的秩」这个规则**可以不读任何权重、只看 `config.json` 就写出来**。必须有一个这样的平凡臂，才能证明预计算流程（`adaq_precompute.py`，实测 <5 min）带来了任何超出「按形状手设」的价值。**若平凡臂与 adaq 打平，那么方法的实用价值主张就必须撤回，只保留诊断价值。** |

### 「缺哪个最致命」的直接回答

**按「审稿人一句话就能击落」的强度排序：**

1. **M1（`baseline_bf16` 锚点）** —— 最致命，因为**成本几乎为零却缺失**，且它的缺失让主表的四路打平变得不可解释。审稿人只需写「Table 2 lacks an unfinetuned reference; the authors cannot distinguish parity-at-high from parity-at-low」，整张主表作废。
2. **M2（LoftQ）** —— 次致命，因为它是**作者自己选定为新颖性参照物的方法**，且本项目的核心阴性发现（误差信号无信息）恰好落在 LoftQ 的作用机制上。**「see LoftQ」是比「see AQLoRA」更难防御的一句话**，因为 AQLoRA 只做层选择，LoftQ 直接消费同一个误差量。
3. **M4（R-曲线）** —— 第三，因为它是**项目自己声明的最小集成员**，且是当前唯一可能产出正面结果的实验。
4. **M3（参数匹配 QLoRA）** —— 第四，因为它使 8B 上任何 adaq vs qlora 的比较都带 +11.5% 参数的混杂。

**M5/M6（GPTQ/AWQ）虽然常被列为「必索」，但在本文的定位下略轻**：本文不是 PTQ 方法论文，主战场是 PEFT 秩分配。它们缺失会被点名，但可以用一段 scope 声明缓解（「本文不竞争 PTQ 压缩率，而是在给定 4-bit 基座下研究适配容量分配」）——**前提是这段声明必须写，而当前库里没有**。

---

## 5. 问题⑤：相关工作 delta（AQLoRA 2608.23816 / PARA 2604.27796）

### 5.1 落档状态：全库 0 引用（独立复核）

我的 grep（`docs/ scripts/ *.md *.py`，排除 `outputs/` 与 tokenizer 噪声）：`AQLoRA`、`PARA`、`2608.23816`、`2604.27796` → **在项目自身的文档与代码中 0 命中**；唯一命中在 `docs/review/jcr-gate.md`（同侪评审，其 `:30` 亦记录「全库检索……均 0 命中」，`:70` 记录「`sota-comparison-protocol.md` 四个 Gap 检索状态全部『待 scholar 核查』」）。

`docs/paper-framework.md:31-37` §2 Related Work 的完整清单：
- 2.1 量化微调：**QLoRA, LoftQ, QA-LoRA, LQ-LoRA**
- 2.2 自适应秩 PEFT：**AdaLoRA, DyLoRA, FLoRA**
- 2.3 低秩逼近与量化：理论交叉

`docs/sota-comparison-protocol.md:15` 的 Gap-2 状态列 = **「待 scholar 核查」**；`:22` 的重点核查清单 = LoftQ、QA-LoRA、LQ-LoRA、IR-QLoRA、AQLM 系、QuaRot 系 —— **不含 AQLoRA、不含 PARA**。

**⇒ 两篇最相关的近期工作既不在检索清单里，也不在 Related Work 里，也没有检索落档。而 `sota-comparison-protocol.md` §1 自己规定「每次正式写作/投稿前重查一次，附检索日期」——该规定未被执行。**

### 5.2 但在讨论 delta 之前：**论文的 intro 前提被项目自己的 `ranks.json` 证伪到小数点后四位**

`docs/paper-framework.md:12`（叙事主线，也是 §1 Introduction 的第一段落点）原文：

> 「现有 PEFT 方法（QLoRA, DoRA, AdaLoRA）在 4-bit 量化基座上微调时，对所有层分配均匀秩，**忽视了量化误差的层间异质性**——**NF4 对不同模块类型的量化误差 ε_l 可相差数倍**。这种异质性意味着：**高误差层需要更大的适配容量**来补偿量化偏差，而低误差层的秩是浪费的。」

我从 `ranks.json` / `ranks_qwen14b.json` 的 `layer_stats` **独立重算**（不依赖 `claim-revision-norm-aware.md` 的转述）：

| 量 | 8B（Llama-3-8B，64 位点） | 14B（Qwen2.5-14B，96 位点） |
|:--|--:|--:|
| `rel_error` min / max | 0.092642 / 0.100753 | 0.092305 / 0.122659 |
| **`rel_error` max/min** | **1.0875×** | **1.3289×** |
| `rel_error` 均值 / CV | 0.093608 / **1.47%** | 0.094280 / **3.98%** |
| **`q_proj` 均值 / `v_proj` 均值** | 0.093589 / 0.093627 = **0.9996** | 0.094266 / 0.094293 = **0.9997** |
| `q_proj` CV / `v_proj` CV | 1.54% / 1.40% | 5.29% / 1.92% |
| `importance`(=‖W‖_F) CV | **56.20%** | **39.04%** |
| **Spearman(`rel_error`, `score`)** | **+0.1440** | **−0.1967** |
| **Spearman(`importance`, `score`)** | **+0.9979** | **+0.9987** |
| Spearman(`rel_error`, `importance`) | +0.1056 | **−0.2194** |
| Spearman(`assigned_rank`, `score`) | +0.9171 | +0.9237 |
| **Spearman(`rel_error`, `assigned_rank`)** | **+0.1112** | **−0.2039** |
| 实际用到的秩取值集合（`r_min=4, r_max=128`） | **{4,5,6,11,12,13}** | **{4,5,6,7}** |

对 `paper-framework.md:12` 的逐句裁定：

1. **「NF4 对不同模块类型的量化误差 ε_l 可相差数倍」→ 假。**
   跨模块类型的比值实测 **q/v = 0.9996（8B）/ 0.9997（14B）**，即 **1.0000× 到小数点后三位**。全局 max/min 也只有 **1.0875×（8B）/ 1.3289×（14B）**。「数倍」需要 ≥2×，**实测差 1.5–18×**。
2. **「忽视了量化误差的层间异质性」→ 前提不成立。**
   `rel_error` 的层间 CV 只有 **1.47%**（8B）。**几乎不存在可被「忽视」的异质性。** 相比之下 `importance` 的 CV 是 **56.20%**，即 **38×** 于误差项的变异度。
3. **「高误差层需要更大的适配容量」→ 在本项目的分配器里未实现，14B 上反向实现。**
   `Spearman(rel_error, assigned_rank)` = **+0.1112（8B）/ −0.2039（14B）**。**在 14B 上，量化误差越大的模块拿到的秩越小。** 机理清楚：`Spearman(rel_error, importance) = −0.2194`（14B），误差与范数负相关，而 score 的排序 99.87% 由范数决定 ⇒ **分配器系统性地把容量从「按本文动机最该补偿」的模块上移走。**
4. **「用预计算的逐层 4-bit 量化误差 ‖E_l‖_F 驱动秩预算分配」→ 实际驱动量是 ‖W‖_F。**
   `Spearman(importance, score) = 0.9979 / 0.9987`，`Spearman(rel_error, score) = 0.1440 / −0.1967`。**score = rel_error × ‖W‖_F 的排序几乎完全由第二个因子决定。** `docs/claim-revision-norm-aware.md` E2 给出的判定「该方法在数学上等价于范数感知（importance-aware）秩分配，『量化误差感知』名不副实」**与我的独立重算完全一致**。
5. **附加发现（文档未记录）：`rel_error` 近常数是 NF4 的结构性后果，不是经验偶然。**
   NF4 用 **per-block(64) absmax 定标** ⇒ 相对误差对权重尺度**不变**；NF4 码本是**对标准正态分布的分位最优码本**（Dettmers et al. 2023）⇒ 只要块内分布接近高斯，相对误差就趋近一个由码本决定的常数。**Transformer 权重经 RMSNorm/LayerNorm 训练后块内分布确实接近高斯。** 因此「NF4 相对重构误差层间近常数」是**可从量化器设计推出的性质**，而「rel_error 驱动秩分配」是**先验上就不可能提供信息的**。
   **这是本项目最有价值的科学陈述，而且它比现有的阴性表述强得多**：现有文档说「我们测到 CV=1.47%，所以误差项没用」；正确的说法是「**任何以『NF4 逐块 absmax 定标后的相对重构误差存在层间异质性』为前提的方法，在该量化器的设计下先验失效；我们用 CV 1.47%/3.98% 与 Spearman 0.144/−0.197 给出了定量确认，并进一步表明乘积打分在此条件下数学退化为范数感知**」。

### 5.3 由 5.2 推出：**与 AQLoRA 的 delta 不是「我们解释为什么」，而是「我们的前提与他们同被证伪，我们给出定量机理 + 一个可操作的替代解释」**

AQLoRA (2608.23816) 的已知主张：**NF4 reconstruction error 用于层选择，效果 not better than random**。

对照本项目实测：

| 维度 | AQLoRA（据任务书所述主张） | 本项目实测 | delta 是否成立 |
|:--|:--|:--|:--:|
| 现象 | NF4 重构误差驱动的层**选择** ≈ 随机 | NF4 相对误差 CV 1.47%/3.98%（近常数）；`Spearman(rel_error, assigned_rank)` = +0.111/−0.204（近零 / 反向）；方案B 消融中 `product` 的 GSM8K strict 比 `uniform` 低 4.96 点、比 `error_only` 低 7.96 点 | **同向，本文更细** |
| 机理 | （据任务书，未给出定量分解） | `score = rel_error × ‖W‖_F` 的排序 **99.8% 由 ‖W‖_F 决定**（Spearman .9979/.9987 vs .1440/−.1967）；两信号近正交（ρ=.1056）甚至负相关（ρ=−.2194）；根因是 NF4 的 per-block absmax 定标 + 正态最优码本 ⇒ 相对误差尺度不变 | ✅ **这是真增量** |
| 任务对象 | 层**选择**（二元 keep/drop） | 预算**分配**（连续水位法 + 截断守恒，`allocate_ranks`；`per_layer_ranks` sum 实测 = 512 ✓） | ✅ 成立，但**实际训练用的是塌缩后的方案A**（§2.8），所以这个 delta 在 artifact 层没有体现 |
| 时序 | — | **事前零训练预计算**（`adaq_precompute.py`，`docs/paper-framework.md:52` 称 <5 min） | ✅ 成立 |
| 结论强度 | 「not better than random」 | 当前 artifact 只能支持「**no evidence of benefit；差异落在同配置重跑噪声底（实测 11.5 GSM8K strict 点）之内**」 | ⚠️ **本文更弱** |

**裁定**：delta **窄幅成立，但当前形态下不足以支撑一次投稿**，理由三条：

1. **零引用 = 零防御。** 审稿人写一句「the layer-selection signal derived from NF4 reconstruction error has already been shown to be no better than random (AQLoRA, arXiv:2608.23816); the authors do not cite or distinguish it」，新颖性主张即告击落。**这不是可以 rebuttal 阶段补救的缺陷，因为补救需要的是引用与定位改写，而这必须在投稿前完成。**
2. **本文的 intro 前提与 AQLoRA 的结论冲突，而本文自己的数据站在 AQLoRA 一边。** `paper-framework.md:12` 声称「ε_l 可相差数倍」「高误差层需要更大容量」；实测 q/v = 0.9996、`Spearman(rel_error, rank)` = +0.111/−0.204。**若照现稿投出，这是文实相反（desk-reject 级）；若引用 AQLoRA 而不改 intro，则是明知故犯。**
3. **本文的结论强度低于 AQLoRA。** AQLoRA 报「not better than random」，本文能报的上限是「no evidence of benefit, within noise」——因为 §2.8 的噪声底（11.5 点）大于所有被测效应（≤2.7 点），**本文没有能力做出「等于随机」这样的等价性断言**（等价性检验需要预设 margin 并证明置信区间落在 margin 内，当前 n=2 种子、无预设 margin，`FINAL_REPORT.md` §3 亦已承认 3 种子 Wilcoxon 最小 p=0.25）。**若论文写成「我们证明误差信号无效」，是过度宣称；只能写成「我们未能检出效应，且给出机理解释」。**

**可辩护的 delta 表述（建议直接采用）**：

> Prior work has reported that layer **selection** driven by NF4 reconstruction error performs no better than random (AQLoRA, arXiv:2608.23816). We do not merely replicate this observation for rank **allocation**; we identify its mechanism. Because NF4 applies per-block(64) absmax scaling and its codebook is quantile-optimal for a standard normal, the *relative* reconstruction error is scale-invariant and near-constant across layers: we measure a layer-wise coefficient of variation of **1.47%** (Llama-3-8B, 64 sites) and **3.98%** (Qwen2.5-14B, 96 sites), with a cross-module-type ratio of **0.9996** and **0.9997** respectively. Consequently, any scoring rule of the form `score_l = rel_error_l × ‖W_l‖_F` is dominated by its second factor: we measure `Spearman(importance, score) = 0.9979 / 0.9987` against `Spearman(rel_error, score) = 0.1440 / −0.1967`. **Such a rule is therefore norm-aware, not quantization-error-aware, and this is a consequence of the quantizer's design rather than an empirical accident.** On the 14B model the effect reverses sign (`Spearman(rel_error, assigned_rank) = −0.2039`), i.e. the allocator assigns *less* capacity to *more* quantization-sensitive modules — the opposite of the stated motivation. Empirically, across a budget-conserving six-arm ablation (all arms at sum-of-ranks = 512), uniform allocation attains the best MMLU (**0.6481**) and the error-only arm — whose rank map is *identical* to uniform (0/64 sites differ) — attains the best GSM8K strict (**0.4746**), while the proposed product score ranks 5th of 6 (**0.3950**). Two arms sharing an identical rank map but trained independently at the same seed differ by **11.5 GSM8K strict points**, which bounds the run-to-run noise floor above every effect we measure; we therefore claim **no evidence of benefit**, not equivalence.

**这段话的每一句都有本评审实测的数字支撑，且它把项目的困境转成了一个有内容的贡献。**

### 5.4 与 PARA (2604.27796) 的 delta：**成立，但同样零落档**

区分点是真实的且容易论证：
- **作用时序**：PARA 作用于**已训练**的 LoRA（post-hoc 秩重分配/裁剪）；本项目在训练**前**用 <5 min 的预计算决定秩（`docs/paper-framework.md:50-52` §3.4 `adaq_precompute.py`；`docs/sota-comparison-protocol.md` Gap 表述「冻结量化 + 预计算误差 + 零搜索零训练秩分配」）。
- **所需信息**：PARA 需要训练后的权重/梯度统计；本项目只需冻结的量化基座与 `‖W‖_F`，**不需要任何训练信号**。
- **对象**：PARA 重分配既有 adapter 的秩；本项目分配预算 R 到未训练的位点。

**但这个 delta 目前是「口头 delta」**：`PARA` 在全库 0 命中，`sota-comparison-protocol.md:22` 的重点核查清单不含它，四个 Gap 的检索状态全是「待 scholar 核查」。**按协议 §1 自己的规定（「每次正式写作/投稿前重查一次，附检索日期」），必须补一份可复核的检索档案**：日期、检索库（arXiv / OpenAlex / Semantic Scholar / Google Scholar）、检索式、命中数、纳入排除标准。

**并且要注意一个反向风险**：既然 §5.2 已证明本项目的 score 实质是 `‖W‖_F` 驱动，而 PARA 一类 post-hoc 方法通常用训练后的权重/激活统计做重分配，**一个诚实的审稿人会问：「既然事前只用 ‖W‖_F 就能得到你的分配，那 PARA 用更丰富的训练后信号做重分配，岂不是严格更强？」** 本项目当前的答案只能是「我们的贡献是零训练成本」——**这是一个成本论证，不是质量论证**，必须在论文中如实降格（`docs/paper-framework.md:51` 已写「Allocation Rule（启发式，不称『最优』）」，方向正确，应贯彻到 abstract 与 contribution 句）。

### 5.5 `docs/paper-framework.md:52` 命题3b 在库中无任何对应测量

§3.4 列出「**命题3b（代理质量，Spearman ρ ≥ 0.7）— 经验性**」。

**问题：ρ 是哪两个量的 Spearman？**
- 若指 `Spearman(score, rel_error × importance)` ⇒ 恒等于 1.0，是重言式，无信息。
- 若指 `Spearman(score, importance)` ⇒ 实测 0.9979 / 0.9987，**≥0.7 成立，但这恰好证明 score 退化为范数**（§5.2），把它当作「代理质量合格」的证据是自我否证的。
- 若指 `Spearman(score, 该层真实的下游敏感度)`（例如「给该层单独加秩后验证损失的下降量」）⇒ **这才是「代理质量」的正确定义，而该实验在库中完全不存在**（无 per-layer sensitivity sweep、无 leave-one-layer-out、无 R-曲线，见 §4 M4）。

**⇒ 命题3b 目前是一个无测量支撑的「经验性命题」。以现稿写入论文，会被要求出示 ρ 的测量方法与数值，而作者拿不出。** 修复：要么删除命题3b，要么把它重述为可测量形式并补做（最小可行版本：对 8B 的 64 个位点，各自单独把秩 +4 / −4，测 2000 条验证样本的 loss 变化，与 score 排序做 Spearman——**这也是 M4 R-曲线的一个廉价替代，且它直接回答「score 是不是好代理」这个全文核心问题**）。

### 5.6 Gap 声明的措辞风险

`docs/sota-comparison-protocol.md:19` 已自我约束「禁止『所有 PEFT 方法都未解决』」，`docs/theory-review-report.md:148` 与 `paper-framework.md:114` 也已给出 E_l 已知性边界的正确表述（「本文框架适用于量化冻结的 QLoRA 范式；当量化参数参与训练时（LoftQ [init], QA-LoRA [quant param opt]），E_l 变为可学习量，『已知偏移』假设不再成立」）。**这些自我约束是正确的，应全部保留并提升到正文。**

但 `paper-framework.md:18` 的 **Gap-2「量化误差补偿仅做初始化（LoftQ）或幅度缩放（QA-LoRA），未做秩分配」** 在 §5.2 的实测下变成一个尴尬的主张：**本文做了秩分配，而秩分配被本文自己的数据证明无收益；LoftQ/QA-LoRA 没做秩分配，而它们做的是直接消费/缩放误差量本身。** 「它们没做 X」只有在 X 有效时才构成 Gap。**建议把 Gap-2 从「未做秩分配」改写为「未检验秩分配这一路径是否有效——本文填补的是这个检验，结论是否定的」**，这与 §5.3 的建议表述一致，且诚实得多。

同理 **Gap-3（AdaLoRA 在 4-bit 基座上重要性分数不可靠）必须整条撤回或重写**，理由见 §2.2/§2.3：AdaLoRA 臂的剪枝从未启动，重要性分数从未影响任何秩，因此本库不存在任何关于「4-bit 基座上重要性分数可靠性」的证据。

---

## 6. 编号修复清单（路径 + 精确改法 + 验收信号）

**优先级：P0 = 不修不能投；P1 = 不修大概率被拒；P2 = 提升接收概率。**
**「验收信号」= 可机器检查的判据，不依赖人工判断。**

### P0（阻断项）

**F1 — 重命名 PTQ 基线，并明确 NF4+DQ 的身份**
- 路径：`scripts/run_eval_ptq_nf4.py:48-54`（`CONFIGS` 键名）、`results_v2_ptq_nf4/summary.json`（键名）、`docs/eval-protocol-v2.md` §1 表、论文 PTQ 表
- 改法：`baseline_nf4` → `base_nf4dq_noadapter`；表行名 → `Llama-3-8B, NF4+DQ (RTN, no calibration), no adapter`。**禁止在任何位置把这一行简称为「RTN」或「4-bit baseline」**。同时在 §1.2 的位置加一句：「bnb 的 4-bit 路径为逐块(64) absmax 定标的最近舍入，无校准、无误差补偿；NF4 码本对 N(0,1) 分位最优，故非均匀。」
- 验收：`grep -rn "baseline_nf4" scripts/ docs/ results_v2_ptq_nf4/` → 0 命中（除历史 artifact 归档目录）；论文全文 `grep -c "RTN"` 的每一处均带限定语。

**F2 — 补 8B 无-adapter 锚点行（两个口径各一行）**
- 路径：`scripts/run_eval_unified.sh`（新增 `METHODS` 取值 `none`）、`scripts/run_eval_ptq_nf4.py:48-54`
- 改法：
  - `run_eval_unified.sh:104` 增加分支：当 `adapter` 为空时 `model_args="pretrained=${BASE_MODEL},dtype=bfloat16"`，输出目录 `<OUT_DIR>/baseline_bf16/`。
  - `run_eval_ptq_nf4.py` 的 `CONFIGS` 已有 `baseline_nf4: None`，保留，改名为 F1 的新键。
  - 各跑 3 种子（评测种子；无训练故成本仅 3×2 次评测）。
- 验收：`results_v2/summary.json` 顶层键包含 `baseline_bf16`；`results_v2/_audit` 的 run 数从 12 增至 15；主表首行为锚点行。
- **附带必查**：三向同值 `0.41470811220621684`（= 547/1319）出现在 `results_v2_ptq_nf4/adaq_nf4`(flex)、`results_v2_ablation_schemeA_s3407/equal_weight`(strict)、`results_v2_ablation/random`(strict)。对这三份产物跑 `python scripts/verify_eval_artifact.py <run_dir> <marker>`，并确认三者的 per-sample 预测不是同一份文件（若 lm_eval 未落 per-sample，则重跑一次并开启 `--log_samples`）。

**F3 — 把 8B 训练步数提到基线可正常运行的量级（AdaLoRA 的前置条件）**
- 路径：`run_pipeline.py:131-134`、`scripts/train.py:30-53`（AdaLoRA 调度）、`scripts/train.py:122-124`（默认超参）
- 改法：`--dataset-size 2000` 在 packing+max_length 2048 下只产生 6.76 step/epoch（实测），3 epoch = **21 步**。三个可选路径，任选其一但必须显式记录：
  - (a) 关闭 packing（`--no-packing`，`train.py:132`），2000 条 → 每 epoch 2000/(4×4)=125 步 → 3 epoch = **375 步**；
  - (b) 保持 packing，`--dataset-size` 提到 20000 → **210 步**；
  - (c) 保持 packing，`--max-length 512`（与 14B 对齐）→ 每 epoch ≈27 步 → 3 epoch = **81 步**（仍偏少，不推荐单独使用）。
  - **推荐 (a)**：它同时消除「1854 tokens/seq 的打包使『3 epoch』的实际数据暴露量不透明」这一问题。
- 验收：所有 `outputs/adapters/*/checkpoint-*/trainer_state.json` 的 `max_steps ≥ 200`；`num_tokens` 与 `dataset_size × epochs × 平均样本长度` 的量级一致（需在 `training_stats.json` 中记录该核算）。

**F4 — 让 `--budget` 成为真正的约束（单一根因修复，成本最低收益最高）**
- 路径：`scripts/train.py:117`（定义）、`:30-108`（`build_lora_config`）、`:241`（print）、`:266`（写 `training_stats.json`）
- 改法：
  1. 在 `build_lora_config` 返回前，对**每个方法**计算实际 sum-of-ranks 与实际参数数：
     - LoRA/DoRA：`sum_ranks = Σ_{m∈target_modules} r_m × n_layers`
     - AdaQ：`Σ rank_pattern.values() × n_layers`
     - AdaLoRA：`init_r × len(target_modules) × n_layers`（**训练开始时的真实占用**）与 `target_r × …`（剪枝完成后的目标占用），**两个都必须记录**
  2. `assert abs(sum_ranks - args.budget) <= tolerance`，`tolerance` 默认 0，**失败即 `sys.exit(2)`，不允许静默继续**。
  3. `training_stats.json` 写入**实测值**而非入参：`{"budget_requested": 512, "sum_ranks_actual": 528, "params_actual": 4620288, "budget_ok": false}`。
  4. 新增 `scripts/verify_budget.py <outputs_dir>`，遍历所有 `adapter_config.json` + safetensors 头，输出一张 `method × seed → (sum_ranks, params, ok)` 表，任一 `ok=false` 则 exit 2；把它挂进 `scripts/aggregate_results.py` 的 `_audit`。
- 验收：`python scripts/verify_budget.py outputs/adapters` exit 0（8B 修复后）；`outputs/adapters_qwen14b` 在 F14 修复前 exit 2 且报告 `qlora sum_ranks=768 (requested 512)`、`adaq sum_ranks=528 (requested 512)`；18 份 `training_stats.json` 均含 `sum_ranks_actual` 与 `params_actual` 字段。

**F5 — 统一训练/评测基座口径（二选一，并在两份协议文档间做一次性裁决）**
- 路径：`scripts/run_eval_unified.sh:104`、`scripts/merge_adapter.py`、`docs/eval-protocol-v2.md` §1 表 + ⚠️ 段、`docs/sota-comparison-protocol.md` §4 Merge 策略
- 改法：
  - **5A（推荐）**：在 `run_eval_unified.sh` 中**真正实现** `EVAL_MODE=merge` 分支——调用 `scripts/merge_adapter.py` 把 adapter 合并进其**训练时的 NF4 基座**（4-bit 反量化 → bf16 → `merge_and_unload` → `save_pretrained`），然后 `model_args="pretrained=<merged_dir>,dtype=bfloat16"`。merged 目录需按 `(method, seed)` 命名并记录 sha256。
  - **5B（低成本）**：`run_eval_unified.sh:104` 改为
    `model_args="pretrained=${BASE_MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16,peft=${adapter},dtype=bfloat16"`，
    与 `scripts/run_eval_ptq_nf4.py:110-122` **逐字对齐**。
  - **文档裁决**：`docs/eval-protocol-v2.md` §1 的「bf16 基座 + PEFT adapter 直接加载（**不是** 4-bit 基座）」与 `docs/sota-comparison-protocol.md` §4 的「方式1: merge 后评测（**推荐，稳定**）」**互相矛盾，必须删掉被否决的那一条并注明裁决日期与理由**。当前状态是：文档推荐方式1，代码只实现了方式2，且方式2 的文档模板本身就漏了 `load_in_4bit`。
- 验收：`grep -n "load_in_4bit\|EVAL_MODE" scripts/run_eval_unified.sh` 至少 1 处非注释命中；**全部 12（或 15，含 F2）份 `results_v2/**/results_*.json` 的 `config.model_args` 含 `load_in_4bit` 或指向 merged 目录**；两份协议文档中关于评测基座的表述 `grep` 后完全一致。

**F6 — 在主表报告实测参数数，并测量 `‖BA‖_F/‖W‖_F`**
- 路径：`docs/sota-comparison-protocol.md` §6 主表模板（`Params (M)` 列现全为 `2.7`）、`scripts/aggregate_results.py`、新增 `scripts/measure_update_norm.py`
- 改法：
  1. §6 模板的 `Params (M)` 列改为由 `verify_budget.py`（F4）**自动生成**，禁止手填。实测应填：qlora **3.41** / dora **3.57** / adaq **3.80** / adalora **6.82**（8B），adaq **4.62** / qlora **6.29**（14B）。
  2. 新增一列 `Params / QLoRA`：1.000 / 1.048 / **1.115** / **2.000**。**这一列必须出现在论文主表里**，否则读者无法看到 adaq 的 +11.5%。
  3. `scripts/measure_update_norm.py`：对每个 adapter，逐层计算 `‖B_l A_l‖_F / ‖W_l‖_F`（`W_l` 取评测基座的反量化权重），输出 mean/median/max，写入该 run 的 `training_stats.json`。**这是量化「基座错配（rel_error 0.093–0.123）相对处理效应大多少」的唯一途径**，也是 §2.5(ii) 与 §3.3(ii) 从「估计」升级为「实测」的必要步骤。
- 验收：主表含 `Params (M)` 与 `Params / QLoRA` 两列且数值与 safetensors 头一致（可用 F4 的 `verify_budget.py` 交叉校验）；每个 run 有 `update_rel_norm_mean` 字段；论文中出现一句「基座错配的相对 Frobenius 误差为 0.093–0.123，而 LoRA 更新的 `‖BA‖_F/‖W‖_F` 实测为 X，两者之比 ≈ Y」。

**F7 — 统一并预注册 GSM8K 指标政策**
- 路径：新增 `docs/eval-protocol-v2.md` §7、`scripts/aggregate_b4_3seeds.py:100`、`scripts/run_eval_ptq_nf4.py:191`、`results_v2/FINAL_REPORT.md`、`docs/claim-revision-norm-aware.md` §1-E3 与 §5、`docs/sota-comparison-protocol.md` §4 任务矩阵
- 改法：
  1. §7 写明：主指标 = `exact_match,strict-match`；**无条件同时报告** `exact_match,flexible-extract`；任何表格两列并列。
  2. 对 chat/instruct 基座，声明 strict 失效**并附证据**：14B strict = .1827/.1304 对官方 Qwen2.5-14B-Instruct GSM8K ≈0.94；故 14B 以 flexible 为解释指标，strict 仍须列出。
  3. **删除 `aggregate_b4_3seeds.py:100` 的预写结论**，改为在 §7 中论证；聚合脚本只输出数据，不输出裁定。
  4. **删除 `docs/sota-comparison-protocol.md` §4 的「AdaQ 优势主战场」与 §3 的「结果差是预期证据」**（见 F16）。
  5. `docs/claim-revision-norm-aware.md` §1-E3（strict）与 §5（flexible）**在每处数字后标注口径**，例如 `product .4230 [strict]` / `product .5129 [flexible]`。
- 验收：`grep -rn "优势主战场\|预期证据\|一律不作为判据" docs/ scripts/` → 0 命中；所有含 GSM8K 数字的 `.md` 文件中，每个数字后 20 字符内出现 `strict` 或 `flexible` 标注。

**F8 — 用同配置重复对实测噪声底，替换掉 4.5 点的估计**
- 路径：新增 `configs/noise_floor/`、`scripts/run_noise_floor.sh`、`docs/claim-revision-norm-aware.md` §5
- 改法：选 **≥3 组**同配置对（建议：`{q12,v4}`、`{q8,v8}`、逐层 product 方案B），每组**独立训练 ≥3 次**（不同 `--seed`，且额外做一组**同 seed 重跑**以分离「种子间」与「同种子非确定性」两个分量）。报告：每组的 GSM8K strict/flexible 与 MMLU 的 std、极差、以及合并的 pooled sd。
  - **已有实测可作为起点**：`outputs/ablation_adapters_schemeA_s3407/{product,importance_only}` 均为 `{q_proj:12,v_proj:4}`、同 seed 3407、字节级不同（128/128 张量 sha256 全不同），GSM8K strict = **0.42532221379833207** vs **0.3100833965125095**，**Δ = 11.52 点（152/1319 题）**。`{error_only,uniform}` 同为 `{q8,v8}`、同 seed，Δ = **1.06 点**。**两组的 Δ 相差 10.9×，说明单点噪声底估计不可用，必须多点估计。**
  - **同时必须排查 Δ=11.52 点的成因**：同配置同 seed 出现如此大的差异，除了 GPU 非确定性（`lora_dropout=0.05` + cuBLAS 非确定性归约 + packing 顺序）之外，还应排除 (a) 两次训练的 `--seed` 实际不同、(b) 其中一次发生了 `train.py:244-254` 的自动 resume、(c) 评测侧的 `--batch_size auto` 在不同显存占用下选择了不同 batch 从而改变 padding 与数值。**在成因查清之前，11.5 点只能作为「噪声底的上界证据」引用，不能作为「噪声底的估计值」。**
- 验收：`configs/noise_floor/` 下有 ≥3 组配置；`results_noise_floor/summary.json` 含每组的 `{n_runs, std_strict, std_flex, std_mmlu, range_strict}` 与 `{same_seed_pairs, cross_seed_pairs}` 的分解；`docs/claim-revision-norm-aware.md` §5 的「4.5 点」被替换为实测值并注明估计方法。

**F9 — 以预算守恒的方案B 为主消融，方案A 降为附录并标注预算破缺**
- 路径：`docs/claim-revision-norm-aware.md` §1-E3 与 §5、`docs/paper-framework.md` §4.3、`results_v2_ablation_multiseed/summary.json`、新增 `results_v2_ablation/REPORT.md`
- 改法：
  1. **主消融用 `results_v2_ablation/`（方案B，逐层，六臂 sum-of-ranks 实测全部 = 512）**。其结论（实测）：MMLU 最高 = `uniform` **0.6480558325024925**；GSM8K strict 最高 = `error_only` **0.4746019711902957**；`product` = MMLU **0.6458481697763852** / strict **0.3949962092494314** / flex **0.5026535253980288**，**strict 排第 5/6，比 uniform 低 4.96 点、比 error_only 低 7.96 点**。
  2. **方案A（`results_v2_ablation_schemeA_*`）移入附录**，并在表注中明写：`equal_weight` sum-of-ranks = **448**（−12.5%）、`random` = **480**（−6.25%）、其余四臂 = 512；且 `error_only` ≡ `uniform`（`rank_pattern` 均 `{q_proj:8,v_proj:8}`）、`importance_only` ≡ `product`（均 `{q_proj:12,v_proj:4}`），**六臂实为四配置**。
  3. **消融表必须新增两列：`rank_pattern`（或「有效秩图摘要」）与 `sum-of-ranks`**，让塌缩与预算破缺在表内可见。当前所有消融表都只列策略名，读者无从发现 `error_only` 与 `uniform` 是同一个实验。
  4. 明确写出：**`error_only` ≡ `uniform` 与 `importance_only` ≡ `product` 的塌缩不是 bug，而是本文核心发现的直接后果**（`rel_error` CV 1.47% ⇒ 误差项无区分力；`Spearman(importance,score)=0.9979` ⇒ 乘积由范数决定）。**把它作为证据呈现，而不是作为实验缺陷隐藏。**
- 验收：主消融表的每一行都有 `sum-of-ranks` 列且全部 = 512；附录表有预算破缺标注；`grep -n "error_only" results_v2_ablation*/REPORT*.md` 的每处都伴随「秩图与 uniform 相同（0/64 位点不同）」的说明。

**F10 — 全量 few-shot / 样本量核查并写入 `_audit`**
- 路径：`scripts/aggregate_results.py`、`scripts/verify_eval_artifact.py`
- 改法：本评审只逐字核查了 `results_v2/adaq_seed42` 与 `results_v2_ptq_nf4/adaq_nf4` 两个目录的日志（结果：`num_fewshot of mmlu_* from None to 5` 覆盖 57 子任务、`num_fewshot of gsm8k from 5 to 4`、truthfulqa/ifeval = 0 ✓）。需扩展为**全部 run 的机读核查**：
  1. 解析每个 `results_*.json` 的 `configs[task].num_fewshot`（**注意：顶层 `config.num_fewshot` 实测为 `None`，lm_eval 把 CLI override 存在 task config 内，聚合器当前无法从顶层字段校验——这是可复现性缺口**）。
  2. 断言 `mmlu*` 全部 = 5、`gsm8k` = 4、`truthfulqa*`/`ifeval` = 0；断言 `sample_len`：mmlu = **14042**、gsm8k = **1319**。
  3. 结果写入 `summary.json` 的 `_audit.fewshot_ok` / `_audit.sample_len_ok`，任一 false 则 exit 2。
- 验收：`results_v2/summary.json._audit` 与 `results_v2_ptq_nf4/summary.json._audit`、`results_v2_qwen14b/summary.json._audit` 三处均含 `fewshot_ok: true` 与 `sample_len_ok: true`；`python scripts/aggregate_results.py --results-dir results_v2` exit 0。
- **附带**：`scripts/run_eval_ptq_nf4.py:188` 的聚合守卫 `if "mmlu" in res and len(res) > 10` **无法检测 `--limit` 污染**（`--limit 25` 时 mmlu 子任务仍有 57 个）。改为显式断言 `sample_len == 14042`。

**F11 — 14B 补 chat-template GSM8K；在补齐前冻结该结论**
- 路径：`scripts/run_eval_b4_qwen14b.py:13`（`SPECS`）、`:24`（硬编码 seed42）、新增 `scripts/run_eval_b4_qwen14b_chat.py`
- 改法：
  1. 新增一组评测：`lm_eval --model hf --apply_chat_template --tasks gsm8k --num_fewshot 5`（或改用 `mgsm`/`minerva_math` 的 chat 变体），NF4 口径与现有一致。
  2. `:24` 的 `f"{run}_seed42"` 改为 `f"{run}_seed{args.seed}"`，与 `run_eval_b4_qwen14b_seeds.py` 合并，消除双脚本。
  3. **在 (1) 完成之前**：14B 的 GSM8K 差值（flex .6588 vs .5254）**不得出现在 abstract / introduction / conclusion / 图表标题**，只能进附录并带警示：「4-shot raw-completion 口径对 Instruct 基座失效（strict .1827/.1304，官方 ≈0.94）；flexible 差值可能主要由答案格式漂移驱动」。
- 验收：`results_v2_qwen14b_chat/summary.json` 存在且含 strict + flexible 两列；`grep -rn "0.6588\|6588\|13.3" docs/*.md` 的每处命中都在附录段落内或已带 chat-口径数值并列。

**F12 — 补齐 14B 三种子 GSM8K（性价比最高的单项修复）**
- 路径：`scripts/run_eval_b4_qwen14b_seeds.py`、`scripts/aggregate_b4_3seeds.py:6-11`
- 改法：**adapter 已全部训好**（实测 `outputs/adapters_qwen14b/{adaq,qlora}_seed{42,3407,2025}/adapter_config.json` 六份齐全）。只需：
  1. 对 seed2025 补跑 gsm8k（`results_v2_qwen14b_seed2025/` 目前**只有 `.done_mmlu`**）；
  2. 对 seed3407 补跑 mmlu + gsm8k（`results_v2_qwen14b_seed3407/` **目录不存在**）；
  3. 跑 `scripts/aggregate_b4_3seeds.py` 生成 `results_v2_qwen14b/summary_3seeds.json` 与 `REPORT_14B.md`（**两者当前均未生成**）。
- 验收：三个 seed 目录各含 `.done_mmlu` 与 `.done_gsm8k`；`summary_3seeds.json` 中每方法每任务 `n_seeds == 3`；`REPORT_14B.md` 存在。
- **成本估算**：4 次评测 run（2 方法 × 2 缺失任务组合）。**这是把 14B 头条从 n=1 升到 n=3 的最短路径，应最先做。**

**F13 — 建立基座与代码的溯源链**
- 路径：新增 `scripts/record_provenance.py`；改 `scripts/train.py`（启动处）、`scripts/run_eval_unified.sh`、`scripts/run_eval_ptq_nf4.py`、`scripts/aggregate_results.py`（`_audit`）
- 改法：每次训练/评测启动时写 `<run_dir>/provenance.json`：
  ```json
  {"base_model_dir": "...", "base_shards_sha256": {"model-00001-of-00004.safetensors": "..."},
   "base_config_sha256": "...", "adapter_config_sha256": "...",
   "git_head": "...", "git_dirty_files_sha256": "...", "pip_freeze_sha256": "...",
   "torch": "...", "transformers": "...", "peft": "...", "lm_eval": "...", "cuda": "...", "gpu": "..."}
  ```
  聚合器 `_audit` 增加：**所有 run 的 `base_shards_sha256` 必须一致，否则 exit 2**；并增加「训练 run 的基座 sha256 必须与其评测 run 的一致」。
- 动机（实测）：全部 18 份 `adapter_config.json` 的 `base_model_name_or_path` 指向 `/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B`（8B）—— **该路径在本机不存在**；评测实际用 `/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B`。**全库检索 `sha256|checksum|md5|model_hash` 于 `scripts/`、`docs/`、`results_v2/` 中 0 命中**（唯一命中在同侪评审文本 `docs/review/nf4-spectral.md`）。即**当前没有任何机器可读证据把训练基座与评测基座绑定**。在 F5 已证明基座口径是本项目最敏感变量之后，这是 P0。
- 验收：每个 run 目录有 `provenance.json`；`aggregate_results.py` 的 `_audit.base_sha_consistent == true`；`adapter_config.json` 的 `base_model_name_or_path` 与实际评测路径一致（或 `provenance.json` 中显式记录两者的映射与各自 sha256）。

**F14 — 修复 14B 的预算不对齐（528 vs 768）**
- 路径：`scripts/run_b4_qwen14b.sh:2`（注释）、`:35-44`、`outputs/adapters_qwen14b/*/training_stats.json`、`docs/claim-revision-norm-aware.md` §4
- 改法：实测（Qwen2.5-14B，48 层，`q_proj` 5120×5120，`v_proj` 5120×1024）：
  | run | `rank_pattern` | sum-of-ranks | params |
  |:--|:--|--:|--:|
  | adaq | `{q_proj:7, v_proj:4}` | **528** (+16) | 4,620,288 |
  | qlora | `{}` (r=8) | **768** (+256) | 6,291,456 |
  两条路径任选，但**必须选一条并删掉假注释**：
  - (a) **重训 qlora 到 sum-of-ranks ≈ 528**：48 层 × 2 模块 × r = 528 ⇒ r = 5.5，非整数；用 `rank_pattern={q_proj:6, v_proj:5}` → (6+5)×48 = **528** ✓，params = 48×(6×10240 + 5×6144) = 48×(61,440+30,720) = **4,423,680**。
  - (b) **重训 adaq 到 sum-of-ranks = 768**：`{q_proj:11, v_proj:5}` → (11+5)×48 = 768 ✓。
  - (c) **不重训，改述**：删除 `run_b4_qwen14b.sh:2` 的「同预算 512 rank」注释，删除 6 份 `training_stats.json` 中的 `"budget": 512`（或改为 `budget_requested` + `sum_ranks_actual`，由 F4 自动生成），并把 `docs/claim-revision-norm-aware.md` §4 的措辞改为：「**在 QLoRA 获得 1.45× 秩预算（768 vs 528）的条件下**，AdaQ 以 0.734× 的参数（4.62M vs 6.29M）取得统计上不可区分的 MMLU（.78429 vs .78372，Δ=+0.00057；seed2025 复现 Δ=+0.00036）」，**并明确声明这不是预算对齐比较**。
  - **注意**：路径 (c) 成本为零且诚实，但会使 14B 表失去「同预算」这一卖点；路径 (a)/(b) 各需 3 次训练（adapter 目录已有三种子，但只有 seed42 的评测齐全）。**推荐 (c) 立即执行 + (a) 排入 P1。**
- 验收：`python scripts/verify_budget.py outputs/adapters_qwen14b` exit 0；`grep -n "同预算 512" scripts/run_b4_qwen14b.sh` → 0 命中；6 份 `training_stats.json` 的 `sum_ranks_actual` 为 528/768（路径 c）或全部 528（路径 a）。

**F15 — 修复 α/r 混杂**
- 路径：`scripts/train.py:87-89`（`per_module_type` 分支的 `alpha_pattern = None`）、`:75-79`（`per_layer` 分支已有正确实现 `alpha_pattern={k: 2*v}`）、`:103-104`（自陈的「铁律2」注释）
- 改法：把 `per_module_type` 分支改为与 `per_layer` 一致：
  ```python
  if rank_scheme == "per_module_type":
      alpha_pattern = {k: 2 * v for k, v in type_ranks.items()}   # 保持 α/r = 2，与 qlora/dora 一致
  ```
  删除 `:87-89` 的「严格复刻主实验早期行为」注释——**复刻一个违反协议 §2 的早期行为不是正当理由**。
  然后**重训 adaq 的全部 12 个 adapter**（8B 3 种子 + 14B 3 种子 + 消融各臂）。
- 动机（实测）：12 份 8B adaq `adapter_config.json` 与 6 份 14B 均为 `alpha_pattern={}` + `lora_alpha=16`。故 adaq 的 `α/r` = **1.333（q_proj）/ 4.000（v_proj）**，而 qlora/dora 为 **2.000/2.000**，adalora 为 **1.000/1.000**。LoRA 前向缩放为 `(α/r)·BA`，**这是一组未声明的、逐模块类型的有效学习率差异，与待检验的处理（秩分配）完全混杂**。旁证：`max|w|` 实测 qlora **0.1250** vs adaq **0.2471**（**1.98×**），与 v_proj 上 α/r 由 2.0→4.0 的 2× 放大吻合。
  协议依据：`docs/sota-comparison-protocol.md` §2 统一超参表「LoRA alpha **α=2r 所有方法统一**」；`scripts/train.py:103-104` 注释自陈「铁律2：公平先于领先」。
- 验收：重训后所有 adaq `adapter_config.json` 的 `alpha_pattern` = `{"q_proj": 24, "v_proj": 8}`（8B）/ `{"q_proj": 14, "v_proj": 8}`（14B），即每个模块 `α_l/r_l = 2`；`scripts/verify_budget.py` 增加一项 `alpha_ratio_ok` 断言（所有方法、所有模块 `lora_alpha_effective / r == 2.0`），exit 0。

**F16 — 删除协议文档中的预设立场表述**
- 路径：`docs/sota-comparison-protocol.md` §3 基线实现要点表 AdaLoRA 行、§4 任务矩阵 GSM8K 行；`docs/paper-framework.md:18`（Gap-2）、`:33`（Gap-3）、`:12`（叙事主线）；`docs/paper-outline.md:28`
- 改法：
  1. §3 AdaLoRA 行的「**按官方实现跑，结果差是预期证据**」→ 删除。替换为：「AdaLoRA 的剪枝调度必须适配实际训练步数（`total_step = max_steps`，`tinit ≈ 0.1×max_steps`，`tfinal ≈ 0.5×max_steps`）；训练结束时必须断言 `lora_E` 秩已降至 `target_r`，否则该臂无效、不得进入对比表。」
  2. §4 GSM8K 行的「**AdaQ 优势主战场**」→ 删除，改为中性描述「对数值精度敏感」。
  3. `paper-framework.md:33` **Gap-3「AdaLoRA 在 4-bit 基座上重要性分数不可靠」→ 整条撤回**。理由（实测）：三个种子的全部 checkpoint（seed42: 2/7/14/21；seed3407、seed2025: 7/14/21）的 `lora_E` 秩**恒为 16**、`total_elems` **恒为 6,816,768**，`tinit=20`/`tfinal=60`/`total_step=200` 的窗口在 `max_steps=21` 下最多推进 1 步而 `deltaT=5` ⇒ **剪枝一次都未触发，重要性分数从未影响任何秩**。当前 AdaLoRA 臂的实测状态是「2.000× 参数预算 + 机制未启动 + 未收敛」（loss 3.096→2.937、token_acc .4990→.4955、grad_norm 0.0884→0.0938，对照 qlora 2.005→1.481 / .5331→.6433 / 0.96→1.23）。**据此论证「4-bit 基座上重要性分数不可靠」是把 harness 缺陷当成科学发现。**
  4. `paper-framework.md:18` Gap-2「量化误差补偿仅做初始化（LoftQ）或幅度缩放（QA-LoRA），未做秩分配」→ 改为「**未检验秩分配这一路径是否有效；本文填补该检验，结论是否定的**」。理由：§5.2 实测表明本文的秩分配无收益，「它们没做 X」只在 X 有效时构成 Gap。
  5. `paper-framework.md:12` 叙事主线中的「**NF4 对不同模块类型的量化误差 ε_l 可相差数倍**」→ **必须删除或改为实测值**。实测：跨模块类型比值 **q/v = 0.9996（8B）/ 0.9997（14B）**；全局 **max/min = 1.0875×（8B）/ 1.3289×（14B）**；层间 CV **1.47% / 3.98%**。「数倍」需 ≥2×，**实测差 1.5–18×**。**照现稿投出属文实相反。**
  6. 同段的「**高误差层需要更大的适配容量**」→ 必须加限定或删除。实测 `Spearman(rel_error, assigned_rank)` = **+0.1112（8B）/ −0.2039（14B）**，即**在 14B 上分配器把更少的秩给了量化误差更大的模块**，与该句方向相反。
- 验收：`grep -rn "预期证据\|优势主战场\|可相差数倍\|重要性分数不可靠" docs/` → 0 命中；`docs/paper-framework.md:12` 段落中每个数值主张都能在 `ranks.json` / `ranks_qwen14b.json` 的 `layer_stats` 中找到对应实测值。

**F17 — 补做 M1/M2/M3/M4（§4 的四个 P0 缺失基线）**
- 路径：新增 `scripts/run_baselines_p0.sh`；`docs/paper-framework.md` §4.2/§4.4；`docs/sota-comparison-protocol.md` §6 主表模板
- 改法与最小规模：
  - **M1 `baseline_bf16`**：见 F2，1 臂 × 3 评测种子，**无需训练**。
  - **M2 LoftQ**：`peft` 已内置 `LoftQConfig`。配置须与 qlora 逐行对齐（`r=8`, `lora_alpha=16`, `target_modules=[q_proj,v_proj]`, `lora_dropout=0.05`, `loftq_bits=4`, `loftq_iter=1`），1 臂 × 3 训练种子。**并在 `docs/paper-framework.md` §2.1 的「LoftQ 补偿初始化，我们分配容量」旁加上实测对比结果**——这句划界主张目前无任何实验支撑。
  - **M3 参数匹配 QLoRA**：`rank_pattern={q_proj:9, v_proj:9}` → (9×8192 + 9×5120) × 32 = **3,833,856**，对 adaq 的 3,801,088 误差 **+0.86%**；sum-of-ranks = (9+9)×32 = **576**（**须在表注中说明：参数对齐必然打破秩对齐，两个口径不可同时满足，这正是 §2.1 GQA 放大效应的直接后果**）。1 臂 × 3 种子。**或者反向做**：把 adaq 的 params 降到 3,407,872 附近（`{q_proj:10,v_proj:5}` → (10×8192+5×5120)×32 = 3,440,640，+0.96%；sum-of-ranks = 480）。**推荐同时报告两个口径各一臂**，因为「秩对齐」与「参数对齐」在 GQA 下不等价这一事实本身就是值得写进论文的方法学贡献。
  - **M4 R-曲线**：R ∈ {128, 256, 512, 768, 1024} × {uniform, product} × 3 种子 = 30 run。**这是 `paper-framework.md:73` 自己声明的「最小集」成员（实验三），且是 §2.8 之后唯一可能产出正面结果的实验。** 若成本不可承受，**最低可接受版本**是 R ∈ {256, 512, 1024} × {uniform, product} × 2 种子 = 12 run。
- 验收：`results_v2/summary.json` 含 `baseline_bf16`、`loftq`、`qlora_parammatched` 三个新键，各 `n_seeds == 3`；`results_v2_budget/summary.json` 存在且含 5（或 3）个 R × 2 策略；`docs/paper-framework.md` §4.4 的状态从「未跑」改为实测数值。

**F18 — 补做 M5/M6/M7（PTQ 参照系）+ 一段 scope 声明**
- 路径：新增 `scripts/run_ptq_reference.py`；`docs/paper-framework.md` §2.1；新增 `docs/eval-protocol-v2.md` §8「Scope 声明」
- 改法：
  - **M7 真 INT4-RTN（g=128）**：纯 PTQ，无需训练，成本最低，**建议最先补**。可用 `optimum-quanto` / `auto-gptq` 的 RTN 模式，或 lm_eval 的 `hf` + `quanto` 后端。
  - **M5 GPTQ（int4, g=128）** 与 **M6 AWQ（int4）**：用 `auto-gptq` / `autoawq`，校准集用与训练同源的 `teknium/OpenHermes-2.5[:512]`（**必须披露校准集，且不得与评测集重叠**）。各 1 组，无种子重复（PTQ 的随机性远低于训练）。
  - **Scope 声明**（`docs/eval-protocol-v2.md` §8）：明确写出「本文不竞争 PTQ 压缩率；GPTQ/AWQ 行的作用是界定『量化误差可被直接补偿到什么程度』，从而回答『为什么还需要秩分配』这一问题。**若 GPTQ/AWQ 已把误差降到使秩分配无意义的水平，这本身就是本文的结论之一。**」——**当前库里没有任何一段这样的 scope 声明，而 §4 已实测 GPTQ/AWQ/HQQ/RTN 在 `docs/`、`scripts/` 中 0 命中。**
- 验收：`results_v2_ptq_ref/summary.json` 含 `rtn_int4_g128`、`gptq_int4_g128`、`awq_int4` 三行，各含 mmlu + gsm8k（strict 与 flexible 双列，依 F7）；`docs/eval-protocol-v2.md` §8 存在；论文 PTQ 表包含这三行 + `base_nf4dq_noadapter`。

**F19 — 补相关工作落档与 delta 段**
- 路径：新增 `docs/related-work-search-log.md`；`docs/paper-framework.md` §2（新增 2.4）；`docs/paper-outline.md:28,38-40`；`docs/sota-comparison-protocol.md` §1 的 Gap 状态列与 `:22` 的重点核查清单
- 改法：
  1. **检索落档**（协议 §1 自己规定「每次正式写作/投稿前重查一次，附检索日期」，**当前四个 Gap 的状态列全部是「待 scholar 核查」，`:22` 的重点核查清单不含 AQLoRA 与 PARA**）：新建 `docs/related-work-search-log.md`，逐条记录 **日期 / 检索库（arXiv, OpenAlex, Semantic Scholar, Google Scholar）/ 检索式 / 命中数 / 纳入排除标准 / 每篇的一句话定位**。必须覆盖：AQLoRA (2608.23816)、PARA (2604.27796)、LoftQ、QA-LoRA、LQ-LoRA、IR-QLoRA、AQLM、QuaRot、SpQR、SqueezeLLM、OmniQuant、HQQ、GPTQ、AWQ、AdaLoRA、DyLoRA、FLoRA、OWL、erank。
  2. **`paper-framework.md` §2 新增 2.4「量化误差信号的可用性」**：引用 AQLoRA，并**直接采用 §5.3 给出的 delta 表述**（该段每句均有本评审实测数字支撑）。
  3. **PARA delta**：时序（post-hoc vs 事前零训练）、所需信息（训练后统计 vs 仅冻结基座的 `‖W‖_F`）、对象（重分配既有 adapter vs 分配预算到未训练位点）。**同时如实降格**：由于 §5.2 证明本文 score 实质由 `‖W‖_F` 驱动，本文相对 PARA 的优势是**成本**（零训练）而非**质量**；`paper-framework.md:51` 已写「Allocation Rule（启发式，不称『最优』）」，该降格必须贯彻到 abstract 与 contribution 句。
  4. **`sota-comparison-protocol.md` §1 的四个 Gap 状态列**：从「待 scholar 核查」改为具体检索结果 + 日期；Gap-3 依 F16(3) 撤回。
- 验收：`docs/related-work-search-log.md` 存在且含 ≥18 条目、每条有日期与库名；`grep -rn "AQLoRA\|2608.23816" docs/paper-framework.md docs/paper-outline.md` ≥1 命中（当前为 **0**）；`grep -rn "PARA\|2604.27796" docs/paper-framework.md` ≥1 命中（当前为 **0**）；§1 Gap 状态列无「待 scholar 核查」残留。

**F20 — 处理 `docs/paper-framework.md:52` 的命题3b（代理质量 Spearman ρ ≥ 0.7）**
- 路径：`docs/paper-framework.md:52`、`docs/theory-review-report.md`、新增 `scripts/measure_proxy_quality.py`
- 改法：命题3b 目前**在库中无任何对应测量**，且「ρ 是哪两个量的 Spearman」在文档中未定义。三种可能中两种不可用：
  - `ρ(score, rel_error × importance)` ⇒ 恒为 1.0，重言式；
  - `ρ(score, importance)` ⇒ 实测 **0.9979 / 0.9987**，≥0.7 成立，**但这恰好是 score 退化为范数的证据**（§5.2），把它当「代理质量合格」是自我否证；
  - `ρ(score, 该层真实下游敏感度)` ⇒ **这才是正确定义，而该实验不存在**。
  两条路径任选：
  - **(a) 删除命题3b**，并在 §3.4 注明「代理质量未经下游验证，本文只主张 score 的可解释分解（§5.2），不主张其为敏感度的好代理」。
  - **(b) 补做最小可行版本**：对 8B 的 64 个位点，各自单独把秩 +4 / −4（其余保持 `per_layer_ranks`），在 2000 条验证样本上测 loss 变化 Δℓ_l，然后报 `Spearman(score, −Δℓ)`。**这同时是 M4 R-曲线的廉价替代，且直接回答全文核心问题「score 是不是好代理」。**
- 验收：`docs/paper-framework.md:52` 或已删除，或其后紧跟实测 ρ 值与测量方法引用；若选 (b)，`results_proxy_quality/summary.json` 含 64 个位点的 `{score, delta_loss_plus4, delta_loss_minus4}` 与 `spearman` 字段。

### P1（强烈建议）

**F21 — 14B 的 `peak_vram_gb = 6.67` 异常**
- 路径：`outputs/adapters_qwen14b/{adaq,qlora}_seed42/training_stats.json`
- 实测：seed42 = **6.67 GB**，而 seed2025/3407 = **17.6 / 17.62 GB**。14B 的 NF4 基座本身约需 9 GB，**6.67 GB 物理上不可能覆盖一次完整的 14B 训练**。可能成因：`train.py:244-254` 的自动 resume 使峰值只统计了后半程；或采集时机在 `torch.cuda.reset_peak_memory_stats()` 之后。
- 改法：在 `train.py` 中把峰值采集移到 `trainer.train()` **之前** `reset_peak_memory_stats()`、**之后**立即 `max_memory_allocated()`，并把「是否为 resume」写入 `training_stats.json`。seed42 这一份数字在修好前**不得进入任何显存图表**（`paper-framework.md` §4.5 计划用显存做 Pareto 散点）。
- 验收：`grep -l "6.67" outputs/adapters_qwen14b/*/training_stats.json` → 0 命中，或该值旁有 `resumed: true` 标注。

**F22 — 显存不能作为预算对齐的证据**
- 路径：`docs/paper-framework.md` §4.5、`results_v2/system_metrics.json`
- 实测：adalora 参数 **2.000×** 于 qlora（6,816,768 vs 3,407,872），而 `peak_vram_gb` 仅 **14.84 vs 14.80（+0.27%）**，因为峰值被 8B 冻结基座 + 激活主导。
- 改法：在 §4.5 明确写「显存-性能散点图用于展示**部署成本**，**不用于论证预算对齐**；预算对齐以 `Params` 与 `sum-of-ranks` 两列为准（F6/F4）」。当前四方法显存跨度仅 0.21 GB（14.80–15.01），该图会近乎一条水平线，不能作为 Pareto 论证。
- 验收：`docs/paper-framework.md` §4.5 含上述限定句。

**F23 — 14B 只跑了 2/4 个方法**
- 路径：`outputs/adapters_qwen14b/`（实测只有 `adaq_*` 与 `qlora_*`，各 3 种子；**无 dora、无 adalora**）、`scripts/run_b4_qwen14b.sh:52-53`（只跑 seed42 两方法）
- 改法：14B 表若要与 8B 主表并列，必须补齐 DoRA（AdaLoRA 依 F16(3)/M8 处理）。**若选择不补，则 14B 表必须降格为「规模可行性预实验（pilot）」，标题与表注均需注明只含 2 个方法，且不得与 8B 主表合并叙述**（§3.4 已证明两表协议不同：2048/21 步 vs 512/255 步）。
- 验收：`outputs/adapters_qwen14b/` 含 ≥3 方法 × 3 种子，或 14B 表的标题/表注含「pilot, 2 methods only」。

**F24 — `run_pipeline.py` 与 `run_b4_qwen14b.sh` 的超参分叉**
- 路径：`run_pipeline.py:131-134`、`scripts/run_b4_qwen14b.sh:35-44`、`scripts/train.py:122-124`
- 实测：`run_pipeline.py` 只传 `--budget 512 --base-rank 8 --epochs 3 --dataset-size 2000`，**不传 `--max-length/--batch-size/--grad-accum`** ⇒ 8B 落到 argparse 默认 `2048/4/4`；14B 走手写 shell 显式给 `512/2/8`。这就是 §2.4(b) 中 8B 21 步 vs 14B 255 步的成因。
- 改法：把训练超参**全部收进 `configs/training_params.yaml`**（该文件已存在，实测在 `configs/` 下），两个入口都从它读取，禁止任何一处依赖 argparse 默认值。并在 `training_stats.json` 中回写**实际生效**的全部超参（而非入参）。
- 验收：`grep -n "max_length\|batch_size\|grad_accum" run_pipeline.py scripts/run_b4_qwen14b.sh` 的每处命中都指向 `configs/training_params.yaml`；18 份 `training_stats.json` 均含 `max_length`/`micro_batch`/`grad_accum`/`effective_batch`/`packing` 五个字段且与 `trainer_state.json` 的 `train_batch_size` 自洽。

**F25 — `total_step` 必须等于真实 `max_steps`**
- 路径：`scripts/train.py:30-53`（`AdaLoraConfig(..., total_step=200)` 硬编码）
- 改法：`total_step` 不能在建模时硬编码——它在 `build_lora_config` 阶段还不知道。两步走：
  1. 先做一次 `trainer = SFTTrainer(...); max_steps = trainer.create_optimizer_and_scheduler(num_training_steps=-1)` 或直接用 `len(train_dataset) / (bs × ga × epochs)` 的 dry-run 计算；
  2. 用算出的值构造 `AdaLoraConfig(total_step=max_steps, tinit=int(0.1*max_steps), tfinal=int(0.5*max_steps), deltaT=max(1, int(0.02*max_steps)))`。
  3. 注册一个 `TrainerCallback`，在 `on_train_end` 断言 `model.peft_config['default'].target_r` 已达到（即 `lora_E` 秩的众数 == `target_r`），否则 `raise`。
- 验收：重训后 `outputs/adapters/adalora_seed42/checkpoint-<max_steps>/adapter_model.safetensors` 的 `lora_E` 秩集合 == `{8}`（而非当前的 `{16}`）；`total_elems` ≈ 3,407,872（而非 6,816,768）；callback 的断言日志存在于 `training_stats.json`。

### P2（提升接收概率）

**F26 — 补一个「按架构静态设定 q:v」的平凡对照（M13）**
- 理由（实测支撑）：`Spearman(importance, score) = 0.9979/0.9987`，而 `importance = ‖W‖_F` 在 transformer 中主要由**模块形状**决定（`q_proj` 4096×4096 vs `v_proj` 4096×1024）。实测 `rel_error` 的 q/v 均值比 = **0.9996/0.9997**，即误差项在模块类型间也毫无区分力。**所以「q_proj 给 ~1.5–3× 于 v_proj 的秩」这个规则可以只读 `config.json` 就写出来，无需任何量化测量。**
- 改法：新增一臂 `static_shape`，`rank_pattern` 直接由架构推出（8B: `{q_proj:12, v_proj:4}`，与 adaq 相同；14B: `{q_proj:7, v_proj:4}`，同）。**若它与 adaq 打平（在 F8 的噪声底内），则 `adaq_precompute.py` 的整个预计算流程（实测 <5 min）不带来任何超出「按形状手设」的价值，方法的实用主张必须撤回，只保留 §5.2 的诊断价值。**
- 验收：`results_v2_ablation/static_shape/` 存在；主消融表含该行；论文中有一句明确交代「本文的分配规则在实践中等价于一条可由架构直接写出的静态规则」。

**F27 — adapter 保存 dtype 统一**
- 实测：`outputs/adapters/*` = `BF16`，`outputs/ablation_adapters*/*` = `F32`（两者 `non_finite = 0`，`max|w|` 分别 0.1250/0.2471 与 0.1678/0.1878）。
- 改法：统一为 `BF16`（`save_pretrained(..., safe_serialization=True)` 前确保 `model.to(torch.bfloat16)`，或在 `SFTConfig` 中固定 `save_dtype`）。当前差异说明主表与消融走了不同保存路径。
- 验收：`find outputs -name adapter_model.safetensors | xargs -I{} python -c "..."` 报出的 dtype 集合 == `{BF16}`。

**F28 — 全量核查 8B 主表是否存在 resume 痕迹**
- 实测：`adalora_seed42` 有 `checkpoint-2`（其余为 7/14/21），且 `save_total_limit=3` 却留存 4 个 checkpoint ⇒ 存在中断-续训。`train.py:244-254` 会自动从最大 `checkpoint-*` 续训。
- 改法：在 `training_stats.json` 记录 `resumed_from: <checkpoint 或 null>` 与 `n_restarts`；对 12 个 8B run + 6 个 14B run 全量核查并列表。AdaLoRA 的 PEFT 内部 step 计数与 SVD 三元组状态在 resume 后是否恢复，**库中无任何验证**，必须在 F25 修复后重新验证。
- 验收：18 份 `training_stats.json` 均含 `resumed_from` 字段。

**F29 — 聚合器增加「同配置重复」检测**
- 理由：§2.8 的六臂塌缩为四配置，**在 12 份 `adapter_config.json` 里是完全可见的**，却没有任何工具发现它；§2.1 的 adalora 2.000× 预算，**在 `results_v2/system_metrics.json` 里已经写着 `adapter_params: 6816768`**，也没有任何工具发现它。**项目已经有数据，缺的是断言。**
- 改法：`scripts/aggregate_results.py` 的 `_audit` 增加三项，任一失败 exit 2：
  1. `distinct_configs`：把每个 run 的 `(rank_pattern, r, lora_alpha, alpha_pattern, use_dora, peft_type)` 规范化后取哈希；**同一张表内出现重复哈希 ⇒ 报错并列出去重后的有效臂数**。
  2. `param_budget_spread`：`max(params)/min(params)`；**> 1.05 ⇒ 报错**（当前主表为 6,816,768/3,407,872 = **2.000**）。
  3. `alpha_ratio_uniform`：所有方法所有模块的 `α/r` 必须相等（当前 adaq 为 1.333/4.000 vs qlora 2.000 ⇒ 报错）。
- 验收：对当前 `results_v2/` 运行，`_audit` 报出全部三项失败（**这是本修复项的正确性自检：它必须能发现本评审已发现的问题**）；修复 F4/F15/M8 后重跑 exit 0。

**F30 — `docs/eval-protocol-v2.md` §6 的写作校准规则需被强制执行**
- 实测：§6 已写「MMLU 是知识问答，2000 条 OpenHermes 指令微调动不动它，**不能作为『方法有效性』的主证据**；应把主证据放到指令敏感任务上」。但 `results_v2/FINAL_REPORT.md` 的主表仍以 MMLU 为首列，14B 的头条也是 MMLU 持平（.78429 vs .78372）。
- 改法：主表列序改为 **IFEval → TruthfulQA(mc1/mc2) → GSM8K → MMLU**（指令敏感在前，知识问答在后并标注「不作为主证据」）；或按 §6 的建议把 MMLU 移入附录。**注意**：`results_v2/summary.json._audit.tasks` 实测已含 `ifeval` 与 `truthfulqa`，且 `FINAL_REPORT.md` 口径行写明「TruthfulQA mc1+mc2 + IFEval 0-shot」、`.done_safety` marker 12/12 齐全 ⇒ **数据已在手，只是没被放到主证据位置**。
- 验收：`results_v2/FINAL_REPORT.md` 主表首列为指令敏感任务；MMLU 列带「不作为方法有效性主证据（见 `eval-protocol-v2.md` §6）」表注。

---

## 7. 三句话总评

1. **装置层面**：8B 主表在**参数预算**（adaq 1.115× / adalora **2.000×** qlora，实测 safetensors 元素数 3,801,088 / 6,816,768 vs 3,407,872）、**评测基座**（NF4 训练 / **BF16 干净基座**评测，实测 12/12 份 `results_*.json` 的 `model_args` 无 `load_in_4bit`，而 `layer_stats.rel_error` 达 0.0926–0.1227）、**基线收敛条件**（AdaLoRA 三个种子的全部 checkpoint `lora_E` 秩**恒为 16**、`total_elems` 恒为 6,816,768，`tinit=20/tfinal=60/total_step=200` 在 `max_steps=21` 下**剪枝一次都未触发**，loss 3.096→2.937、token_acc .4990→.4955）三个维度同时不对齐，而 14B 表两臂 sum-of-ranks 为 **528 vs 768**、GSM8K 仅 **n=1**、strict 口径对 Instruct 基座失效（.1827/.1304 对官方 ≈0.94），**没有任何一张表能独立支撑一个因果结论**。

2. **信号层面**：项目自己的 `ranks.json`/`ranks_qwen14b.json` 经我独立重算，`rel_error` 层间 CV 仅 **1.47%/3.98%**、跨模块类型比值 **0.9996/0.9997**（而 `paper-framework.md:12` 声称「可相差数倍」）、`Spearman(importance, score) = 0.9979/0.9987` 对 `Spearman(rel_error, score) = 0.1440/−0.1967`、`Spearman(rel_error, assigned_rank) = +0.1112/−0.2039`（**14B 上把更少的秩分给了量化误差更大的层**），**方法名为「量化感知」而实为「范数感知」，且这是 NF4 的 per-block absmax 定标 + 正态最优码本所决定的结构性后果**——这一机理陈述比现有的阴性表述强得多，是本项目真正可发表的贡献；但同配置同种子的两次独立训练在 GSM8K strict 上相差 **11.52 点**（`product` .42532 vs `importance_only` .31008，`rank_pattern` 均为 `{q_proj:12,v_proj:4}`），而全部被测效应 ≤2.7 点，**信噪比约 1:4.3，任何方向的效应主张（含「等于随机」的等价性主张）都超出当前数据的分辨能力**。

3. **落档层面**：`AQLoRA`(2608.23816) 与 `PARA`(2604.27796) 在项目自身的 `docs/`、`scripts/` 中**全库 0 命中**（我的 grep 与同侪 `jcr-gate.md:30,70` 独立一致），四个 Gap 的检索状态仍是「待 scholar 核查」，而作者自选的新颖性参照物 **LoftQ 从未被运行**（M2）；加上主表**没有任何无-adapter 锚点行**（M1，实测 `results_v2/summary.json` 顶层键仅 `['adalora','adaq','dora','qlora','_audit','significance_test']`，而 NF4 表已显示四个微调臂的 MMLU **全部低于** `baseline_nf4`：.6334/.6343/.6357/.6335 vs .6364）、项目自己声明为「最小集」成员的实验三 R-曲线 **0 产物**（M4）、基座 sha256/checksum 在代码与产物中 **0 命中**（F13，adapter 记录的 `base_model_name_or_path` 是**本机不存在的死路径**）——**当前形态投 Q2 会在「基线缺失 + 文实相反 + 新颖性零落档」三条上同时被击落，且这三条都不需要审稿人做主观争论；但 F2/F12/F14(c)/F16 四项成本极低（合计 <1 天）即可消除其中最致命的文实相反与假标签，F17 的 M1/M3 与 F19 可在 1 周内补齐，M4 R-曲线是唯一的大额投入，也是唯一还可能产出正面结果的实验。**

---

## 附录 A：本评审实测数字索引（便于作者逐条复核）

| 数字 | 值 | 来源 |
|:--|:--|:--|
| 8B adapter 元素数 | qlora 3,407,872 / dora 3,571,712 / adaq 3,801,088 / adalora 6,816,768 | `outputs/adapters/*_seed42/adapter_model.safetensors` 二进制头 |
| 8B 参数比 | 1.000 / 1.048 / **1.115** / **2.000** | 同上 |
| 8B sum-of-ranks | qlora 512 / dora 512 / adaq **512** / adalora **1024** | `adapter_config.json` + 层数 32 |
| 14B adapter 元素数 | adaq 4,620,288 / qlora 6,291,456 | `outputs/adapters_qwen14b/*_seed42/adapter_model.safetensors` |
| 14B sum-of-ranks | adaq **528** / qlora **768** | `adapter_config.json` + 层数 48 |
| AdaLoRA `lora_E` 秩 | **`{16}` 于全部 10 个 checkpoint（3 种子）** | `outputs/adapters/adalora_seed*/checkpoint-*/adapter_model.safetensors` |
| AdaLoRA 调度 | `init_r=16, target_r=8, total_step=200, tinit=20, tfinal=60, deltaT=5`；`max_steps=21` | `adapter_config.json` + `trainer_state.json` |
| 8B 训练步数 | **21**（6.76 step/epoch）；tokens ≈623K；1,854 tok/seq | `outputs/adapters/*/checkpoint-21/trainer_state.json` |
| 14B 训练步数 | **255**（84.4 step/epoch）；tokens ≈2.07M | `outputs/adapters_qwen14b/*/checkpoint-255/trainer_state.json` |
| LR 对称性 | step10 = 1.3090e-4、step20 = 4.8943e-6，**12/12 份逐位相同**；14B 峰值 1.9999e-4 | `trainer_state.json.log_history` |
| α/r | qlora/dora 2.000；adaq **1.333(q)/4.000(v)**；adalora **1.000** | `adapter_config.json`（adaq `alpha_pattern={}` 全 18 份） |
| `max|w|` | qlora 0.1250 / adaq 0.2471（**1.98×**） | safetensors 逐张量扫描 |
| `rel_error` 8B | min .092642 max .100753 mean .093608 CV **1.47%** max/min **1.0875×** | `ranks.json.layer_stats` |
| `rel_error` 14B | min .092305 max .122659 mean .094280 CV **3.98%** max/min **1.3289×** | `ranks_qwen14b.json.layer_stats` |
| `rel_error` q/v 均值比 | **0.9996（8B）/ 0.9997（14B）** | 同上，按模块类型分组 |
| `importance` CV | **56.20%（8B）/ 39.04%（14B）** | 同上 |
| Spearman 8B | (rel,score) **+0.1440**；(imp,score) **+0.9979**；(rel,imp) +0.1056；(rank,score) +0.9171；**(rel,rank) +0.1112** | 独立重算 |
| Spearman 14B | (rel,score) **−0.1967**；(imp,score) **+0.9987**；(rel,imp) **−0.2194**；(rank,score) +0.9237；**(rel,rank) −0.2039** | 独立重算 |
| 实际秩取值集合 | 8B **{4,5,6,11,12,13}**；14B **{4,5,6,7}**（配置 `r_min=4, r_max=128`） | `per_layer_ranks` |
| 方案A 六臂 `rank_pattern` | equal_weight {q10,v4}=**448**；error_only {q8,v8}=512；importance_only {q12,v4}=512；product {q12,v4}=512；random {q7,v8}=**480**；uniform {q8,v8}=512 | `outputs/ablation_adapters_schemeA_*/adapter_config.json` |
| 方案B 秩图差异 | **error_only vs uniform = 0/64**；**importance_only vs product = 2/64**；equal_weight vs product = 39/64；random vs product = 50/64 | `outputs/ablation_adapters/*/adapter_config.json` 逐键比对 |
| 同配置噪声（方案A s3407, strict） | product **.42532221379833207** vs importance_only **.3100833965125095** ⇒ **Δ = 11.52 点** | `results_v2_ablation_multiseed/summary.json` |
| 同配置噪声（方案A s3407, strict） | error_only .3934799090219864 vs uniform .38286580742987114 ⇒ Δ = 1.06 点 | 同上 |
| 方案B 消融（预算守恒 512） | uniform MMLU **.6480558325024925（最高）**；error_only strict **.4746019711902957（最高）**；product MMLU .6458481697763852 / strict **.3949962092494314（第5/6）** / flex .5026535253980288 | `results_v2_ablation/**/results_*.json` |
| 8B 主表（BF16 基座，n=3） | qlora .6533±.0006/.4711±.0046；dora .6543±.0015/.4799±.0055；adalora .6538±.0012/.4774±.0009；adaq .6540±.0007/.4766±.0031 | `results_v2/FINAL_REPORT.md` |
| NF4 表（n=1） | baseline .636376584532118/.41243366186504926；qlora .6333855576128756/.41243366186504926；dora .6343113516593077/.4230477634571645；adalora .6356644352656317/.4184988627748294；adaq .6335279874661729/**.40636846095526913** | `results_v2_ptq_nf4/summary.json` |
| NF4 表 flexible | baseline **.41925701288855194** = qlora **.41925701288855194**（逐位同值；strict stderr 末两位 .445 vs .447 ⇒ 独立计算） | 同上 |
| 14B seed42 | adaq .7842899871813132 / strict .18271417740712662 / flex **.6588324488248674**；qlora .7837202677681242 / strict .13040181956027294 / flex **.5253980288097043** | `results_v2_qwen14b/summary.json` |
| 14B seed2025 | **仅 mmlu**：adaq .7840051274747187 / qlora .7836490528414756（Δ=+0.00036） | `results_v2_qwen14b_seed2025/` |
| 14B seed3407 | **结果目录不存在**（adapter 已训好）；`summary_3seeds.json` 与 `REPORT_14B.md` **均未生成** | `results_v2_qwen14b_seed3407/`（缺） |
| 三向同值 | **0.41470811220621684 = 547/1319** 出现于 `adaq_nf4`(flex)、`schemeA_s3407/equal_weight`(strict)、`ablation/random`(strict) | 三处 `results_*.json` |
| 样本量 | mmlu `sample_len=14042`；gsm8k `n-samples {original:1319, effective:1319}` —— 全部表全量，无 `--limit` 污染 | 各 `results_*.json` |
| few-shot | `mmlu` 57 子任务全部 5；`gsm8k` 4；`truthfulqa_mc1/mc2`、`ifeval` 0 | `results_v2/adaq_seed42/*.log`、`results_v2_ptq_nf4/adaq_nf4/*.log`（**仅抽查 2 个目录**，见 F10） |
| 评测 `model_args`（8B 主表） | `{'pretrained':'.../models/Meta-Llama-3-8B','peft':'<adapter>','dtype':'bfloat16'}` —— **12/12 份无 `load_in_4bit`** | `results_v2/**/results_*.json` |
| 评测 `model_args`（NF4 表） | 含 `load_in_4bit:True, bnb_4bit_quant_type:'nf4', bnb_4bit_use_double_quant:True` —— **10/10 份** | `results_v2_ptq_nf4/*/results_*.json` |
| 评测 `model_args`（14B） | 含 nf4 全套 ⇒ **唯一 train/eval 自洽的表** | `results_v2_qwen14b/*/results_*.json` |
| adapter dtype | `outputs/adapters/*` = **BF16**；`outputs/ablation_adapters*/*` = **F32**；四者 `non_finite=0` | safetensors 头 |
| 基座身份 | `/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B`：`LlamaForCausalLM`, hidden 4096, 32 layers, **8 kv heads**, bf16, 4 分片（4,976,698,672 / 4,999,802,720 / 4,915,916,176 / 1,168,138,808 B）⇒ **base 非 Instruct** | `models/Meta-Llama-3-8B/config.json` |
| 死路径 | `/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B` **本机 NOT PRESENT**，却是 12 份 8B `adapter_config.json` 的 `base_model_name_or_path` | 实测 |
| 溯源 | `sha256/checksum/md5/model_hash` 于 `scripts/ docs/ results_v2/` **0 命中**（唯一命中在 `docs/review/nf4-spectral.md` 同侪文本） | grep |
| 缺失产物 | `*budget* *R128/256/768/1024* *loftq* *gptq* *awq* *hqq* *baseline_bf16*` 目录扫描 **0 命中** | `find . -maxdepth 2 -type d` |
| 缺失引用 | `AQLoRA / 2608.23816 / PARA / 2604.27796` 于 `docs/ scripts/` **0 命中** | grep（与 `jcr-gate.md:30` 一致） |
| `peak_vram_gb` 8B | qlora 14.80 / adaq 14.81 / adalora 14.84 / dora 15.01（adalora 参数 2.000× 而显存 +0.27%） | `results_v2/system_metrics.json` |
| `peak_vram_gb` 14B | seed2025/3407 = 17.6/17.62；**seed42 = 6.67（物理不可能）** | `outputs/adapters_qwen14b/*/training_stats.json` |

## 附录 B：本评审自我披露的局限

1. **未运行任何训练或评测**（GPU 被占用，且任务要求只读）。所有数字来自既有 artifact 的解析与独立重算，**不含任何新实验**。因此 F8（噪声底）、F17（缺失基线）、F20（命题3b）的修复效果无法在本评审中验证。
2. **few-shot 核查为抽查**：只逐字 grep 了 `results_v2/adaq_seed42` 与 `results_v2_ptq_nf4/adaq_nf4` 两个目录的日志（结论一致 ✓）。其余 22 个 run 未逐条核查——这是 F10 存在的原因，也是我**不把「few-shot 全表一致」写成无条件结论**的原因。
3. **`‖BA‖_F/‖W‖_F` 未实测**：§2.5(ii) 与 §3.3(ii) 中「基座错配比处理效应高约一个数量级」是**基于 `rel_error`（实测 0.0926–0.1227）与 `max|w|`（实测 qlora 0.1250 / adaq 0.2471）的量级估计**，非直接测量。F6(3) 是把它变成实测的修复项。
4. **Δ=11.52 点的成因未查明**：同配置同种子出现该量级差异，除 GPU 非确定性外还可能涉及 `--seed` 实际取值、自动 resume、`--batch_size auto` 的 padding 差异。本评审只确认了「两份 `adapter_model.safetensors` 的 128/128 张量 sha256 全不同、整文件 sha 不同」，**未确认两次训练的完整命令行是否一致**（库中无该记录，这正是 F13 要解决的溯源缺口）。因此该值应作为「噪声底的上界证据」引用。
5. **三向同值 547/1319 判定为巧合，但未做 per-sample 比对**：lm_eval 未开启 `--log_samples`，库中无逐题预测，无法直接排除产物复用。已列为 F2 的附带必查项。
6. **同侪评审的 md5 表述存在分歧**：`nf4-spectral.md:20,39` 记「error_only md5 ≡ uniform（8940e8dbaf65）」；我实测 `configs/ablation_ranks/ranks_error_only.json` = `19b80ce915cd3560ac50a70726ad645a` ≠ `ranks_uniform.json` = `7b3f73af8ed789604aca09c2e0adef2b`。塌缩的**权威证据在训练产物层**（`adapter_config.json` 的 `rank_pattern` 逐键 0/64 不同），不在 config 文件层。建议三份评审统一表述。
7. **未评估理论部分**（Weyl 扰动界、平均谱抬升命题、代理质量上界）的正确性——超出本评审角色范围，由 `docs/theory-review-report.md` 与同侪评审覆盖。本评审只指出 §5.5 的命题3b 缺少对应测量。
8. **14B 的 Qwen2.5-14B-Instruct 官方 GSM8K ≈0.94 为记忆值，未联网核实**，仅用于说明 strict .1827/.1304 的量级异常。F11 要求在论文中引用可核实的官方数值来源。
