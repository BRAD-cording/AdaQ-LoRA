# AdaQ-LoRA 投稿前评审：量化协议与精度–效率声明

- **评审角色**：量化与低比特推理方向审稿人（JCR Q1/Q2 标准）
- **评审对象**：`/home/ubuntu/AdaQ-LoRA`，2026-11-15/16 仓库状态（GPU 训练进行中，全程只读评审）
- **评审范围**：①训练态量化与预计算 rel_error 协议一致性；②NF4 评测的部署有效性；③BF16/NF4 跨表协议；④"量化感知"主张边界；⑤PPL/延迟/显存测量协议与精度–延迟双报告完备性；另附主动发现。
- **环境**：torch 2.11.0+cu130 / transformers 4.57.6 / peft 0.21.2 / bitsandbytes 0.50.2 / lm-eval 0.4.13；GPU 实测 2×NVIDIA RTX 4090 D 24GB（`nvidia-smi`，注意与 docs 中"A10"记载冲突，见 F5）。

## 总体判定：**Major Revision**

数据与实验工程诚实、可复现（Q① 协议一致性通过），但论文正文性文档（paper-outline / paper-framework）目前承载三类被仓库自身数据证伪或无制品支撑的主张——"量化误差层间异质性"功效叙事、合成自测 ρ 写成"实测 ✅"、无制品的推理延迟数字——外加两处结构性缺陷（跨表协议未标注、14B 非 iso-budget）。按现状投稿 Q1/Q2 大概率 reject；修复以文本改写 + 3 个小型补测为主，均可短期完成。

---

## 0. 证据基础（本评审实读的文件与行号）

| 文件 | 关键位置 | 用途 |
|:---|:---|:---|
| `scripts/train.py` | :32,:37,:50,:98（lora_alpha=2·base_rank）；:115-132（argparse 默认：budget 512、base-rank 8、epochs 3、lr 2e-4、batch 4、accum 4、seq 2048、target q_proj,v_proj、OpenHermes-2.5[:2000]）；:146-151（BitsAndBytesConfig）；:158（sdpa）；:160（prepare_model_for_kbit_training） | 训练态量化配置、公平性核对 |
| `adaq-lora-research-pipeline/scripts/adaq_precompute.py` | :91-96（BitsAndBytesConfig）；:97-102（4bit GPU 加载 + FP16 CPU 参考）；:105（named_parameters）；:114-117（dequantize_4bit + 活 quant_state）；:119-122（err/rel/score 定义）；:183（默认 target-modules 7 类）；:20-83（水位法分配） | Q① 核心证据 |
| `scripts/measure_epsilon_l.py` | :125-130（BitsAndBytesConfig 同款）；:136-138（FP16 CPU 参考）；:153（dequantize_4bit）；:107-118（--self-test 合成 ε=0.02 分支） | ε_l 复测、F1/F3 证据 |
| `scripts/run_eval_ptq_nf4.py` | :48-59（五配置 + 任务）；:110-122（NF4 model_args）；:139-149（lm_eval 调用，seed 42） | Q②③ PTQ 表协议 |
| `scripts/run_eval_ablation.py` | :76（BF16 model_args，无量化） | 消融表协议 |
| `scripts/run_eval_b4_qwen14b.py` / `_seeds.py` | :26 / :34-39（NF4 同款参数） | 14B 表协议 |
| `scripts/run_eval_ppl.py` | :13（RUNS 默认六项）；:22-33（BF16/NF4 分支 + wikitext + seed 42） | Q⑤ PPL 协议 |
| `scripts/profile_training.py` | :49（flash_attention_2，与 train.py:158 sdpa 不一致）；:115-150（训练显存/吞吐测量点） | Q⑤ 延迟溯源 |
| `scripts/aggregate_system_metrics.py` | :33-53（peak_vram 读自 training_stats.json） | 显存声明溯源 |
| `ranks.json` / `ranks_qwen14b.json` | metadata + layer_stats（本评审独立重算核实） | Q④ 信号分解 |
| `results_v2/epsilon_l_report.json` | eps min .09260 / median .09320 / max .10072 / mean .09358；powerlaw alpha min .1815 / median .4356 / max 1.5905，r2_median .8018；zero_mean mean_abs_e ≈9.5e-4 | F1/F2 证据 |
| `results_v2_ptq_nf4/summary.json`、`results_v2_ppl/summary.json`、`results_v2_qwen14b/summary.json`、`results_v2/system_metrics.json`、`results_v2/FINAL_REPORT.md`、`results_v2/summary.json` | 全部数值 | 统计功效核算 |
| `docs/paper-framework.md`、`docs/paper-outline.md`、`docs/eval-protocol-v2.md`、`docs/claim-revision-norm-aware.md`、`docs/sota-comparison-protocol.md`、`docs/training-optimization-plan.md` | 全文/节选 | 主张核对 |
| `results_v2/*/eval_gsm8k.log` 等 tqdm 流 | "2.33s/it"、"5.75s/it" 唯一出现处 | Q⑤ 延迟溯源 |

---

## 1. Q①：训练态 NF4 与预计算 rel_error 的量化协议是否严格一致

**判定：✅ 一致（double-quant 二级误差被如实计入）；两处溯源瑕疵需修（1.3、1.4）。**

### 1.1 配置逐字一致（三方核对）

| 参数 | `train.py:146-151` | `adaq_precompute.py:91-96` | `measure_epsilon_l.py:125-130` |
|:---|:---|:---|:---|
| load_in_4bit | True | True | True |
| bnb_4bit_quant_type | "nf4" | "nf4" | "nf4" |
| bnb_4bit_compute_dtype | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 |
| bnb_4bit_use_double_quant | True | True | True |

三个脚本的 `BitsAndBytesConfig` 完全一致，无任何参数漂移。

### 1.2 反量化路径正确（这是本工作最扎实的一点）

`adaq_precompute.py:105` 直接取 `dict(q_model.named_parameters())`，:114-117 对每个参数调用
`bnb.functional.dequantize_4bit(q_param.data, q_param.quant_state)`——
使用的是**实际 4-bit 加载产生的活 `Params4bit.quant_state` 对象**（含 double-quant 的嵌套 absmax quant_state），不是手工重建 scale/offset。因此：

- double-quant 的**二级量化误差被完整计入** ‖E_l‖（dequantize_4bit 内部会先解嵌套的 8-bit absmax scale）；
- 不存在常见复现错误"用一级 absmax 近似反量化"导致的误差低估；
- 训练时前向看到的量化权重与预计算测量的权重是同一比特级对象（同配置、同加载路径）。

`measure_epsilon_l.py` 走同一路径（:153），其产物 `epsilon_l_report.json` 与 `ranks.json` 的 layer_stats 独立吻合（mean ε .09358 vs .09361），交叉验证了测量可重复性。

### 1.3 瑕疵 A（轻，需写入方法节）：FP 参考是 float16 投影

`adaq_precompute.py:100-102` 与 `measure_epsilon_l.py:136-138` 均以 `torch_dtype=torch.float16` 加载"真值"W。checkpoint 原生为 BF16：bf16(8,7)→fp16(5,10) 的 cast 引入 ~2⁻¹¹ 量级相对差（指数范围覆盖权重值域，无溢出风险），相对 ε≈0.094 可忽略（<0.6% 的 ε 扰动），但**必须写进方法节**——审稿人复核 ‖E_l‖ 时会问"你的 W 真值是什么精度"。

### 1.4 瑕疵 B（溯源，必须修）：信号权重与训练权重的同一性无哈希记录

`ranks.json` metadata 记载 `model_path=/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B`——该路径在训练/评测机上**不存在**（本评审 `ls` 实证）。即：驱动全部分配的 ε_l 信号是在另一台机器的 modelscope 权重副本上算的，而训练/评测用 `models/Meta-Llama-3-8B`，两者同一性没有任何哈希记录。
缓解证据：本机用训练副本重测的逐层 ε（`epsilon_l_report.json`）与 ranks.json layer_stats 在 ~3e-4（0.03%）水平一致（如 layer0 q_proj .100716 vs .10075；mean .093584 vs .09361），统计上强烈支持同源，且该量级差异不可能改变整数秩分配。但"统计一致"≠"身份证明"，审稿人有权要求硬证据。修复见清单 #16。

### 1.5 覆盖面与脚本默认值不一致（易埋雷）

`adaq_precompute.py:183` 默认 `--target-modules` 为 7 类模块，而实际产物 `ranks.json` 仅 64 个 q_proj/v_proj 位点（32 层×2），与 `train.py:125` 默认一致。**本次运行是对的，脚本默认值是错的**——任何人重跑默认命令将得到 7 模块 ranks 文件，与训练 target 不匹配且不报警。修复见清单 #15。
同时论文必须明示：秩分配只作用于 q/v 注意力投影，MLP 与 o_proj 不参与——不得暗示"全模型误差驱动分配"。

### 1.6 正面确认：LoRA scaling 无混杂

所有方法 `lora_alpha = 2×rank`（train.py:32,:37,:50,:98），adaq 的 alpha_pattern 同样 α_l=2r_l ⇒ 有效 scale α/r ≡ 2 恒定。adaq 的变秩不引入 scaling 混杂变量，这是 iso-budget 之外的第二个公平性设计点，建议写进 §4.1。

---

## 2. Q②：NF4 评测是否是正确的压力测试？能否声称部署收益？

**判定：⚠️ 作为"量化态端到端质量"压力测试有效；作为部署/性能收益证据无效。**

### 2.1 机制事实

bnb NF4 路径 = **4-bit 真实存储 + 即时反量化 + bf16 算术**（compute_dtype=bfloat16，所有配置一致；评测脚本 `run_eval_ptq_nf4.py:110-122` 原样传递）。因此：

- **显存收益是真的**：权重确实以 4-bit+double-quant scale 驻留（8B 模型 ~4.5GB 权重），训练峰值 14.8-15.0GB（`system_metrics.json`）在 24GB 卡上可行，这可以声称；
- **吞吐收益不存在**：没有 NF4 matmul kernel，每次前向都做 dequantize-on-the-fly，生成速度通常**慢于** bf16 同权重。任何"推理加速 / 部署吞吐 / 边缘落地"的句子都是把仿真量化计算当真实 4-bit kernel 收益，审稿人（本方向）一眼识破；
- 仓库中**零个**真实 4-bit kernel 对照（GPTQ/AWQ+Marlin、torchao int4、llama.cpp Q4 均未出现）；`scripts/merge_adapter.py` 存在但没有任何 eval 使用 merge 路径（4-bit 基座也无法 merge 回写）。

### 2.2 NF4 评测协议回答的两个合法问题（论文应如此定位）

1. **量化态质量保持**：在量化基座上训练出的 adapter，挂回同款量化态推理时质量是否保持（`results_v2_ptq_nf4` 的存在意义）；
2. **基座量化效应隔离**：同一 adapter 的 BF16 vs NF4 差（PPL +0.53：adaq 7.4314→7.9661，qlora 7.4341→7.9691，`results_v2_ppl`）度量的是基座量化损失，不是 adapter 差异。

### 2.3 必须主动报告的边界事实

`results_v2_ptq_nf4/summary.json`：四个 adapter 方法 MMLU 全部**低于** no-adapter 的 baseline_nf4（.6334-.6357 vs .6364）。2k OpenHermes SFT 对 MMLU 是轻微负作用——这是 SFT 数据属性而非任一方法的属性，但如果不主动写，审稿人会替你做这个减法并质疑全部方法。GSM8K strict：baseline .4124，dora_nf4 .4230 最高，adaq_nf4 .4064 最低，极差 1.7pt；n=1319 下二项 SE≈1.36pt ⇒ **全表任意两方法差异均在 ~1.3 SE 内，无统计功效**（叠加单训练种子，见 §3.3）。

### 2.4 结论

可以保留 NF4 表并声称"量化态下的质量持平"；**禁止**任何部署速度/吞吐收益措辞；若期刊要求部署证据，需补真实 4-bit kernel 基线（成本另计，非本次修复必需——改写声明即可闭环）。精确替换文本见清单 #9、#10、#11-L6。

---

## 3. Q③：BF16 与 NF4 跨表混用是否审稿致命伤？

**判定：❌ 当前形态（协议不进表题）是致命伤；✅ 双协议本身正当，按方案 A 处理即可治愈。不建议统一成单协议。**

### 3.1 现状与风险

| 表 | 数据源 | 基座 | 训练种子数 | 评测种子 |
|:---|:---|:---|:---|:---|
| 主表 | `results_v2`（`run_eval_unified.sh`/`eval_worker.sh`，BF16 model_args） | **BF16，无量化** | 3（42/3407/2025） | 42 |
| PTQ 表 | `results_v2_ptq_nf4`（`run_eval_ptq_nf4.py:110-122`） | **NF4+dq+bf16 compute** | 1（42） | 42 |
| 14B 表 | `results_v2_qwen14b`（`run_eval_b4_qwen14b.py:26`） | **NF4 同款** | 1 | 42 |
| 消融表 | `results_v2_ablation*`（`run_eval_ablation.py:76`，BF16） | BF16 | 2 | 42 |

同一 adaq adapter：BF16 表 GSM8K strict .4766±.0031，NF4 表 .4064——**7 个点的落差纯属协议差**。两表若同时出现在论文而 caption 不标协议，审稿人要么发现矛盾质疑数据可信度，要么误把协议差当方法效应。`docs/eval-protocol-v2.md` 内部已有清醒认识（"当前主表口径 = bf16 + adapter"、"若论文正文写'4-bit 合并后推理'需另跑 merge 组"），但该认识**没有传导到 outline 的表格定义**（§4.2/§4.3 caption 无协议行）。

### 3.2 两个协议各自回答什么（应原样写入论文）

- **Protocol-BF16**（主表/消融表）：隔离基座量化噪声后，秩分配策略对 adapter 学习质量的净效应。基座不量化 ⇒ 方法间差异只来自 adapter。
- **Protocol-NF4**（PTQ 表/14B 表）：部署态（量化基座+adapter 挂载）端到端质量，含基座量化损失的完整效应。
- 两者互补，缺一不可：只有 BF16 会被问"量化态下还成立吗"；只有 NF4 会被问"差异是分配还是量化噪声"。**统一协议反而削弱论证**（统一到 NF4 失去干净对照；统一到 BF16 失去部署态证据），故不推荐，除非目标期刊明文强制单一口径。

### 3.3 精确处理方案（方案 A，推荐）

1. 每张表 caption 加协议行 + 种子数（文本见清单 #6）；
2. §4.1 加"双协议声明"段（清单 #5），含跨表禁比规则与 +0.53 PPL 协议差锚点；
3. **补 BF16 主表锚行**：当前 BF16 表没有 no-adapter 基线，审稿人无法判断 LoRA 在 BF16 态是提升还是损伤（NF4 表有 baseline_nf4，BF16 表缺对称锚）。补 1 个 run（base BF16，MMLU+GSM8K）即可；
4. PTQ 表加单种子+噪声脚注（清单 #7）；若期刊要求，扩到 3 训练种子（成本 ~6 run）；
5. 正文任何位置禁止跨表数值相减/相比。

---

## 4. Q④："量化感知"主张在 adaq_nf4 最差时还能不能写、能写什么

**判定：❌ 作为功效来源不能写；✅ 作为信号机制描述可以写，且必须附分解数据。"quantization-aware training/QAT"一词全文禁用（本工作不是 QAT：训练循环无 fake-quant 前向，bnb 是存储量化）。**

### 4.1 信号分解（本评审独立重算 `ranks.json` layer_stats 核实）

- ε_l（rel_error）：mean .09361，**CV=1.47%**，max/min=1.087 ⇒ 层间相对误差近似常数；
- ‖W_l‖（importance）：CV=56.2%；
- score=rel×norm：代数上 **score ≡ ‖E_l‖_F**（rel=‖E‖/‖W‖ ⇒ rel·‖W‖=‖E‖）；
- R²(log score, log‖W‖)=**0.9995**（Pearson .999766）；R²(log score, log ε)=0.0014。
- 14B 同构：rel CV=3.98%，‖W‖ R²=0.9933。

即：由于 ε_l 近似常数，‖E_l‖=ε_l·‖W_l‖ ∝ ‖W_l‖——**水位法实际按权重范数分配秩**。"量化误差驱动"在字面上为真（信号确实来自反量化误差），在因果叙事上为假（决定分配差异的变异 99.95% 来自范数而非误差）。

### 4.2 现有 Gap 陈述被自家数据证伪

`paper-framework.md` 叙事主线："NF4 对不同模块类型的量化误差 ε_l 可相差数倍"。实测：q/v 的 ε 差 ≤1.09×（layer0：q_proj .10072 vs v_proj .09990）；"数倍"存在于**绝对**误差 ‖E‖（7.31 vs 1.33，5.5×），而它由 ‖W‖ 驱动。按现状写 Gap-1/Gap-4 = 把把柄递给任何一个会打开 `epsilon_l_report.json` 的审稿人。

### 4.3 "adaq_nf4 最差"的统计学解读

.4064 vs 表内最高 .4230（dora_nf4）：差 1.66pt ≈ 1.2 个非配对二项 SE（1.36pt），单训练种子——**"最差"本身不显著**。正确的表述不是"方法在 NF4 下失败"，而是"NF4 协议下（单种子）四方法与 no-adapter 基线在噪声内不可区分"。但这不 rescue 功效叙事：若 ε 异质性真驱动收益，error_only 消融应稳定优于 importance_only——内部文档 `claim-revision-norm-aware.md` §5 已结论"no evidence product outperforms either signal alone"，且 random 分配 GSM8K .4321±.0375（两种子）不低于 product .4230±.0032，种子间波动（最大 5.3pt）> 策略间差异（≤0.9pt）。

### 4.4 能写什么（措辞天花板）

1. **机制句**（真）：分配信号是冻结 NF4 量化误差的 Frobenius 范数 ‖E_l‖_F（经官方 dequantize_4bit 路径，含 double-quant 二级误差）；
2. **分解句**（必须同段出现）：由于 ε_l 层间近常数（CV=1.5%），该信号经验上等价于范数主导的分配规则（R²=0.9995）；
3. **效果句**（上限）：BF16 协议下与三个基线在种子噪声内持平；NF4 协议下单种子无显著差异；秩预算精确可控（水位法总秩守恒）、训练显存零额外开销（14.81 vs 14.80GB）、LoRA scale 恒 2 无混杂；
4. **负结果价值句**：给出"量化误差代理 vs 范数代理 vs 随机分配"的可复现对照与噪声地板测量；
5. **禁写**：QAT/quantization-aware training、"利用量化误差层间异质性获得提升"、"outperforms/显著优于/SOTA/p<0.05"（FINAL_REPORT 纪律已自禁，保持）、任何跨协议比较。

内部方向 B 已拍板（`claim-revision-norm-aware.md`，状态"提案，未改正文"）——**本评审判定：该提案必须在投稿前落进正文与标题**，否则审计材料与投稿文本分裂，属于可被定性为选择性呈现的风险。精确替换文本见清单 #2/#3/#4/#7/#8。

---

## 5. Q⑤：PPL 与延迟/显存声明的测量协议；精度–延迟双报告完备性

**判定：PPL ⚠️ 协议正确、覆盖不全；延迟 ❌ 无制品、禁止引用；显存 ⚠️ 只支持"零额外"；双报告不完备。**

### 5.1 PPL（`run_eval_ppl.py:22-33`）

- 协议本身有效：lm_eval wikitext `word_perplexity`，loglikelihood 类指标为确定性前向（无采样随机性），batch auto、seed 42、BF16/NF4 分支参数与主协议逐字一致；
- **覆盖缺口**：RUNS 默认只含 `qlora,dora,adalora,adaq,qlora_nf4,adaq_nf4`（:13）——缺 baseline_bf16、baseline_nf4（无 adapter 锚）、dora_nf4、adalora_nf4。因此"+0.53 = 基座量化效应"目前是"带 adapter BF16 vs 带 adapter NF4"的差，最干净的锚（无 adapter 两态之差）没测；
- 只测 seed42 adapter ⇒ 不含训练种子变异；BF16 四方法极差 .009（7.4314-7.4402）无解释力，只能写"PPL 对分配策略不敏感"；
- `sample_len=62` 为 harness 0.4.13 wikitext 默认切块数，论文报 harness 版本 + 任务配置即可；
- NF4−BF16 差在两个方法上高度一致（qlora +0.5350，adaq +0.5347）⇒ 基座量化效应可加性良好，这是可以写的正面结论。

### 5.2 延迟：**"adaq 2.33 s/it / dora 5.75 s/it（2.3×）"在整个仓库无可测制品支撑**

本评审穷尽溯源：

1. 两个数字字符串仅存在于 lm_eval 的 **tqdm 进度流**（`results_v2/*/eval_gsm8k.log`、`eval_safety.log`、`logs/b4_adaq_s3407.log` 等），且同一日志内 s/it 在 1-24 之间波动（generate_until + batch auto 下的瞬态进度条值），不构成测量；
2. `results_v2/system_metrics.json` 无任何 latency 字段；
3. 仓库无任何推理延迟基准脚本；`scripts/profile_training.py` 测的是**训练** tokens/s（50 步、NF4 基座、adapter 挂载、flash_attention_2 :49——与主训练 sdpa（train.py:158）协议不一致），训练日志 7.6-8.1 s/it、14B 6.1-7.7 s/it，与"2.33"量级都对不上；
4. `docs/` 全文 grep `延迟|latency|s/it|2.33|5.75` 零匹配——即该数字来自仓库外（口头/简报），**论文与一切报告禁止引用**。

因此"2.33 s/it 是合并后还是挂载测的、batch/seq 条件"这个问题在本仓库内**无答案可给**——它没被测过。若论文要保留 outline §4.5 表 3 的"推理吞吐"列，必须按清单 #10 的协议补测（merge vs adapter-attached × BF16/NF4 基座、固定 batch/seq、warmup、中位数+IQR、独占 GPU、报硬件与版本）；否则删除该列。另注：merge 路径只对 BF16 基座可定义（4-bit 基座不可回写 merge），且 merge 与挂载的数值不等价需单独报告（`scripts/numerical_invariance_check.py` 已有雏形）。

### 5.3 显存

`peak_vram_gb` 来自各 adapter 目录 `training_stats.json`（训练态 `max_memory_allocated`，单次运行、无重复、无方差）：qlora 14.80 / dora 15.01 / adalora 14.84 / adaq 14.81。四方法差 ≤0.21GB，属噪声级 ⇒ 只支持**"adaq 相对 qlora 零额外显存开销"**，不支持任何"省显存优势"；且这是训练态数字——推理态显存对照（BF16 基座 vs NF4 基座）未测，若写"NF4 相比 BF16 节省 X%"需补推理显存对照。参数量声明（adaq 3.80M / qlora 3.41M / adalora 6.82M，8B iso-budget 512）✅ 可写。

### 5.4 精度–延迟双报告完备性

精度侧：完备但功效受限（主表 3 种子 + 自禁显著性声明 ✅ 诚实；PTQ/14B 单种子 ⚠️ 需标注）。延迟侧：**零数据**。⇒ 双报告不完备；当前可提交的效率主张仅限"adapter 参数量 + 训练峰值显存 + 预计算耗时（<5min，需附实测计时）"。

---

## 6. 附加发现（超出简报范围，主动闭环）

- **F1 理论数值锚失效**：theory 侧"ε₄≈0.02 经验值"（`measure_epsilon_l.py` docstring；self-test 分支 :107-118 的合成扰动即按 0.02 构造）vs 实测 ε∈[0.0926, 0.1007]，**高 4.7×**。Theorem 1（Weyl 界）不依赖 ε 数值仍严格成立；但一切以 ε≈0.02 代入的数值推论/示例必须按实测重算。
- **F2 H1 幂律假设被实测否定**：`epsilon_l_report.json` powerlaw：α min .1815 / median .4356 / max 1.5905（层间差 8.8×），r2_median .8018。命题 2 推导链（H1"各层指数相近"→KKT→r\*∝c^{2/α}）的前提在真实模型上不成立 ⇒ Allocation Rule 只能定位为 heuristic（framework 已如此定位，outline §3.3 仍把推导链当卖点，需降级，清单 #13）。附带：α 是对 W_fp 谱的拟合，r² 中位数 .80 本身也只是中等。
- **F3 诚信级问题——合成自测被写成"实测 ✅"**：outline §3.3 命题 3b"Spearman ρ(score, σ²_{r+1}) ≥ 0.7（实测: ρ=0.9836 ✅）"。该 0.9836 来自 `--self-test` 合成矩阵分支；`paper-framework.md` L2 自己承认"自测 ρ=0.9836（合成数据）。实测 ρ 在真实模型上需实验验证"。两份文档互相矛盾，outline 的"实测 ✅"若进论文即虚假陈述。**必须改**（清单 #1），并二选一：补真实模型 ρ 实测，或删除数值承诺、降为 limitation。
- **F4 14B 表非 iso-budget**：qlora r=8×96 sites=768 总秩（6,291,456 参数）vs adaq budget 512（4,620,288）。"参数量少 26.5%"是预算设定差异，不是方法效率。当前表只能支撑"以更小预算持平 MMLU（.7843 vs .7837）+ flex GSM8K 更高（.6588 vs .5254，单种子）"的参数效率 claim；iso-budget 结论需补跑 qlora@512 或 adaq@768（清单 #14）。14B strict .1827/.1304 为格式不匹配产物（_meta 已承认），只可报告不可解读。
- **F5 硬件记载错误**：docs 多处写"A10 24GB"（paper-framework.md:118-120,:156；training-optimization-plan.md:1,:4;:33；sota-comparison-protocol.md:135），实测 2×RTX 4090 D 24GB。论文硬件节与复现说明必须写实用卡（清单 #12）。
- **F6 PTQ/14B 表单训练种子**：需在表内标注并给出噪声量级（清单 #7）。
- **F7 precompute 默认 target-modules 与产物不一致**（见 §1.5，清单 #15）。
- **F8 profile_training.py 的 flash_attention_2 与主训练 sdpa 不一致**（:49 vs train.py:158）——若引用其 tokens/s 必须同协议重跑。
- **F9 正面确认（保留并发扬）**：(a) `eval-protocol-v2.md` 的 per-task marker + `verify_eval_artifact.py` 机读校验、对 v2.0 假表事故的复盘，是罕见的反自欺工程；(b) `FINAL_REPORT.md` §3 honest bounds 与"禁止显著优于/SOTA/p<0.05"纪律；(c) α=2r 恒定 scale 的公平设计；(d) `claim-revision-norm-aware.md` 的负结果自查已到位——问题只在于这些结论尚未回写正文。
- **F10 训练超参一致性**：argparse 默认（epochs 3、lr 2e-4、batch 4×accum 4、seq 2048、budget 512、base-rank 8、OpenHermes-2.5 2000 条）与 outline §4.1 声称一致 ✅（lr 实测日志 .000199 为 warmup/decay 中间值，正常）。

---

## 7. 修复清单（编号 = 最终回复编号；P0=阻断投稿的纯文本修复，P1=小型补测，P2=保留相应主张才需要）

1. **[P0|诚信] `docs/paper-outline.md` §3.3 命题3b 行**
   原：`Spearman ρ(score, σ²_{r+1}) ≥ 0.7 (实测: ρ=0.9836 ✅)`
   改为：`Spearman ρ(score, σ²_{r+1})：合成数据自测 ρ=0.9836（measure_epsilon_l.py --self-test）；真实模型上的 ρ 未测——投稿前必须补测，否则删除本命题的数值承诺并在 Limitations 说明`
2. **[P0] `docs/paper-framework.md` 叙事主线段**
   原：`NF4 对不同模块类型的量化误差 ε_l 可相差数倍。这种异质性意味着：高误差层需要更大的适配容量来补偿量化偏差，而低误差层的秩是浪费的。`
   改为：`NF4 的相对误差 ε_l 层间近似恒定（实测 CV=1.5%，范围 0.093–0.101），但绝对误差 ‖E_l‖_F 随权重范数在层间相差数倍（本模型同层 q_proj 与 v_proj 相差约 5.5×）。分配信号 score=‖E_l‖_F·‖W_l‖_F 的对数变异 99.95% 由 log‖W_l‖ 解释（R²=0.9995）——本方法的实际行为是范数主导的秩预算规则，我们以负结果形式如实报告该分解。`
3. **[P0] `docs/paper-outline.md` §1 ¶3 Gap-4**
   原：`NF4 对不同模块类型的量化误差可差数倍——**Gap-4**`
   改为：`量化误差的绝对尺度 ‖E_l‖_F 在层间/模块间相差数倍（由权重范数驱动；相对误差 ε_l 层间差仅 ±7%），现有 PEFT 未将该信号用于容量分配——**Gap-4**`
4. **[P0] `docs/paper-outline.md` Title**
   原：`**AdaQ-LoRA: Quantization-Aware Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning of LLMs**`
   改为：`**AdaQ-LoRA: Norm-Aware Rank Budgeting for LoRA Fine-Tuning on NF4-Quantized LLMs**`
   备选标题同步改：`Norm-Aware Rank Budgeting for 4-bit Quantized LoRA Fine-Tuning`
5. **[P0] `docs/paper-outline.md` §4.1 末尾新增双协议声明（论文正文同款）**
   英文：`All results are reported under one of two explicitly labeled protocols. Protocol-BF16: the trained adapter is evaluated on a full-precision BF16 base; this isolates the rank-allocation effect from base-quantization noise (main and ablation tables, 3 and 2 training seeds respectively). Protocol-NF4: the adapter is evaluated on the same NF4 base as in training (bitsandbytes load_in_4bit, nf4, double quantization, bf16 compute, dequantize-on-the-fly), with the adapter attached and not merged; this measures end-to-end quality in the quantized-serving state (PTQ and 14B tables, single training seed). Numbers across protocols are not comparable: the same adapter differs by +0.53 word-perplexity (7.43 → 7.97, wikitext) purely from base precision. Because Protocol-NF4 uses BF16 arithmetic with 4-bit storage, no inference-throughput or latency advantage is claimed anywhere in this paper.`
   中文对照：`所有结果在两个显式标注的协议之一下报告。Protocol-BF16：adapter 在全精度 BF16 基座上评测，隔离基座量化噪声（主表 3 种子、消融表 2 种子）。Protocol-NF4：adapter 挂回与训练同款的 NF4 基座（bitsandbytes load_in_4bit+nf4+double-quant+bf16 compute，即时反量化、不 merge），度量量化服务态的端到端质量（PTQ 表与 14B 表，单训练种子）。跨协议数值不可比：同一 adapter 仅因基座精度不同，wikitext word-PPL 相差 +0.53（7.43→7.97）。由于 Protocol-NF4 是 4-bit 存储 + bf16 算术，本文任何位置不声称推理吞吐或延迟优势。`
6. **[P0] 三张表 caption 追加协议行**
   - 主表：`Protocol: BF16 base + adapter (no quantization at eval). Mean±std over training seeds {42, 3407, 2025}; eval seed 42; MMLU n=14042 (5-shot), GSM8K n=1319 (4-shot).`
   - PTQ 表：`Protocol: NF4 base (load_in_4bit, nf4, double-quant, bf16 compute) + adapter, single training seed (42). Binomial SE at n=1319 ≈ 1.4pt; all between-method gaps ≤1.7pt are within noise. Not comparable with the BF16 main table.`
   - 14B 表：`Protocol: NF4 base + adapter, Qwen2.5-14B-Instruct, single training seed, descriptive only. Rank budgets differ by design (AdaQ total rank 512 vs QLoRA 8×96=768): this is a parameter-efficiency comparison, not an iso-budget comparison. GSM8K strict-match is not applicable to Qwen2.5-Instruct output format and is reported for completeness.`
7. **[P0] PTQ 表正文解读句（替换任何"adaq_nf4 最差"表述）**
   `在 Protocol-NF4（单训练种子）下，四个 adapter 的 GSM8K-strict 位于 40.6%–42.3%，任意两两差异不超过 1.3 个二项标准误（SE≈1.4pt，n=1319），且 MMLU 全部略低于 no-adapter 基线（63.3%–63.6% vs 63.6%）——后者是 2k 指令 SFT 的属性而非任一分配策略的属性。本表因此作为"量化服务态下质量持平"的证据报告，不用于方法排名。`
8. **[P0] `docs/paper-outline.md` Abstract 结论行**
   原：`- 结论：零额外开销下的参数效率提升`
   改为：`- 结论：以冻结 NF4 量化误差 ‖E_l‖_F 为分配信号（经验上范数主导，R²=0.9995），AdaQ-LoRA 在 BF16 协议下与 QLoRA/DoRA/AdaLoRA 在种子噪声内持平，在 NF4 协议下（单种子）无显著差异；秩预算精确守恒、训练显存零额外开销。本文不声称推理加速（评测为即时反量化的 bf16 计算，非 4-bit kernel），并以负结果形式报告误差信号与范数信号的不可区分性。`
9. **[P0] 论文方法节新增"推理路径"段（英文正文）**
   `Inference path. Under Protocol-NF4, evaluation loads the base with bitsandbytes (load_in_4bit=True, nf4, double_quant=True, compute_dtype=bfloat16) and attaches the PEFT adapter without merging: base weights are stored in 4 bits and dequantized to bf16 on the fly; adapter arithmetic is bf16. Merged-weight inference is only definable on a BF16 base and is not part of Protocol-NF4. Since no NF4 matmul kernel is involved, all efficiency claims in this paper are restricted to (i) training-time peak GPU memory and (ii) adapter parameter count; no throughput or latency advantage is claimed.`
10. **[P0 删除 + P2 补测] 延迟声明**
    - 删除一切对"2.33 s/it / 5.75 s/it / 2.3× 推理差"的引用（该数字在仓库中无制品，仅 lm_eval tqdm 瞬态值；见 §5.2）；
    - `docs/paper-outline.md` §4.5 表 3 行，原：`| 表 3 | 效率表 | 训练 profile |`，改为：`| 表 3 | 效率表（仅参数量+训练峰值显存+预计算耗时；推理吞吐未测、禁止占位） | system_metrics.json + 补测计时 |`
    - 若坚持报延迟，按此协议补测后填入：`Latency protocol: single otherwise-idle GPU (state model and driver); paths = {BF16-base merged, BF16-base adapter-attached, NF4-base adapter-attached}; fixed input/output lengths 512/128; batch ∈ {1, 8}; ≥10 warmup and ≥100 timed generations per cell; report median and IQR of per-request latency and tokens/s; report torch/transformers/peft/bitsandbytes versions; merged-vs-attached numerical divergence reported separately.`
11. **[P0] 论文 §6 Limitations 从 5 条扩为 8 条（新增文本）**
    - `L6 (simulated-quantization compute): NF4 evaluation uses bitsandbytes dequantize-on-the-fly with bf16 arithmetic. No true 4-bit kernel (GPTQ/AWQ-Marlin, torchao int4) is benchmarked; deployment throughput of the quantized serving state remains unverified.`
    - `L7 (power-law heterogeneity): measured per-layer spectral exponents span α ∈ [0.18, 1.59] (median 0.44, median fit R² = 0.80), violating assumption H1 ("similar exponents across layers") that underlies the marginal-condition derivation; the allocation rule is therefore a heuristic, and its empirical proxy is norm-dominated (ε_l CV = 1.5%).`
    - `L8 (error-magnitude anchor and signal lineage): measured NF4 relative error is ε_l ∈ [0.093, 0.101], ≈4.7× the 0.02 rule-of-thumb used in earlier drafts; all numerical illustrations are re-anchored to measured values. ranks.json was precomputed on a modelscope copy of Meta-Llama-3-8B on a different host; per-layer ε re-measured on the training copy agrees to 0.03%, but weight hashes were not recorded at precompute time.`
12. **[P0] 硬件记载修正**
    - `docs/paper-framework.md:118-120`：`A10 24GB 单卡` → `NVIDIA RTX 4090D 24GB 单卡（双卡机器，训练/评测各占一卡）`；L4 标题 `A10 单卡约束` → `RTX 4090D 单卡约束`；:156 同步；
    - `docs/training-optimization-plan.md:1`：`A10 单卡训练效率优化计划` → `RTX 4090D 单卡训练效率优化计划`；:4 `NVIDIA A10 24GB 单卡` → `NVIDIA RTX 4090D 24GB 单卡`；:33 `A10 Ampere 可能不兼容` → `4090D（Ada）支持 FA2，但主训练实际使用 sdpa（train.py:158）`；
    - `docs/sota-comparison-protocol.md:135`：`batch_size = 4（A10 bf16 推理）` → `batch_size = 4（RTX 4090D bf16 推理）`
13. **[P0] `docs/paper-outline.md` §3.3 命题 2 推导链降级**
    原：`推导链: H1 → 边际条件 → r*_l ∝ c_l^{2/α} → 代理 ‖E_l‖_F·I_l`
    改为：`推导链（条件性）: 若 H1（各层谱指数相近）成立 → KKT 边际条件 → r*_l ∝ c_l^{2/α}。实测 α 层间 0.18–1.59（epsilon_l_report.json），H1 不成立 ⇒ Allocation Rule ‖E_l‖_F·I_l 定位为启发式（与 paper-framework 一致），命题 2 只作动机呈现，不作最优性依据`
14. **[P0 措辞 + P2 补跑] 14B claim 天花板（`results_v2_qwen14b` 解读与论文 §4.6）**
    `以少 26.5% 的 adapter 参数（4.62M vs 6.29M；总秩预算 512 vs 768——非 iso-budget，属参数效率对比而非同预算对比），AdaQ-LoRA 在单种子 NF4 评测下 MMLU 持平 QLoRA（0.7843 vs 0.7837），GSM8K flexible-match 更高（0.6588 vs 0.5254）；strict-match 对 Qwen2.5-Instruct 输出格式不适用（度量的是格式合规），仅完整报告不作解读。iso-budget 结论需补跑 QLoRA@总秩512（混合 r=5/6）或 AdaQ@768。`
15. **[P1] `adaq_precompute.py:183` 默认 `--target-modules` 改为 `"q_proj,v_proj"`**（与 train.py:125 一致），或运行时断言 target-modules ⊆ 训练 target 集合、不一致即报错退出。
16. **[P1] ranks.json 谱系补录**：metadata 增加 `model_sha256`（config.json、model.safetensors.index.json、各 shard 的 sha256）；若 /mnt 原机不可达，用本机 `models/Meta-Llama-3-8B` 重跑 precompute（复杂度自述 <5 min），确认整数秩分配逐位不变后以新文件替换并归档旧文件。
17. **[P1] PPL 覆盖补齐**：`RUNS=baseline_bf16,baseline_nf4,dora_nf4,adalora_nf4 bash`（`run_eval_ppl.py:13` 已支持环境变量；baseline 无 adapter 时 :29-30 自动跳过 peft 参数）。补完后"+0.53 基座量化效应"改以无 adapter 两态之差为锚，正文措辞同步（清单 #5 中的 +0.53 数字如变化需更新）。
18. **[P1] BF16 主表补 no-adapter 锚行**：base BF16 跑 MMLU(5-shot)+GSM8K(4-shot) 一个 run，加入主表首行 `base (no adapter)`。

**优先级汇总**：P0 全部为文本修复（#1-#9、#10 删除部分、#11-#14 措辞部分），投稿前必须完成；P1 约一天 GPU（#15-#18）；P2 仅在保留延迟/14B iso-budget 主张时需要（#10 补测、#14 补跑）。

---

## 8. 五问速答（供作者快速核对）

| # | 问题 | 判定 | 一句话 |
|:--|:---|:--|:---|
| ① | 训练 NF4 与预计算 rel_error 协议一致？ | ✅（2 处溯源瑕疵） | 三方 BitsAndBytesConfig 逐字一致；dequantize_4bit 走活 quant_state，double-quant 二级误差如实计入；瑕疵=FP16 参考投影 + 信号权重无哈希谱系 |
| ② | NF4 评测是有效压力测试？可声称部署收益？ | ⚠️ 质量测试有效 / 部署收益禁止 | 4-bit 真实存储 + bf16 即时反量化计算：显存收益可写，吞吐/延迟收益不可写，无真实 4-bit kernel 对照 |
| ③ | 跨表 BF16/NF4 混用致命？ | ❌ 现状致命，可治愈 | 双协议各自正当且互补；caption 协议行 + §4.1 声明 + 跨表禁比 + 补 BF16 锚行；不建议统一单协议 |
| ④ | adaq_nf4 最差还能写"量化感知"？ | ❌ 功效叙事不可写 / ✅ 机制描述可写 | score≡‖E‖ 且 99.95% 范数驱动（R²=.9995）、ε CV=1.5%；"最差"本身不显著（<1.3SE 单种子）；改名 norm-aware，QAT 一词禁用 |
| ⑤ | PPL/延迟/显存协议对不对？双报告完备？ | PPL⚠️ / 延迟❌ / 显存⚠️ / 不完备 | PPL 协议对但缺 4 个锚；2.33/5.75 s/it 全仓库无制品（tqdm 瞬态值）禁止引用；显存只支持"零额外"；延迟侧零数据 ⇒ 双报告不完备 |

*评审完成于 2026-11-16。所有行号与数值均以评审时仓库状态为准；本评审全程只读，未改动任何进程、配置或既有文件。*
