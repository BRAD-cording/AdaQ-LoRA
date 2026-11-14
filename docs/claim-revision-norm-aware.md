# 主张重构：从「量化误差感知」到「范数感知 + 阴性结论」（方向 B 落档稿）

> 状态：**提案，未改正文**。需作者确认后再同步到 `docs/paper-framework.md` / `paper-outline.md`。
> 触发：m04183 拍板方向 B；2026-11-14/15 的 NF4 根因分析与双种子消融给出了更硬的证据，措辞需要比原计划更保守。

---

## 1. 证据链（三条，都可复核）

### E1 — NF4 下 AdaQ 不如不分配（P0-B1，10/10 全绿）

| 方法（NF4 加载） | MMLU | GSM8K |
| --- | --- | --- |
| baseline（纯 NF4，不挂 adapter） | 0.6364 | 0.4124 |
| qlora_nf4 | 0.6334 | 0.4124 |
| dora_nf4 | 0.6343 | 0.4230 |
| adalora_nf4 | 0.6357 | 0.4185 |
| **adaq_nf4** | 0.6335 | **0.4064** |

AdaQ 在 GSM8K 上是五个里最差，且低于不分配（baseline）。若主张成立（"量化误差大的层需要更多秩"），NF4 下应是它的主场，实测相反。

### E2 — 打分函数的真实驱动项是权重范数，不是量化误差

打分 `score_l = rel_error_l × importance_l`，其中
`rel_error_l = ‖dequantize(W_l^NF4) − W_l^fp16‖_F / ‖W_l^fp16‖_F`，`importance_l = ‖W_l‖_F`。

64 层实测：

| 量 | 变异系数 CV | 与 score 的 Spearman |
| --- | --- | --- |
| rel_error | **1.47%**（0.0926–0.1008，几乎恒定） | 0.144 |
| importance | **56.2%**（13.33–75.27） | **0.998** |

结论：score 的排序 **99.8% 由 ‖W‖ 决定**，量化误差项只贡献噪声。因此该方法在数学上等价于**范数感知（importance-aware）秩分配**，"量化误差感知"的名不副实。

### E3 — 双种子消融：product 不优于随机/均匀（P1-D4，24/24，FAIL=0）

| strategy | MMLU (mean±std, n=2) | GSM8K (mean±std, n=2) |
| --- | --- | --- |
| random | 0.6464±0.0020 | **0.4321±0.0375** |
| **product（完整 AdaQ score）** | **0.6464±0.0032** | 0.4230±0.0032 |
| error_only | 0.6457±0.0013 | 0.4189±0.0359 |
| uniform | 0.6454±0.0003 | 0.4147±0.0450 |
| equal_weight | 0.6456±0.0004 | 0.4124±0.0032 |
| importance_only | 0.6456±0.0012 | 0.3700±0.0847 |

种子间反转：s3407 上 product 双榜第一（.6486/.4253），s2025 上 random 第一（.6478/.4587）而 product gsm8k 只有 .4208。**seed42 单种子的 product .4829 > random .4435 未复现**。
唯一稳定的是方差：product 的 gsm8k std ±.0032 最小，random ±.0375、importance_only ±.0847。

---

## 2. 建议的措辞（按段落给出，可直接替换）

### 摘要 / Abstract（替换"量化感知"表述）

> 我们研究 4-bit 基座上的 LoRA 秩分配。一个自然的假设是按逐层量化误差分配秩；但我们在 Llama-3-8B + NF4 上量化发现，NF4 的**相对**重构误差在层间几乎是常数（CV 1.47%），而以 `误差×范数` 构造的打分其排序 99.8% 由权重范数决定——该规则实际是**范数感知**的。我们给出扰动界与谱抬升分析，并在 2 个种子 × 2 个任务上做了六策略消融：该分配与随机/均匀分配在均值上不可区分（GSM8K：random .4321 vs ours .4230），且种子间出现方向反转。我们据此报告一个**阴性但可复核**的结论，并给出该设置下效应量的噪声上界。

### 贡献（三条，全部可验证）

1. **诊断性贡献**：给出 NF4 相对误差层间平坦性的量化证据（CV 1.47%）与打分函数的主导项分解（Spearman 0.998 vs 0.144），说明"量化误差驱动秩分配"在该范式下不成立。
2. **理论贡献**：扰动界（Weyl）+ 平均谱抬升命题 + 代理质量上界，并明确指出连续性松弛条件在实际配置下被满足/不被满足的边界。
3. **实证贡献（阴性）**：6 策略 × 2 种子 × 2 任务的消融表明，范数感知秩分配不优于随机/均匀；报告方差分解与种子反转，给出效应量的可用上界。

### §3.3 方法小节（改口径）

- 规则名改为 **norm-aware / importance-aware allocation**；
- 明确 `score_l ∝ ‖W_l‖_F`（若保留 `×rel_error`，须写"该项在实践中近似常数，对排序无实质贡献"，并给 CV 与 Spearman）；
- **删除**"量化误差异质性"作为动机；异质性只保留在 `importance` 上（CV 56.2%）。

### §5 结果分析（必须新增一段）

> Across two seeds the proposed allocation is not separable from random or uniform assignment; the between-seed spread (GSM8K up to 5.3 points for random) exceeds the between-strategy gap (≤0.9 points). We therefore report point estimates without significance claims and treat the single-seed advantage previously observed at seed 42 as not replicated.

### §6 Limitations（新增 L6，并把 L3 改写）

- **L6（新增）**：效应量低于种子噪声。n=2、8B、2 任务，无显著性检验；不得据此声称优越性。
- **L3 改写**：原"NF4 量化冻结假设"改为"量化误差项近似常数假设"——并给出 E2 的 CV/Spearman 作为该假设被**实证削弱**的证据。

### 必须删除/降级的表述

| 原表述 | 处置 |
| --- | --- |
| "量化感知的秩分配" | 改为"范数感知（importance-aware）" |
| "高误差层需要更大容量来补偿量化偏差" | 删（E2 不支持，且 rel_error 平坦） |
| "提升参数效率"/"优于均匀分配" | 删（E3 不支持） |
| "零额外显存开销下提升" | 保留"零额外开销"这半句，"提升"删 |
| Gap-1「忽视量化误差异质性」 | 改为"忽视权重范数的层间异质性" |

---

## 3. 仍未落袋的部分

- **P1-B4 Qwen2.5-14B**（训练中）：用于检验"换到 14B 是否仍无增益"。若 14B 上 product 明显领先，则结论变为"效应随规模出现"；若无，则负面结论更硬。**在它出数前，摘要里不写任何规模泛化从句**。
- 论文正文尚未改动，等作者确认本提案。

---

## 第 4 节：P1-B4 规模泛化实验（Qwen2.5-14B-Instruct，单种子 seed42）

**设置**：Qwen2.5-14B-Instruct；SFT 数据 `teknium/OpenHermes-2.5[:2000]`；3 epoch、lr 2e-4、max-length 512、batch 2×grad-accum 8；评测统一 **NF4**（`load_in_4bit=True`，`bnb_4bit_quant_type=nf4`，`double_quant=True`，compute bfloat16）。结果文件 `results_v2_qwen14b/summary.json`。

| method | adapter 参数 | MMLU (acc) | GSM8K strict | GSM8K flexible |
| --- | --- | --- | --- | --- |
| adaq | 4.62 M | 0.7843 | 0.1827 | **0.6588** |
| qlora | 6.29 M | 0.7837 | 0.1304 | 0.5254 |

**可以写进论文的（措辞上限）**

> 在 14B 规模上，AdaQ 以少 26.5% 的 adapter 参数取得与 QLoRA 持平的 MMLU（.7843 vs .7837），并在 GSM8K 上观察到单种子的正向差异（flexible-extract .6588 vs .5254）。

**必须同时写进 Limitations 的三条**

1. **单种子**（seed42），无重复，不构成统计证据；8B 双种子为阴性，14B 单种子为正向 → **规模依赖的迹象，不是稳健增益**。
2. **strict-match 不可用**：两种方法 strict 都只有 .18/.13，而 flexible 到 .66/.53。Qwen-Instruct 的默认输出格式不满足 `#### N` 约束，strict 度量的是格式合规性而非推理能力。**14B 表格必须用 flexible-extract 并注明**。
3. **绝对水位偏低**：14B base + 2000 条 OpenHermes SFT 后 GSM8K flexible 只有 .53–.66，提示小样本 SFT 带来格式漂移与能力回退；该绝对值不能与官方 Qwen2.5-14B 成绩比较。

**因此：第 3 节里"B4 出数前不写规模泛化从句"的限制现在可以解除，但只能按上面"措辞上限"那一句写，且必须带 Limitations 三条。禁止写成"AdaQ 在 14B 上优于 QLoRA"或"AdaQ 具备规模可扩展性"。**
