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

> [2026-11-16 评审裁定更正——**非预算对齐比较**] 在 14B 规模上，**在 QLoRA 获得 1.45× 秩预算（sum-of-ranks 768 vs AdaQ 实测 528）的条件下**，AdaQ 以 0.734× adapter 参数（4,620,288 vs 6,291,456）取得统计不可区分的 MMLU（.7843 vs .7837）。原「少 26.5% 参数」是预算错配（training_stats.json 的 "budget":512 为假标签，--budget 参数从未生效）的产物，**禁止任何参数效率主张**；GSM8K flexible-extract 差异（.6588 vs .5254）为单种子观察，strict-match 对 Instruct 基座失效（.18/.13 vs 官方≈0.94），三种子预注册判定（docs/preregistration-14b-3seeds.md）出数前不得写任何规模结论。

**必须同时写进 Limitations 的三条**

1. **单种子**（seed42），无重复，不构成统计证据；8B 双种子为阴性，14B 单种子为正向 → **规模依赖的迹象，不是稳健增益**。
2. **strict-match 不可用**：两种方法 strict 都只有 .18/.13，而 flexible 到 .66/.53。Qwen-Instruct 的默认输出格式不满足 `#### N` 约束，strict 度量的是格式合规性而非推理能力。**14B 表格必须用 flexible-extract 并注明**。
3. **绝对水位偏低**：14B base + 2000 条 OpenHermes SFT 后 GSM8K flexible 只有 .53–.66，提示小样本 SFT 带来格式漂移与能力回退；该绝对值不能与官方 Qwen2.5-14B 成绩比较。

**因此：第 3 节里"B4 出数前不写规模泛化从句"的限制现在可以解除，但只能按上面"措辞上限"那一句写，且必须带 Limitations 三条。禁止写成"AdaQ 在 14B 上优于 QLoRA"或"AdaQ 具备规模可扩展性"。**

---

## 第 5 节：P2-D3 交互效应（product = 误差 × 重要性）结案

**问题**：AdaQ 的 score 定义为 `rel_error × importance`，主张是"误差与重要性的乘积优于任一单独信号"（交互效应）。

**证据（8B，双种子，flexible-extract）**

| strategy | flex mean | 说明 |
| --- | --- | --- |
| error_only（仅误差） | 0.5254 | 单信号 |
| uniform | 0.5217 | 无信息基线 |
| random | 0.5216 | 随机基线 |
| product（乘积） | 0.5129 | 主张的完整信号 |
| equal_weight | 0.5069 | 等权基线 |
| importance_only（仅重要性） | 0.4989 | 单信号 |

**结案判定：交互效应未检出。** 理由三条：

1. product（.5129）不仅没有超过两个单信号，还分别低于 error_only（.5254，差 −1.25 点）和高于 importance_only（.4989，差 +1.4 点）——两个方向的差都小于同配置跑间噪声 **4.5 点** [2026-11-16 评审裁定更正：4.5 点为 n=1 对估计。另一同配置**同种子**对（product vs importance_only，seed3407，128/128 张量 sha 全不同）实测差达 **11.52 点**，另一对仅 1.06 点 ⇒ 噪声底是重尾**区间 ≈[0.9, 11.5] 点**，任何单点下界不可用；「不可区分」须改口为「未观察到超出噪声底的差异，n 不足以确立统计等价」。专项噪声底实验（≥3 对 × ≥3 次）见 review/fair-baseline.md F8。]
2. 六策略 overall 极差 2.7 点 < 噪声底 4.5 点 → 整张表**没有任何一对差异可被支持**。
3. 机理层面已解释：`rel_error` 逐层 CV 仅 1.47%（Spearman 与 score 0.144），`importance = ‖W_fp‖` CV 56.2%（Spearman 0.998）→ 乘积 99.8% 由范数项决定，**"乘积"实际上退化成了"仅重要性"**，交互项没有独立贡献。

**论文写法（唯一允许的表述）**：`We find no evidence that the product of quantization error and weight importance outperforms either signal alone; the difference is within run-to-run noise, and the product is dominated by the importance term.`

**禁止**：任何"product 最优/更好/有协同"的表述；任何以 seed42 单种子 `.4829 > .4435` 为依据的结论（该单种子结果未能复现）。

---

## 第 6 节：α/r 混杂披露与因果降级（B6，2026-11-16 新增）

**事实（实测，见 review/fair-baseline.md + review/rank-budget.md）**：全部 18 个 adaq adapter（8B 主表 3 seed + 14B 3 seed + 消融 6 策略）使用 `lora_alpha=16` + `alpha_pattern={}`（per_module_type 方案A），故有效缩放 α/r 为：
- q_proj：16/12 = **1.333**
- v_proj：16/4 = **4.000**

而对照臂 qlora/dora 为 α=2r=**2.000**（均匀），adalora 为 α/r=**1.000**。这**违反方法自身文档的「铁律2 α=2r」**（scripts/train.py L103-104）——成因是 train.py L87-89 的 per_module_type 分支设 alpha_pattern=None（退回全局 α=16），此前未被发现。

**影响证据**：adapter 权重 max|w| 实测 adaq **0.2471** vs qlora **0.1250**（1.98×）。α/r 失配直接改变 adapter 权重的有效学习率与前向缩放 → adaq 与 qlora 两臂在训练优化起点上即不对等。

**后果（因果降级）**：所有 adaq vs qlora/dora 的比较（8B 主表、14B、NF4 表）都被 **α/r 混杂**。「秩分配」不是唯一变量，差异**不能干净归因于秩分配**——可能部分或全部来自 α/r 缩放差异。因此：
- 任何「秩分配 → 精度差异」的因果主张**降级为描述性关联**；
- 14B GSM8K flex 配对差（+8.87pt）即便通过 Phase3-D 噪声底重判，**仍存 F11（无 chat-template 格式漂移）与 F15（α/r 混杂）两个未排除的替代解释**，不得写「AdaQ 秩分配优于 QLoRA」。

**修复**：
- ✅ scripts/train_phase3.py 已修（F15）：per_module_type 分支默认 `alpha_pattern={mod: 2r}`（α=2r，混杂消除），仅 `--legacy-alpha16` 复刻旧行为。**Phase-3 全部重训臂（D 噪声底 / A 主表 / C R曲线 / E 基线）均为 α=2r 干净配置。**
- ⚠️ 现有 adapter（主表 12 + 14B 6 + 消融 12）**保留 α/r 混杂**，仅作描述性阴性结论用；因果主张须以 Phase-3 α=2r 重训数据为准。
- TODO：14B 臂 α=2r 重训（B6 重训，GPU 排队）后，14B 比较才是干净的等-α 比较。

**论文写法（唯一允许的因果表述）**：
> The comparison between AdaQ and QLoRA in the main results is confounded by an α/r mismatch (AdaQ effective α/r = 1.33 for q_proj and 4.00 for v_proj, vs. 2.00 for QLoRA), a bug in the allocation code path that we identify and correct in the Phase-3 reruns. All AdaQ-vs-baseline differences reported here are therefore descriptive associations, not causal effects of rank allocation.
---

### 第 4 节更新（2026-11-16 · 3 种子出数 + 修正案 4414d9e 重判）

上方 §4 正文写于「单种子 seed42」期。现 3 种子数据已出（results_v2_qwen14b/REPORT_14B.md）：
- adaq flex mean 0.5770±0.0749，qlora 0.4882±0.0579；配对差 +13.34/+9.02/+4.25（均值 **+8.87pt**，3/3 同号）。
- Phase3-D 实测噪声底 N=max(N_seed 0.30, N_rep 4.09)=**4.09pt**（同 product 配置 9 次独立训练，results_v2_phase3_noise/summary.json）。
- **修正案 R1 机械套用：Δ_flex 8.87 > N 4.09 且 3/3 同号 → R1 满足**。判定措辞（红线）：「14B 上 AdaQ 相对 QLoRA 在 GSM8K-flex 观察到正向差异（均值 +8.9pt），待 chat-template 口径确认。」

**但这不是因果结论** —— R1 强制附带条件 + 五条保留（详见 REPORT_14B.md 重判段）：
1. **pending B5** chat-template 复评（格式 artifact 未排；出数前该差值禁进 abstract/intro/conclusion/图表标题）；
2. **N 是 8B 代理**，8B→14B 外推未验证（待 B11 14B 本地噪声对）；
3. **α/r 混杂**（14B adaq α/r 1.33/4.0 vs qlora 2.0，待 α=2r 重训）；
4. **seed42 拓扑/显存异常**（6.67GB vs 17.6GB）且单种子扛 +13.34 主导均值；
5. **重尾残余**（pooled N 4.09 < 单对极值 11.52）。

**措辞红线（不变）**：禁写「优于/等价/打平/显著」；MMLU 写 "no difference observed; underpowered to establish equivalence"（R3）；强制同报参数量+秩预算+非预算对齐声明（R5）；strict 永不作判据（R4）。**B5/B11/α-r 重训出数前，14B 正向差异只能进附录 + 标「未排除格式与规模混杂」。**

---

### §4 增补：B11 14B 本地噪声底复核（2026-11-16，零 GPU 现有数据）→ B1 结论诚实下修

§4 的「修正案 R1 机械套用：Δ_flex 8.87 > N 4.09 → R1 满足」用的是 **8B 代理** N=4.09pt（保留②明列 8B→14B 外推未验证）。B11 用现有 adaq×3 种子 flex 零 GPU 算出 **14B 本地噪声底**：

- adaq flex 三种子 .6588/.5118/.5603 → **N_seed(14B,adaq) = 极差 14.71pt**（std 7.49）
- 配对差 三种子 +13.34/+9.02/+4.25 → 极差 9.09pt，std 4.57，SE 2.64（n=3）

**发现：14B 本地种子噪声（14.71pt）>> 8B 代理（4.09pt），且 > Δ_flex（8.87pt）**。14B（Qwen2.5-14B-Instruct）gsm8k-flex 无 chat-template 下跨种子极不稳（几乎肯定 F11 格式错配 inflate；对比 8B Phase3-D N_seed 仅 0.30pt）。

**下修后 14B 结论（替代 §4「R1 满足→观察到正向差异」）**：
> 14B 上 AdaQ vs QLoRA 的 GSM8K-flex 配对差方向一致（3/3 同号，均值 +8.9pt），但量级**落在 14B 自身种子噪声内**（adaq 单臂极差 14.7pt / 配对差 std 4.6pt，N >> 8B 代理 4.1pt），且由 seed42 拓扑异常 run（flex .6588，peak VRAM 6.67GB）主导。**不构成稳健正向差异。** 在 B5（chat-template 复评格式）+ B6（α=2r 重训）排除混杂前，不得写任何 14B 方向性结论。（n=3 配对 t≈3.36/p≈0.04 名义显著但脆弱：n 欠功效 + seed42 离群主导 + 格式错配 inflate 方差 + α/r 混杂，不作正向证据。）

**B5 为决定性实验**（已排队 waiter 3350890）：若 14.71pt 噪声主要是格式错配，chat-template 下噪声与信号都应收窄；若 +8.87 是格式 artifact，B5 暴露之。**B11 GPU 重复（N_rep 14B）降为可选**——N_seed(14B)=14.71 已单独淹没信号，无需 N_rep；seed42 是否离群待 B5（chat-template 三种子重评）澄清。
