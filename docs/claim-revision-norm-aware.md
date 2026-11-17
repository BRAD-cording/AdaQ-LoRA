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

---

## 第 7 节：主修订追踪表（peft §8 骨架 × 全部阴性证据 × 当前状态）

> 本节把 claim-revision 从 peft §9 审计的 **16% 覆盖（4/25）** 扩到 **100% 处置**。
> §8.1-8.8 的英文重写处方（Abstract/贡献/§3.3 Method+Lemma/§5 Discussion/Limitations L1-L10/Conclusion/标题3选/必删表）**全文在 docs/review/peft-architecture.md §8，本节不复制，只追踪状态 + 补 peft §8 成稿后新增的数据**。
> 状态图例：✅ 已处置（措辞/文档/撤回，零GPU 完成） · 🔄 Phase-3 队列修复（GPU 自动，m05457 D→B→E→C→A） · ⏳ 待 B5/额外GPU · 📝 写作（待数据稳定落正文）

### 7.1 N1-N25 阴性/矛盾证据全处置（peft §9 的 25 项）

| # | 证据 | peft §9 判定 | 当前处置 | 状态 |
|:--:|:---|:--:|:---|:--:|
| N1 | NF4 口径 adaq 五者最差（GSM8K .4064 < base_nf4dq .4124） | ⚠️ | §8.1 Abstract + §8.4 Discussion 已给正文措辞；E1 证据链在 §1 | ✅ |
| N2 | score 排序 99.8% 由范数决定（ρ=.998 vs ‖W‖，.144 vs ε） | ✅ | E2 + §8.3 退化引理（‖E_l‖_F=ε_l‖W_l‖_F，ε 常数⇒∝‖W‖） | ✅ |
| N3 | 双种子六策略不可区分 + 种子反转 + seed42 .4829 未复现 | ✅ | E3 + §8.1；Phase3-D 实测噪声底 N=4.09 进一步坐实 | ✅ |
| N4 | 噪声底「4.5 点」取自 n=1 | ⚠️ | Phase-1 改区间[0.9,11.5]（§5）；**Phase3-D 9run 实测 pooled N=max(N_seed 0.30, N_rep 4.09)=4.09pt**（8B）；B11 算出 **14B N_seed=14.71pt** | ✅ 实测 |
| N5 | strict-match 是格式 artifact | ✅ | §8.5 L8 + eval-protocol §9 strict/flex 政策 | ✅ |
| N6 | 14B 单种子 + 措辞上限 + 禁用句 | ✅ | §4 + **B11 下修（ccadf1e）**：14B 本地噪声 14.71 > 信号 8.87 → 不稳健正向，seed42 疑离群 | ✅ 强化 |
| N7 | n=3 无有效 Wilcoxon（min p=0.25） | ⚠️ | §8.5 L7 已并入（主表 n=3 + 消融 n=2 均无有效检验） | ✅ |
| N8 | AdaLoRA 实跑总秩 1024（2× 预算），剪枝从未启动 | ❌ | **Phase3-B 重训**（train_phase3 F25 动态调度 total_step=auto + 训后断言秩降 target_r）；过渡期 §8.5 L6 披露 + Gap-3 撤回（b0d4dde） | 🔄 Phase3-B |
| N9 | 14B adaq 总秩 528 vs qlora 768；「同预算」假标签 | ❌ | §4 改述（b0d4dde）「QLoRA 获 1.45× 秩预算下 AdaQ 0.734× 参数 MMLU 不可区分」+ 声明非预算对齐；假标签已删；等预算重训=额外GPU（未排） | ✅ 披露 |
| N10 | 等秩口径 adaq 参数反多（8B +11.5%、14B +6.8%） | ❌ | −26.5% 撤回（b0d4dde，grep 清零）；§8.2 贡献4 改「+11.5%/+6.8% 开销，参数效率主张须等参数口径」 | ✅ |
| N11 | 训练预算三表不同（主表 21 步/消融 126/14B 255） | ❌ | **Phase3-A 主表重训 375 步**（F3 --no-packing）；过渡期 §8.1+§8.5 L5+eval-protocol §9 披露三表不可横比 | 🔄 Phase3-A |
| N12 | H1 幂律实测失败（α∈[0.181,1.591]，median 0.436，R².802） | ❌ | §8.3「derivation status」+ §8.5 L2：Prop2 降条件命题，规则降启发式 | ✅ |
| N13 | ε 实测 0.0936（非假设 0.02，差 4.7×） | ❌ | §8.5 L3；theory-review-report A2 + outline §3.2 的 0.02 须改 0.0936（P0-9 写作） | ✅ 待写作 |
| N14 | Prop3b ρ=0.9836 来自合成硬编码 | ❌ | §8.3「must not be cited as a measurement」；命题3b 撤 ✅ 标（b0d4dde）；outline :91 改「合成自测」（P0-9） | ✅ |
| N15 | 实现 score≡‖E_l‖_F ≠ 文档 ‖E‖·‖W‖ | ⚠️ | §8.3 显式声明「stored quantity = rel_error×importance = ‖E_l‖_F exactly」；outline :84 公式改（P0-2 写作） | ✅ 待写作 |
| N16 | Scheme A 把 64 位点塌成 2 整数 | ❌ | §8.3 truncation/grouping + §8.5 L4；逐层 Scheme B 算了未训（P2-2 可选） | ✅ |
| N17 | 六策略只有 4 种不同分配（product≡importance、error≡uniform） | ❌ | §8.1+§8.4 收敛证据；P1-1 消融表加「realised rank pattern」列（写作） | ✅ 待写作 |
| N18 | 相关工作缺 PARA（撞名）+ AQLoRA（独立同向阴性） | ❌ | **B10 完成**：related-work-search-log.md（999d60d 锚点 + 7ab7be4 扩 24 篇 OpenAlex 核验）；§8.4 convergent evidence 引 AQLoRA | ✅ |
| N19 | 秩预算扫描 R∈{128..1024} 标「必须」但从未跑 | ❌ | **Phase3-C 跑 R-曲线**（R∈{128,512,1024}×{adaq,qlora}×seed{42,3407}=12run，configs/rcurve/ ranks 已生成） | 🔄 Phase3-C |
| N20 | PPL sample_len=62，全距 0.009 | ❌ | §8.5 L9：不进正文，如入附录标 n=62 + 不可解释 | ✅ |
| N21 | 14B peak_vram 6.67GB（adaq≡qlora，<8B 14.81）物理不可信 | ❌ | train_phase3 F21 修（reset_peak_memory_stats 训前 + resumed 标志）；seed42 旧值标拓扑异常存疑；新臂用 F21 重测 | ✅ 修+待重测 |
| N22 | 主表训练命令不可复现（无日志/无脚本）；checkpoint 漂移 | ❌ | **Phase3-A 重训**（脚本入 git + provenance）；Phase-2（311c814）已补 env lock + results_ledger.csv + main-run-provenance.md | 🔄 Phase3-A+✅入库 |
| N23 | adaq 在 8B 主表三指标均非最优（DoRA 第一） | ⚠️ | §8.1 明写「DoRA holding the highest point estimates」；P1-3 主表补 flex 列（dora .4877>adaq .4850>adalora .4824>qlora .4809） | ✅ 待写作 |
| N24 | Gap-3 被自家数据反驳（adalora flex≈adaq 且拿 2× 预算） | ❌ | Gap-3 整条撤回（b0d4dde，sota-protocol:16）；同 N8 | ✅ |
| N25 | 14B seed 结果 + 预注册门无预算对齐条款 | ❌ | 三种子齐（REPORT_14B+summary_3seeds）；预注册补 R5 强制同报参数+秩预算+非预算对齐声明；B11 补 14B 本地噪声底 | ✅ |

**覆盖度：peft §9 审计时 ✅4/⚠️5/❌16（完全覆盖 16%）→ 现在 ✅ 已处置 21 项 + 🔄 Phase-3 队列修复 4 项（N8/N11/N19/N22）= 25/25 全处置（100%）。**

### 7.2 peft §8 成稿后新增的数据（须并入正文，peft §8 未含）

| 新数据 | 值 | 影响 | 落点 |
|:---|:---|:---|:---|
| Phase3-D 实测噪声底（8B） | N=max(N_seed 0.30, N_rep 4.09)=**4.09pt**（9run product 配置 gsm8k-flex，mean .4939 std .0129） | 替换 peft §8 的「0.9–4.5」估计为实测；§8.5 L7 噪声区间锚定 | §8.5 L7 + §4.3 |
| B11 14B 本地种子噪声 | **N_seed(14B,adaq)=14.71pt**（flex .6588/.5118/.5603）>> 8B 代理 4.09，且 > 信号 Δ8.87 | **14B「正向」被自身种子噪声淹没**；B1 下修为「方向一致但量级不稳健」；B5 chat-template 成决定性实验 | §4 + §8.5 L8（重写）|
| F18 真 INT4-RTN-g128 锚 | 对称 g128 round-trip，权重 rel_err mean 0.123（吻合理论 0.124）；评测排队（bf16 后 GPU0） | PTQ 表补真 RTN 行（对照 base_nf4dq_noadapter）；坐实「baseline_nf4 非 RTN」命名更正 | PTQ 表 + eval-protocol §9 |
| B5 chat-template 复评 | 排队（B8 后 GPU1，6run adaq+qlora×3seed + --log_samples） | 定 14B flex 信号是否格式 artifact；出数前 14B 差值禁进 abstract/intro/conclusion | §4.6（待 B5）|

### 7.3 P0/P1/P2 修复清单状态（peft §10.2）

**P0（不改则拒稿）**：P0-1 撤误差驱动叙事 ✅b0d4dde · P0-2 方法公式对齐 ✅§8.3+📝outline:84 · P0-3 训练预算披露 ✅eval-protocol§9+🔄Phase3-A重测 · P0-4 AdaLoRA 预算 🔄Phase3-B · P0-5 14B 预算对齐 ✅披露b0d4dde（重训=额外GPU未排）· P0-6 撤−26.5% ✅b0d4dde · P0-7 PARA/AQLoRA ✅B10 · P0-8 秩预算扫描 🔄Phase3-C · P0-9 理论降级 ✅§8.3/§8.5+📝outline:91/theory-report

**P1（影响可信度）**：P1-1 消融披露策略塌缩 📝 · P1-2 噪声底改区间 ✅（实测 4.09/14.71）· P1-3 主表补 flex 列 📝（数据已在手）· P1-4 删 strict 优势主战场 ✅b0d4dde · P1-5 Params 列真值 ✅b0d4dde · P1-6 14B 三种子齐 ✅REPORT_14B · P1-7 主表可复现 🔄Phase3-A+✅Phase2入库 · P1-8 显存重测 ✅F21修+🔄新臂重测 · P1-9 PPL n=62 处置 ✅§8.5L9 · P1-10 A1/A3 升格 📝 · P1-11 图资产 📝（epsilon_distribution.png 已有）

**P2（提升竞争力，返修期）**：P2-1 **成本归一化分配 r_l∝score_l/(d_in+d_out)**（peft 称「最便宜的正向实验」「阴性→诊断+修正升级关键」，⏳ 额外GPU 未排，见 §7.4）· P2-2 per-site Scheme B 跑一次（⏳ 额外GPU）· P2-3 匹配步数重跑主表 🔄Phase3-A · P2-4 PARA 式事后 SVD 对照（⏳）· P2-5 ≥6 种子（⏳）· P2-6 8-bit 鲁棒性（⏳，机制预测性验证）

### 7.4 ★P2-1 成本归一化分配——B11 后仅存的正向路径（须用户决策）

peft §12 总评 + §10.2 P2-1 指出：本文真正可发表资产是「机制 + 收敛阴性 + 功效边界」，但若加 **P2-1 成本归一化分配**（r_l ∝ score_l/(d_in+d_out)，等参数口径对比 uniform）拿到一个正向修正，论文可从「阴性报告」升级为「**诊断 + 修正**」双贡献，达 Q1 分析型期刊或 TMLR 可投水平。

**为何 B11 后这条更重要**：14B「正向」已被本地噪声（14.71pt）淹没，全文几乎无稳健正向信号。P2-1 是 peft 认定的「最便宜的正向实验」（8B 会把 q:v 从 12:4 翻到约 4:6，等参数口径），是**唯一可能给论文一个正向贡献的低成本实验**。成本：~3 臂（cost-norm/uniform/adaq）×2-3 种子 8B 训练+评测 ≈ 与 Phase3-D 同量级（~6-10h GPU）。

**决策点**：是否在 Phase-3（D→B→E→C→A）之后追加 P2-1？若追加，论文有「诊断+修正」双贡献上探 Q1/TMLR；若不追加，论文是纯「机制+阴性+功效边界」四件套，投 TMLR/Neural Networks/Neurocomputing（peft §11 评 ★★★★☆，但 TMLR 非 JCR 无影响因子，须与单位 JCR 要求核对）。

### 7.5 达小修剩余路径（综合）

1. **GPU 自动**（~40h，无需干预）：Phase3 E→C→A 出数 → 解 N8/N11/N19/N22 + P0-4/P0-8/P1-7；bf16/F18/B8/B5 链式自动跑 → 解 N1 锚点/F18 PTQ 锚/B8 flex 审计/B5 14B 信号真伪。
2. **数据到后写作**（📝，零GPU）：按 §7.1-7.3 把 peft §8.1-8.8 英文处方落进 paper-outline.md 正文（Abstract/贡献/§3.3/§5/Limitations/Conclusion/标题），补 P1-1/P1-3/P1-10/P1-11 图表，N13/N14/N15 的 outline/theory-report 数值订正。
3. **用户决策**：P2-1 成本归一化追加与否（§7.4，决定上探 Q1 还是稳投 TMLR/Q2）。
4. **撞车优先核**（非GPU 人工，投稿前）：RA-LoRA/Bayesian-LoRA/QuAILoRA 读全文给精确 delta（related-work-search-log §5.4）。

**结论**：零GPU 能做的处置已 100% 覆盖（25/25 N 项 + P0/P1 状态全标）。达小修的硬阻断 = Phase-3 + B5 的 GPU 墙钟（~40h+），出数后按 §7.5-2 落正文即达小修区间（peft 评「按 §8 骨架改完可达 Minor」）。P2-1 是上探 Q1 的可选正向实验，待用户拍板。

## 8. B8 flex 抽取器审计结果（2026-11-17 执行，F8/P0-4 闭合）

- 双臂 κ(A vs B)=0.7015(adaq)/0.7535(qlora)，均 **FAIL<0.95** → flex 抽取器与模型意图判断系统性不一致，flex 指标本身不可靠。
- **双臂差异抽取偏差**：flex 漏抽率（输出含 gold 但抓错）adaq 18.0%(144/800) vs qlora 26.5%(212/800) → flex 对 qlora 压得更低 → **14B flex +8.87pt 差值方向上被抽取偏差放大**，在 B5 chat-template 出数前不得作为任何方向证据。
- 失败机制实证（doc_id 0）：instruct 无 chat-template 生成后泄漏 fewshot 续写（"### Human:" 污染区），flex 正则抓到 '2.' 而非意图答案 $18；strict==gold 仅 .1775/.1200（instruct 不输出 #### 格式）。
- 处置：14B 所有 flex 差值禁入 abstract/intro/conclusion（与修正案 4414d9e R1-pending-B5 一致）；B8 全套数字见 docs/b8-flex-audit-results.md；评测口径节须写入「flex 抽取器审计不通过（κ=0.70/0.75）」。
- 诚实边界：双标注为机械规则（A保守=flex==gold且gold以数字边界现于输出；B宽松=A或末300字符含答案表述），κ 是可标注性代理非人工双盲；人工复核可改 annotA/annotB 列重跑 kappa。

