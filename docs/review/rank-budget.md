# AdaQ-LoRA 投稿前评审 —— 秩预算分配与组合优化

- **评审范围**：低秩适配的秩预算分配机制——误差驱动分配规则、离散约束下的预算守恒、预算规模消融与策略对照、启发式 vs 最优解的界定。
- **日期**：2026-11-16　**项目**：/home/ubuntu/AdaQ-LoRA　**评审人**：秩预算分配与组合优化专家
- **Verdict**：**Major Revision**（分配算法本体 sound；但实际训练的消融矩阵存在预算不守恒、参数量不匹配、α/r 混杂、重复臂四类缺陷；paper-framework.md 现行贡献措辞与实测数据矛盾，claim-revision 提案方向正确但尚未覆盖本评审新发现的公平性问题）

---

## 0. 证据基础（全部实读/实跑，非转引）

| 证据 | 来源 | 核验方式 |
| --- | --- | --- |
| 分配算法 | `adaq-lora-research-pipeline/scripts/adaq_precompute.py:20-83`（水位法）、`:86-127`（score 计算） | 全文实读 |
| 消融策略生成 | `scripts/generate_ablation_configs.py:24-62`（六策略 score）、`:91-99`（方案A聚合） | 全文实读 |
| 训练配置装配 | `scripts/train.py:54-110`（adaq 分支）、`:87-89`（方案A 强制 alpha_pattern=None）、`:117`（--budget） | 全文实读 |
| 12 份训练产物 | `outputs/ablation_adapters_schemeA_s{2025,3407}/{6策略}/adapter_config.json` | 逐份 json.load 实测 rank_pattern |
| 秩文件 | `ranks.json`（8B, 64 层）、`ranks_qwen14b.json`（14B, 96 层） | layer_stats 全量统计（CV/比值/恒等式） |
| 消融配置 | `configs/ablation_ranks/ranks_{6策略}.json` | per_layer_sum 与方案A types 实测 |
| 启动方式 | `scripts/run_ablation_schemeA.sh:35`、`run_ablation_schemeA_multiseed.sh:33`、`run_b4_qwen14b.sh:36`、`run_b4_qwen14b_seeds.sh:28` | 全部 `--rank-scheme per_module_type` |
| 结果 | `results_v2_ablation_multiseed/REPORT.md`、`results_v2_qwen14b/summary.json`、`docs/claim-revision-norm-aware.md` | 全文实读 |
| 自测 | `adaq_precompute.py --self-test` | 本次评审实跑，通过（守恒 512、边界 [4,20]、单调违反 0、Spearman 0.9836） |

**本评审关键实测数字**（后文引用）：

- 8B rel_error：q_proj CV=1.54%，v_proj CV=1.40%，**跨类型均值比 rel_q/rel_v = 1.000**；importance 跨类型比 imp_q/imp_v = 3.478。
- 恒等式核验：max|score − rel_error×importance| = **0.000e+00**（8B、14B 均成立）。
- 方案A 有效总秩（32 层/类型 × rank_pattern）：product 512、importance_only 512、error_only 512、uniform 512、**equal_weight 448（−12.5%）**、**random 480（−6.25%）**；14B adaq（48×7+48×4）= **528（+3.1%）**。
- 14B：q_proj rel CV=5.29%（max 0.1227 vs mean 0.0943，存在离群层）；per-layer q ranks ∈{4,5,6,7}（sum 317, mean 6.60, **median 7** → 方案A 超额 +16 的直接来源）。
- 参数量（8B，按 r×(d_in+d_out)×32 层，q:8192、v:5120）：product/importance 3,801,088；uniform/error_only 3,407,872；equal_weight 3,276,800；random 3,145,728。product 比 uniform **多 11.5%**、比 random **多 20.9%**。
- 14B 参数量实测（summary.json）：adaq 4,620,288 vs qlora 6,291,456（−26.5%），与 48×(7×10240+4×6144) 手算一致。

---

## 1. 分配形式化是否 sound（冻结量化 → 预计算 score → 零搜索）

### 1.1 逐层水位法本体：sound ✅

`allocate_ranks`（adaq_precompute.py:20-83）：

- **终止性**：while 循环每轮至少把一层永久移入 fixed（clamp 触发即 discard，不回迁），≤n 轮终止。
- **守恒精确性**：自由层 int() 截断后用最大余数法补足缺口（:67-79），且 :81-82 双 assert（总量==budget、全部 r∈[r_min,r_max]）——**失败是响亮失败，不会静默违约**。实测 ranks.json 与 ranks_qwen14b.json 的 per_layer_sum 均恰为 512。
- **取整损失**：逐层为 **0**（最大余数法精确补齐）；每层与实数解偏差 <1，单调性仅允许 ±1 并列（self-test :153-159 用相邻对检查，对全序足够）。
- **可行性预检**：:28-31 检查 n·r_min ≤ B ≤ n·r_max，正确。
- 小瑕疵（不阻塞）：若某轮后 free 非空但 score_free==0（全零 score 自由层），:43 break 后 :65 会 ZeroDivisionError；真实数据 score>0 不触发，建议加显式报错。

**结论：作为"离散约束下预算守恒的整数分配器"，算法与实现是 sound 的，且有可复跑 self-test 佐证。这部分可以作为工程贡献写进论文。**

### 1.2 致命代数恒等式：score ≡ ‖E‖_F，"乘积规则"名不副实 ❌

adaq_precompute.py:119-122：

    rel = ‖W_q − W_fp‖_F / (‖W_fp‖_F + 1e-8)
    score = rel × ‖W_fp‖_F  ≡  ‖W_q − W_fp‖_F   （本评审实测差值恒为 0）

即 **product 策略就是"绝对重构误差范数"单信号**，不是"误差 × 重要性"双信号交互。"乘积"只是把 ‖E‖ 拆成 rel×norm 再乘回去的伪装参数化。由此：

- `docs/paper-framework.md:92` 把 exp2-a3（product）写作 **‖E_l‖_F·‖W_l‖_F —— 与代码不符**（代码是 rel×‖W‖ = ‖E‖，不是 ‖E‖·‖W‖ = rel×‖W‖²）。方法表描述了自己的方法却是错的。
- `scripts/generate_ablation_configs.py:5-9` docstring 把 error_only 写作 r∝‖E_l‖_F——**代码实际用 rel_error（:31），即 ‖E‖/‖W‖**。标签与实现相反：真正的 ‖E‖_F 策略恰恰是 product。
- 叠加 rel_error 平坦（8B 跨类型比 1.000、层内 CV≈1.5%），有 ‖E‖ ∝ ‖W‖：**该规则在 NF4+Llama-3-8B 上数学等价于范数感知分配**。claim-revision E2（Spearman 0.998 vs 0.144）与本评审独立实测一致，此结论成立。

### 1.3 预算守恒：逐层成立，**实际训练口径（方案A）被打破** ❌（本评审新发现，claim-revision 未覆盖）

链路：per-layer 分配（守恒 512，assert 保证）→ **按模块类型取 median 聚合**（adaq_precompute.py:207、generate_ablation_configs.py:99）→ train.py 直接消费 type_ranks，**无任何再校验**（train.py:117 的 --budget 仅在 :266 回写 training_stats.json，是装饰性参数）。

median 聚合不守恒的实测结果：

| 配置 | rank_pattern | 有效总秩 | Δ vs 512 |
| --- | --- | --- | --- |
| product / importance_only | q=12, v=4 | 512 | 0 |
| error_only / uniform | q=8, v=8 | 512 | 0 |
| **equal_weight** | q=10, v=4 | **448** | **−64（−12.5%）** |
| **random** | q=7, v=8 | **480** | **−32（−6.25%）** |
| **14B adaq（seed42 已训练）** | q=7, v=4 | **528** | **+16（+3.1%）** |

14B 的 +16 来源清晰：q per-layer mean=6.60 而 **median=7**（分布左偏，含 r=4 离群层），48×(7−6.60)≈+19→取整后 +16。**median 是有偏聚合算子，任何非对称 per-layer 分布都会破坏守恒。**

因此 `docs/paper-framework.md:128`"adaq_precompute.py 的迭代水位法保证截断后总秩守恒"这句话**对 per-layer 为真、对全部已训练实验（方案A）为假**：equal_weight/random 两臂是在少 12.5%/6.25% 预算下训练的，14B 主对比是在多 3.1% 预算下训练的。"公平对齐的实验验证"（paper-framework.md:21 贡献(c)）以当前产物**不成立**。

### 1.4 秩预算 ≠ 参数预算：更深的公平性缺陷 ❌

q_proj（4096×4096）与 v_proj（1024×4096）维度不同，"总秩相同"不等于"参数量相同"：product {12,4} = 3.80M 参数 vs uniform {8,8} = 3.41M vs random {7,8} = 3.15M。**秩预算守恒在异维模块下不是资源守恒的正确不变量。**

后果两面：

1. product 臂比 uniform 多 11.5% 参数仍打平 → **阴性结论反而更硬**（多给参数都没赢）；
2. 但论文**不得**在 8B 上做任何"参数效率"表述；14B 的 "−26.5% 参数" 是事实陈述（4.62M vs 6.29M，实测吻合），可用，但须同时披露其秩口径为 528≠512。

**修复方向**：增加 param-matched uniform 基线（解 32×r×13312 = 3,801,088 → r=8.92，取 r=9，3.83M，与 product 差 <1%），或把预算不变量改为 Σ r_l×(d_in+d_out)_l 并在 allocate_ranks 中以维度加权分配。

### 1.5 r_min/r_max 截断影响：r_min 全程绑定、r_max 从未绑定 ⚠️

- **r_max=128 从未生效**（8B per-layer 最大 13，14B 最大 7）——形同虚设，论文不必渲染"截断再归一"的复杂度，或降低 r_max 到有意义区间（如 32）再谈。
- **r_min=4 绑定几乎全部 v_proj 层**：无 clamp 的比例解为 r_q≈12.43、r_v≈3.57（8B，由 imp 比 3.478 导出）；r_v<4 被抬到地板，v 类共占 128/512=25% 预算。即 **(12,4) 的形状主要由 r_min 决定，而非 score 的精细结构**——"误差驱动分配"的叙事进一步被削弱为"地板约束 + 维度比例"。
- **缺失 r_min 敏感性分析**：r_min∈{2,4,8} 只改预计算即可得到三套 rank_pattern（无需训练），必须补，否则审稿人一句"你的结论是不是 r_min 的 artifact"无法回答。

### 1.6 α/r 混杂：方案A 全部训练臂 α=16 固定 ⚠️（本评审新发现）

train.py:87-89 在 per_module_type 下强制 alpha_pattern=None → 所有臂 lora_alpha=16（12 份 adapter_config.json 实测 alpha_pattern={}）。于是各臂各层的实际缩放 α/r 不同：product q=1.33 / v=4.0，uniform=2.0，equal_weight q=1.6 / v=4.0，random q=2.29 / v=2.0。train.py:103-104 的注释自己承认"非均匀秩下 α 须随 r 缩放（α_l=2·r_l）"，但方案A 分支刻意跳过（"严格复刻主实验早期行为"）。

**消融对比因此混杂了两个因子：秩分配 × 逐层有效学习率缩放。** REPORT.md 观察到的"product gsm8k std 最小（±.0032）"完全可能是 v_proj α/r=4.0 的缩放效应而非分配的稳定性，在做出 α 对照（修复#5）之前不得归因于分配规则。

### 1.7 启发式 vs 最优解：必须明确降格 ⚠️

水位法"score 比例分配 + clamp + 最大余数"是**启发式**：它只在"逐层效用为对数型（或等价凹形）"的假设下才是约束最优，而项目**没有测量任何逐层 ε_l(r) 衰减曲线**，理论部分（Weyl 扰动界 + 谱抬升）给的是界与单调性，不是分配最优性。论文措辞上限：**"an exactly budget-conserving, training-free proportional heuristic"**；禁止任何 "optimal / optimality" 表述，除非补做 ε_l(r) 曲线测量并证明比例解与凹优化解一致。

---

## 2. 消融矩阵有效性（六策略能否支撑任何"分配有效"结论）

### 2.1 六策略实为 4 个不同配置（12 份 adapter_config.json 实测）

| 策略 | rank_pattern（两种子完全一致） | 有效总秩 | 参数量(8B) | 备注 |
| --- | --- | --- | --- | --- |
| product | q=12, v=4 | 512 | 3.80M | 与 importance_only **逐字段相同** |
| importance_only | q=12, v=4 | 512 | 3.80M | 与 product **逐字段相同** |
| error_only | q=8, v=8 | 512 | 3.41M | 与 uniform **逐字段相同** |
| uniform | q=8, v=8 | 512 | 3.41M | 与 error_only **逐字段相同** |
| equal_weight | q=10, v=4 | **448** | 3.28M | 预算违约 |
| random | q=7, v=8 | **480** | 3.15M | 预算违约 + 未播种 |

**"六策略消融"实际只有 4 个配置点，其中 2 对是同一配置的重复训练。** 论文若按 6 行报表即构成对读者的隐性误导（行数暗示自由度），必须显式标注两对重复并把它们改述为"噪声底校准对"。

### 2.2 product ≡ importance_only 是结构性的，不是巧合

比例分配对 score 的逐层常数倍不变：rel_error 近似常数（CV 1.4-1.5%、跨类型比 1.000）⇒ score_product = rel_l×imp_l ≈ c×imp_l = c×score_importance ⇒ 实数解逐层相同；残余 ±1 取整差又被 median 聚合完全吸收。**该消融对在 NF4+Llama-3-8B 设置下检验"误差项贡献"的统计功效恒等于零**——不是样本量问题，是设计不可识别问题。

由此回答问题③："product 与 importance_only 同 rank_pattern 还叫消融吗"——**不叫**。它的正确身份有两个，二选一或并用：

1. **退化性证据（推荐，零成本）**：把"两臂相同"升格为 E2 的机器可验证证明——在 generate_ablation_configs.py 里加重复检测守卫（修复#4），论文中写 "the product and importance arms produce bit-identical allocations because CV(rel_error)=1.5%; this identity is itself our central diagnostic finding"。
2. **可识别性实验（需补做）**：要让"误差项是否贡献"成为可检验命题，必须换到 rel_error 有真实方差的 regime：(a) RTN-INT4 或 mixed-bit 量化（误差谱异质）；(b) Hessian/激活加权误差 ‖E_l X_l‖（OWL 类代理，层间方差大）；(c) 14B q_proj 已有苗头（rel CV 5.29%、max 0.1227 离群层）——在方案B（per-layer）下 product 与 importance_only 的分配会开始分化，可做纯分配对比（不训练也比什么都不做强）。

### 2.3 error_only ≡ uniform：塌缩已被内部认知，但处理方式要改

rel_error 平坦 ⇒ score 近似常数 ⇒ 分配塌缩为 uniform（实测两臂 {8,8} 相同）。作为训练臂它浪费 GPU；作为证据它有用（E2 的第二条腿）。**建议：不再作为独立训练臂，改为分配层面的解析结论 + CV 数字报告**；已跑的两种子数据改标为 uniform 的重复对，并入噪声底估计。

### 2.4 random 臂：三处缺陷

1. `_rnd.random()` **未播种**（generate_ablation_configs.py:57），分配不可复现，仅靠已落盘的 ranks_random.json 幸存；docstring 声称"多 seed 平均"未实现（两种子共用同一 pattern）。
2. median 聚合后 random 退化为 {7,8}——**近似 uniform 且预算少 32**，不是有意义的"随机分配下界"。真正的方案A随机臂应在守恒约束下采样：32(r_q+r_v)=512 ⇒ r_q+r_v=16，r_q~U{4..12}，如 {5,11}/{9,7}/{11,5}。
3. 当前 random 与 uniform 的差异（1 个秩单位 + 32 预算差）远小于噪声底，该臂提供的信息≈0。

### 2.5 equal_weight 臂：概念与公平双重问题

z-score 等权（generate_ablation_configs.py:38-50）把 CV=1.4% 的 rel_error 除以自身 std，**z(rel) 是被放大到 O(1) 的纯噪声**，与 z(imp) 等权相加后 score 的相当部分由噪声驱动；再叠加预算 448（−12.5%）。该臂既不能解释"等权组合"也不能公平对比。若保留必须：修聚合守恒（修复#1）+ 在文中说明 z(rel) 的噪声放大；否则删除。

### 2.6 噪声底方法学：方向对，估计器弱

现有 4 对同配置重复（product/imp ×2 种子、error/uniform ×2 种子）的 flex 差：**4.53、1.75、0.08、0.83 点**。REPORT.md 取最大值 4.5 作为噪声底——对"不可区分"结论是保守方向（安全），但 n=4 的 max 是 ad-hoc 估计器，且 4.53 与 0.08 的悬殊本身说明跑间噪声重尾。六策略极差 2.65 点 < 4.53 ⇒ 阴性结论成立；但若用中位数 ~1.3 做底，importance_only(.4989) 与 error_only(.5254) 的 2.65 点差反而"显著"——**这正是不能拿单点统计量当噪声底的原因**。修复：设专门重复臂（uniform × ≥4 种子），报告跑间方差的分布（min/median/max），预注册 flex 为主指标、按种子配对分析。

### 2.7 问题①结论：每个消融实验的贡献与总判定

- **product 臂**：贡献 = 完整方法的代表配置 +（与 importance 合并后）噪声底校准对。不支持任何"有效"结论。
- **importance_only 臂**：贡献 = E2 的实证（与 product 相同配置这一事实本身）。作为独立策略对照无增量信息。
- **error_only 臂**：贡献 = rel_error 平坦性的第二实证。同上。
- **uniform 臂**：必要基线，贡献成立；但需补 param-matched 版本（r=9）才能回应"product 多 11.5% 参数"的质疑。
- **equal_weight 臂**：当前形态贡献为负（预算违约 + 噪声放大概念缺陷），修后重跑或删。
- **random 臂**：当前形态贡献≈0（近似 uniform、预算违约、不可复现），必须按 2.4 重造。

**六策略设计能否支撑"product 有效"？——不能，且永远不能（在 rel 平坦的 regime 下 product 与 norm 臂是同一配置，"有效"命题不可识别）。能否支撑"任何分配有效性"？——8B 上不能：4 个不同配置的 flex 极差 2.65 点 < 同配置跑间噪声（0.08–4.53 点），叠加预算/参数量/α 三重混杂，方向性结论无一可立。能支撑的是：(a) 诊断性阴性结论（claim-revision 已正确表述）；(b) "同预算下范数感知分配不优于均匀/随机"这一负面陈述——但须先修复 §1.3/§1.4/§1.6 的公平性缺陷或如实披露，否则连阴性结论都会被审稿人以"你的消融不公平"打回。**

---

## 3. 零搜索预计算 vs 可学习分配（AdaLoRA）：定位与诚实措辞

### 3.1 定位判断

两条正交轴：**信号来源**（量化伪影 vs 训练梯度/敏感度）与**时机**（训前一次性 vs 训练中动态）。AdaQ 占位"量化伪影 + 训前零搜索"；AdaLoRA（train.py:47-53：init_r=16→target_r=8，tinit=20/tfinal=60/deltaT=5）占位"训练敏感度 + 动态剪秩"。当前定位错误风险在于把 AdaQ 写成 AdaLoRA 的竞品——**8B 证据显示它在质量轴上无增益，唯一诚实的差异化卖点是成本轴**：无校准数据、无梯度、无训练循环耦合、确定性可复现、离线一次完成。14B 单种子正向（flex .6588 vs .5254，参数 −26.5%）是"规模依赖迹象"，seed2025/3407 未出（results_v2_qwen14b_seed2025/summary.json 为空），在此之前维持 claim-revision §4 的措辞上限不变。

### 3.2 贡献句的精确改法（问题④答案）

paper-framework.md:21 现行三条贡献全部需要重写：

- (a) "量化误差驱动的分配规则" → **不成立**（§1.2：score≡‖E‖且 rel 平坦 ⇒ 实为范数驱动）。改为：
  > "A training-free, exactly budget-conserving integer rank allocator (iterative water-filling with largest-remainder rounding), together with the diagnostic finding that under NF4 the quantization-error signal degenerates: score = rel_error × ‖W‖_F ≡ ‖E‖_F, and since CV(rel_error)=1.5% with rel_q/rel_v=1.000, the allocation is norm-driven, not error-driven."
- (b) 理论贡献保留，但补一句边界："the proportional rule is a heuristic; optimality would require per-layer ε_l(r) decay curves, which we do not measure"。
- (c) "公平对齐的实验验证" → **当前产物不支持**（预算 448/480/528、参数 3.15–3.80M、α/r 1.33–4.0 三重不对齐）。改为：
  > "A negative, replication-anchored empirical result: across 2 seeds × 2 tasks, four distinct allocations (six nominal strategies, two pairs bit-identical) are indistinguishable under a noise floor estimated from identical-config replicate pairs (0.08–4.53 pts flex); we additionally quantify three fairness confounds (budget, parameter count, α/r scaling) that bound any future positive claim."
- 新增 (d)（工程/方法学，可辩护）：重复臂自动检测守卫、self-test、噪声底校准协议——这些是本项目真正独有的可复用产出。

**与可学习分配的关系句**（放 Related Work 结尾）：
> "AdaQ is complementary to learned allocation (AdaLoRA): it trades adaptivity for zero search cost and determinism. Our 8B results show this trade buys no quality under NF4 when the error proxy degenerates; a learned allocator remains the recourse when static proxies are uninformative, and the 14B single-seed signal suggests the trade-off may be scale-dependent — a hypothesis we state but do not test."

---

## 4. 最小可辩护消融集（问题⑤答案）

### 4.1 臂的取舍

| 处置 | 臂 | 理由与规格 |
| --- | --- | --- |
| **保留** | uniform-r8 | 基线；必须补 **param-matched uniform-r9**（3.83M≈product 3.80M）作为公平基线 |
| **保留+改名** | norm_only（原 importance_only） | 真实驱动信号的直接代表 |
| **保留+改标签** | abs_error（原 product） | 明示 ≡‖E‖_F；与 norm_only 相同配置这一事实作为退化性证据报告（一对重复保留即可，不再多种子重复训练） |
| **重造** | random-conserving | 播种 RNG；方案A下按 r_q+r_v=16, r_q~U{4..12} 采样；或方案B per-layer 随机 score 过 allocate_ranks（天然守恒） |
| **新增** | noise-floor 重复臂 | uniform-r8 × ≥4 种子，产出跑间方差分布，替代 max-of-2-pairs |
| **新增（廉价）** | α 对照 | product × {α=16 固定, α=2r 缩放} 各 1 种子，界定 §1.6 混杂的量级 |
| **新增（零训练）** | r_min 敏感性 | r_min∈{2,4,8} 三套 rank_pattern 对比表，回应 §1.5 |
| **新增（可选）** | 异质误差 regime | RTN-INT4 或 Hessian 加权 score 的分配对比（先只比 rank_pattern，不训练），让"误差项贡献"首次可识别 |
| **删除（训练臂）** | error_only | 解析上 ≡ uniform（CV 1.4%），改为文字+CV 数字报告；已跑数据并入噪声底 |
| **删除或修后重跑** | equal_weight | z(rel) 噪声放大 + 预算 448；若保留必须先修聚合守恒并在文中说明构造 |

新增训练成本估算：random 重造 2 种子 + uniform×4 + α 对照 2 + param-matched uniform 2 ≈ **10 次 8B 训练**（seq1024×2000 样本，单卡小时级），换来消融矩阵从"4 配置 3 混杂"升级为"可辩护"。

### 4.2 统计与报告协议

1. 预注册 flex 为主指标（strict 仅作格式合规性附注——REPORT.md 已有此结论，采纳）。
2. ≥3 种子/臂，按种子配对报告差值与符号反转，不做无配对的 mean±std 排名。
3. 所有表加三列：有效总秩（Σ 实测）、参数量、α/r 集合——公平性自证。
4. 任何"A 优于 B"句子的前置条件：配对差 > 噪声底分布的 max，否则一律写 "indistinguishable within run-to-run noise"。
5. generate_ablation_configs.py 输出前做 pairwise distinctness 检查，重复即写入 identical_to 字段并打印 WARNING（防止再次静默训练重复臂）。

---

## 5. 编号修复清单（路径 + 精确改法）

1. **`scripts/generate_ablation_configs.py:91-99` 与 `adaq-lora-research-pipeline/scripts/adaq_precompute.py:199-207`**：median 聚合改为守恒聚合——先取类型均值，再在类型层做"缩放 + 最大余数取整 + [r_min,r_max] clamp"的整数水位，使 Σ n_type×r_type == budget；写盘前 assert。修后重新生成 equal_weight（448→512）与 random（480→512）配置。
2. **`scripts/train.py:117,168-170`**：--budget 从装饰参数变为强制校验——build_lora_config 返回后，按 PEFT rank_pattern 匹配规则计算有效总秩（逐层求和），!= args.budget 时 fail-fast（或 --allow-budget-drift 显式豁免并把 delta 写入 training_stats.json）。
3. **`scripts/generate_ablation_configs.py:55-57`**：`_rnd.seed(args.seed)`，新增 --seed 参数并把种子与生成时间写入输出 JSON 的 meta；否则 random 策略永久不可复现。
4. **`scripts/generate_ablation_configs.py` main()**：生成全部策略后做 pairwise rank_pattern 去重检查，重复对写入各 JSON 的 `identical_to` 字段 + 打印 WARNING（落实 §4.2-5 守卫）。
5. **`scripts/train.py:87-89`**：方案A 分支补 alpha_pattern={mod: 2r}（与 :103-104 注释一致），或加 --alpha-mode {fixed,scaled} 且默认记录进 training_stats.json；已完成的 12+2 个训练臂在论文表格中逐臂标注 α/r 实际值。
6. **`scripts/generate_ablation_configs.py:4-11` docstring 与 `docs/paper-framework.md:92`**：修正策略公式——error_only: r ∝ rel_error = ‖E‖_F/‖W‖_F；product: r ∝ rel_error×‖W‖_F **≡ ‖E_l‖_F**（绝对误差）；删除任何 "‖E‖·‖W‖" 写法。
7. **`docs/paper-framework.md:12`（摘要段）**：删除"NF4 对不同模块类型的量化误差 ε_l 可相差数倍"（实测 rel_q/rel_v=1.000，该句为假）与"高误差层需要更大的适配容量"；"提升参数效率"删"提升"；:17 Gap-1 改"忽视权重范数的层间异质性"；:128 补一句"方案A median 聚合在 equal_weight/random/14B 上破坏守恒（448/480/528），训练口径的守恒声明仅对 per-layer 成立"，或按修复#1 改代码后删此顾虑。
8. **`docs/paper-framework.md:21` 贡献三条**：按 §3.2 的 (a)(b)(c)(d) 精确替换。
9. **重跑两个违约臂**：equal_weight、random（修#1、#3 后各 2 种子），并在 REPORT.md 主表加"有效总秩/参数量/α-r"三列；若算力不允许，表格脚注如实披露违约幅度与方向（equal_weight/random 少预算仍不劣 ⇒ 阴性结论方向不变，但必须写明）。
10. **`results_v2_ablation_multiseed/REPORT.md` 噪声底段**：把单点 4.5 改为 4 对重复的完整分布（4.53/1.75/0.08/0.83），注明 max 作底的保守性与 n=4 局限；补 uniform×4 种子重复臂计划。
11. **14B 披露**：`docs/claim-revision-norm-aware.md` §4 表格下加一行："方案A 有效秩预算 528（+3.1%），参数量 4.62M 为实测事实；seed2025/3407 完成前维持措辞上限"。14B q_proj rel 离群层（max 0.1227, CV 5.29%）值得在方案B per-layer 下复查 product vs norm 分化——这是现成的可识别性素材。
12. **`adaq_precompute.py:41-45,63-65`**：free 非空且 score_free==0 时抛显式 ValueError（当前路径 ZeroDivisionError）；self_test() 补一条"score≡rel×importance 恒等式"断言与 r_min 敏感性打印，把 §1.2/§1.5 的诊断固化进可复跑测试。

**优先级**：#6/#7/#8（措辞与事实矛盾，投稿前必须）> #1/#2/#5/#9（公平性，审稿人可复算发现）> #3/#4/#10/#11/#12（可复现性与稳健性加固）。

---

## 6. 三句话总评

1. 分配器本体（水位法 + 最大余数 + 双 assert + self-test）是本项目最扎实的资产，逐层口径下预算精确守恒、取整零损失，可直接作为工程贡献；但训练全部走了方案A，median 聚合把守恒打破（equal_weight 448、random 480、14B 528），叠加秩预算≠参数预算（product 比 uniform 多 11.5% 参数）和 α=16 固定带来的 α/r 混杂，"公平对齐的实验验证"这条贡献以当前产物不成立。
2. 六策略消融实测只有 4 个不同配置（product≡importance_only、error_only≡uniform 均逐字段相同），且同配置是结构性的（score≡‖E‖_F、rel_error 跨类型比 1.000）——该矩阵在 NF4+Llama-3-8B 上对"误差项贡献"的检验功效恒为零，只能支撑诊断性阴性结论，任何正向"分配有效"表述都不可辩护。
3. 出路是把论文重心从"方法有效"彻底移到 claim-revision 已起草的"诊断 + 阴性 + 噪声底方法学 + 守恒分配器"四件套，并按 §5 清单修掉公平性三混杂（预算/参数量/α）、重造 random 臂、补 param-matched uniform 与 r_min 敏感性——14B 单种子正向只能作为"规模依赖迹象"假设陈述，seed2025/3407 出数前不得升格。
