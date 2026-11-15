# AdaQ-LoRA 投稿前架构评审（PEFT / 论文结构方向）

> 评审人角色：JCR Q1/Q2 PEFT 与投稿架构评审
> 评审日期：2026-11-16（远端仓库 `/home/ubuntu/AdaQ-LoRA`，git HEAD `220fd60`）
> 评审方式：**只读**。所有评分均绑定实读文件/实跑复算的证据；未读到的内容标注 GAP，不臆测。
> 交付边界：本文件为新建，未修改任何既有文档、脚本、结果或运行中的训练进程。

---

## 0. 证据来源清单（本次评审实读/实算，可复核）

**文档（全文读完）**
- `docs/paper-framework.md`（158 行）
- `docs/paper-outline.md`（179 行）
- `docs/claim-revision-norm-aware.md`（144 行，重构提案；:3 自述"提案，未改正文"）
- `docs/theory-review-report.md`（206 行，v3.0，2026-07-30）
- `docs/sota-comparison-protocol.md`（221 行）
- `docs/eval-protocol-v2.md`（119 行）
- `docs/training-optimization-plan.md`（140 行）

**结果产物（实读原始 JSON，非转述）**
- `results_v2/FINAL_REPORT.md`、`results_v2/summary.json`（含 `significance_test`、`_audit`）
- `results_v2/*/**/results_*.json` → **本次评审自行重算 12 run 的 GSM8K strict + flexible-extract 与 MMLU 逐种子值**
- `results_v2_ablation_multiseed/REPORT.md`、`results_v2_ablation_schemeA_s{2025,3407}/**/results_*.json`
- `results_v2_ptq_nf4/summary.json`、`results_v2_qwen14b/summary.json`、`results_v2_qwen14b_seed2025/{adaq,qlora}/**/results_*.json`（**seed2025 MMLU 为本次首次提取，任何文档均未记录**）
- `results_v2_ppl/summary.json`（`sample_len=62`）
- `results_v2/epsilon_l_report.json`（A1/A2/A3/H1 实测产物，**存在但三份写作计划文档均未引用**）
- `docs/figures/erank_vs_error.json`（命题 1' 数值验证产物；**同名 PNG 不存在**）

**代码与配置（实读，用于核对"方法节描述 = 实际执行"）**
- `adaq-lora-research-pipeline/scripts/adaq_precompute.py:86-127`（:115-122 score 计算式）
- `scripts/generate_ablation_configs.py:24-62`（六策略真实取分逻辑）
- `scripts/generate_synthetic_ranks.py:40-52`（ρ=0.9836 的来源）
- `scripts/train.py:124,132,173-174,206-230`（`--max-length` 默认 2048、`packing` 默认 True、`save_steps=10`）
- `scripts/run_ablation_schemeA_multiseed.sh:30-35`、`scripts/run_b4_qwen14b.sh:36-44`、`scripts/run_b4_qwen14b_seeds.sh:26-37`
- `scripts/run_eval_ablation.py:76`（消融评测 `--model_args`）
- `scripts/run_eval_ptq_nf4.py:4-8,110-118`
- `scripts/aggregate_b4_3seeds.py:97-101`（14B 预注册判定规则）
- `outputs/adapters/{qlora,dora,adalora,adaq}_seed{42,2025,3407}/adapter_config.json` + `training_args.bin` + `training_stats.json` + `checkpoint-*/trainer_state.json`（12 run 全查）
- `outputs/ablation_adapters_schemeA_s3407/*/trainer_state.json`、`outputs/adapters_qwen14b/*/adapter_config.json`
- `ranks.json`（8B，`layer_stats` n=64 全量复算）、`ranks_qwen14b.json`（n=96）、`configs/ablation_ranks/ranks_*.json`（6 份 rank_pattern 逐一对比）

**外部文献（本次联网核实）**
- AQLoRA: arXiv [2608.23816](https://arxiv.org/html/2608.23816v1)（2026-08）
- PARA: arXiv [2604.27796](https://huggingface.co/papers/2604.27796)（2026-04，Kumaravelu / Gupta / Srijith）

---

## 1. 结论（Verdict）

| 项 | 判定 |
|:---|:---|
| **按现大纲（`paper-outline.md` / `paper-framework.md`）直接成稿投稿** | **REJECT**（拒稿级：核心动机被自家实测数据否证 + 方法节公式与实现不符 + 一项"必须"实验缺失） |
| **按 `claim-revision-norm-aware.md` 提案改完再投** | **MAJOR REVISION**（提案方向正确但覆盖不全：漏掉 6 项阴性/矛盾证据，见 §7 覆盖矩阵） |
| **按本评审 §8 的重构骨架 + §9 修复清单改完再投** | 可达 **MINOR REVISION → 可投**（Q2/Q1 分析型期刊或 TMLR 型"claims 匹配 evidence"场刊） |
| **本次评审给出的总判定** | **MAJOR REVISION**（论文正文尚未动笔，返工成本可控；数据资产本身诚实、可复算、且具备外部独立佐证） |

**综合评分：3.6 / 10（现状态）**；**若采纳 §8 重构 + §9 P0/P1 全修：预估 6.8 / 10**

一句话总纲：**这不是一个"方法失败了"的项目，而是一个"方法的前提被自己证伪了、并且被另一组人独立证伪了"的项目——后者是可发表的，前者不是。现在的三份写作计划仍在前者的轨道上。**

---

## 2. 分章节评分表

评分口径：**claim 与 evidence 的匹配度**（不是文笔）。10 = 可直接投稿；7-8 = 小改；4-6 = 需重写该节；≤3 = 该节当前版本构成拒稿理由。

| # | 章节 | 评分 | 判定依据（实读证据） | 必须动作 |
|:--:|:---|:--:|:---|:---|
| 1 | **Abstract**（outline :16-21） | **2.5/10** | :17 问题陈述="uniform rank 忽视量化误差异质性"，被 `ranks.json` 复算 CV(rel_error)=1.48% 直接否证；:19 "error-driven allocation" 与 `adaq_precompute.py:115-122` 实现不符；:21 结论"参数效率增益"无同预算证据 | 用 §8.1 重写版整体替换 |
| 2 | **Abstract（提案替换稿）**（claim-revision :56） | **6.0/10** | 方向正确、阴性表述诚实；但未披露训练预算（21 步）、未提 H1/α 失败、未提 AQLoRA 独立佐证、未提 Scheme A 只有 2 组 | 补 4 项（§9 P0-3/P0-5/P1-2/P1-6） |
| 3 | **§1 Introduction**（outline :27-31；framework :10-12） | **3.0/10** | Gap-4"NF4 逐模块误差相差数倍"**实测 CV 1.48%（相差 8.9% 极差）**；Gap-1 需重锚到 importance CV 56.65%（真）；Gap-3"AdaLoRA 4-bit 重要性不可靠"从未被测，且 `sota-comparison-protocol.md:90` 预写"结果差是预期证据"= 确认偏误 | 重写 Gap-1/3/4；删 framework :12 整句 |
| 4 | **§2 Related Work**（outline :37-51） | **3.0/10** | 全仓 grep `PARA`/`AQLoRA` **零命中**（仅 `sota-comparison-protocol.md:22` 列了 AQLM/QuaRot）；PARA 与本文标题 "Adaptive Rank Allocation" 正面撞名；AQLoRA 的 NF4-误差选层≈随机对照实验是本文最强外部佐证，完全未用 | 按 §8.4 补 5 条 delta 句 + 引用 |
| 5 | **§3.1-3.2  preliminaries / E_l 定义** | **6.5/10** | A1 零均值**已实测成立**（`epsilon_l_report.json`: mean_of_layer_mean_e 9.81e-06，max_abs 1.78e-05，vs mean_abs_e 9.53e-04）→ 可从 ⚠️ 升 ✅；但 A2 的 ε₄≈0.02 经验值**实测为 0.0936（4.7×）**，theory-review-report :15 与 outline §3.2 仍写 0.02 | 升 A1；全文替换 ε 值；引用实测产物 |
| 6 | **§3.3 理论（Thm1 / Prop1' / Prop2 / Prop3a / Prop3b）** | **4.0/10** | Thm1（Weyl）✅ 成立；Prop1' 数值验证存在但**效应量 +0.45%@ε=0.02、+0.62%@ε=0.08（erank 1.2020→1.2096）**，且用 α=1.5 而实测 α 中位数 0.436；**Prop2 的前提 H1 实测失败：α∈[0.181,1.591]（8.8× 跨度）、r²_median=0.802**；Prop3b 的 ρ=0.9836 来自 `generate_synthetic_ranks.py:40-52` **硬编码合成数据**，outline :91 却标"实测 ✅" | Prop2 降为条件命题并**报实测违反**；Prop3b 改标"合成自测"，删"实测 ✅"；Prop1' 补真实 α |
| 7 | **§3.4 预计算与分配规则** | **3.5/10** | 实现 `score = rel_error × importance ≡ ‖E_l‖_F`（复算 max\|score−rel×imp\|=0.000e+00），而 outline :84 与 framework :89-93 写 `r_l ∝ ‖E_l‖_F·I_l`；**Scheme A 把 64 位点塌成 2 组 `{v_proj:4, q_proj:12}`**，"逐层自适应"实际只有 ~1 bit 信息；14B Scheme A 取整后 48×(7+4)=**528 ≠ 512**（破坏自家守恒断言 `adaq_precompute.py:81`） | 公式与实现对齐；正文披露 Scheme A 粒度；修 14B 预算 |
| 8 | **§4.1 实验设置** | **3.0/10** | 三张表用了**三种训练预算**：主表 `global_step=max_steps=21`（seq 2048 + `packing=True`）、消融 126（seq 1024）、14B 255（seq 512）；三份文档统一写"3 epochs / 2000 条 / seq 2048"从未提 21 步；主表训练命令**无脚本、无日志可复现**（`logs/` 无 train 日志，`.bash_history` 无记录）；`train.py:217 save_steps=10` 与主表 checkpoint-{2,7,14,21} 不符 → 代码已漂移 | 每表脚注报 `global_step`/`num_tokens`；补主表训练脚本入 git；标 commit |
| 9 | **§4.2 主表（4 方法×3 种子）** | **5.0/10** | **公平性 ✅**：12 run `training_args.bin` 完全一致（lr 2e-4/ep 3/mbs 4/ga 4/paged_adamw_8bit/bf16，distinct configs=3 即仅种子不同）；预算 ✅ 32×(12+4)=512=32×16；样本全量 ✅ 14042/1319；统计诚实 ✅。**但**：21 步优化 → 阴性结果被训练预算混淆；adaq 在三指标上**均非最优**（flex dora .4877 > adaq .4850；strict dora .4799 > adaq .4766；MMLU dora .6543 > adaq .6540）；`FINAL_REPORT.md` 只报 strict | 补 flexible 列；脚注报步数；措辞改"四方法在种子噪声内不可区分，DoRA 点估计最高" |
| 10 | **§4.3 消融（6 策略×2 种子）** | **4.0/10** | **6 策略实际只有 4 种不同分配**：product ≡ importance_only `{4,12}`、**error_only ≡ uniform `{8,8}`**（`configs/ablation_ranks/ranks_*.json` 逐份核对）；`generate_ablation_configs.py:31,33,35` 实际取分是 `rel_error`/`importance`/`score(=‖E‖)`，与其 docstring :5-9 及 framework :88-94 的 ‖E‖·‖W‖ 公式**三者不符**；"噪声底 4.5 点"取自 n=1 观测，另一对重复配置（error_only/uniform）实测差仅 0.08/0.83 点 → 噪声非齐性 | 表注披露策略塌缩与真实取分式；噪声底改为区间 [0.08, 4.5]；结论仍成立但需重写论证 |
| 11 | **§4.4 秩预算扫描 R∈{128..1024}** | **0/10** | `paper-framework.md:75-82` 标为**必须**（~50h），仓库无任何对应结果目录 | P0：补跑或删该节并说明；不可留"计划中" |
| 12 | **§4.5 效率/显存表** | **3.0/10** | `training_stats.json`：8B adaq 14.81 GB、消融 17.43 GB、**14B adaq 6.67 GB = qlora 6.67 GB**（14B 显存低于 8B，物理不可信，疑为度量点/口径错误）；预计算 <5 min 无 profile 产物 | 重测 14B 显存；效率表按同口径重跑 |
| 13 | **§4.6 跨模型泛化** | **2.5/10** | 仅有 14B，且 **adaq 528 rank vs qlora 768 rank（基线多 45.5%）**（`adapter_config.json` 反算，参数量 4,620,288 / 6,291,456 与公式精确吻合）；`run_b4_qwen14b.sh:2` 注释"同预算 512 rank"**与产物矛盾**；违反 `sota-comparison-protocol.md:42,78` | 只能支撑"更少参数下的平价"；重跑等预算臂或删"泛化"表述 |
| 14 | **§4.7 统计口径** | **8.0/10** | outline :137-142 与 `summary.json.significance_test`（57+3 任务全部标 "仅 3 对配对值；Wilcoxon 需 >=6 对才有效"）一致，明确禁写"显著/p<0.05/SOTA"，是全文最规范的一节 | 保留；补 bootstrap CI 与最小可检出效应量 |
| 15 | **§5 Discussion / Analysis** | **2.0/10** | outline :146-148"高 ε 层获得更高秩 → 验证方法直觉"**双重错误**：ε 平坦（CV 1.48%）且 Scheme A 下秩是按模块类型而非按层分配；图4/图5 均无成品 PNG | 用 §8.5 重写版替换；图4 改为 rel_error vs importance 双面板 |
| 16 | **§6 Limitations（L1-L5 + 提案 L6）** | **4.0/10** | 提案 L6/L3 改写到位，但**漏 6 项**：训练预算混淆（21/126/255）、H1/α 实测失败、Scheme A 粒度塌缩、14B 预算不对齐、ε 实测 0.094 vs 假设 0.02、Prop3b 合成来源 | 按 §8.6 的 L1-L10 替换 |
| 17 | **§7 Conclusion** | **2.5/10** | outline :162"量化误差驱动的秩分配，零额外开销"——前半句已被否证；后半句在 NF4 评测中 adaq 最差（.4064，五者最低且低于无 adapter 基线 .4124） | 用 §8.7 替换 |
| 18 | **图表资产** | **3.0/10** | `results_v2/epsilon_distribution.png` ✅ 存在；`docs/figures/erank_vs_error.json` 有数据但**无 PNG**；outline :169-179 规划的图1/2/3/5、表1-4 均无成品 | 出图前先定稿口径（三表不可混） |

**分章节均分：3.6/10**（18 节，最低 0，最高 8.0）

---

## 3. 问题①：动机是否被算法与方法论支持？（三处自洽性逐点核对）

**判定：不自洽，且不自洽点比提案承认的更深一层。** 提案只处理了"名不副实"（score 由范数主导），未处理"方法节写的公式根本没被执行"和"理论前提 H1 实测失败"。

### 3.1 三处一致性矩阵

| 位置 | 现在卖什么 | 实际执行/实测是什么 | 自洽？ |
|:---|:---|:---|:--:|
| **引言**（framework :10-12, :17-19；outline :27-31 Gap-1/Gap-4） | 量化误差异质性被忽视；"高误差层需要更大适配容量来补偿量化偏差，低误差层的秩是浪费的" | `epsilon_l_report.json` / `ranks.json` 复算：rel_error min 0.092598 / median 0.093197 / max 0.100716 / mean 0.093584 → **CV 1.48%，极差仅 8.9%** | ❌ **被否证** |
| **方法**（outline :84；framework :89-93） | `r_l ∝ ‖E_l‖_F · I_l`，`I_l = ‖W_l‖_F` | `adaq_precompute.py:115-122`：`rel = err/(norm_w+1e-8)`；`score = rel * norm_w` ⇒ **score ≡ ‖E_l‖_F**（复算 max 偏差 0.000e+00）。消融 `product` 取的是这个 score（`generate_ablation_configs.py:35`），不是 ‖E‖·‖W‖ | ❌ **公式≠实现** |
| **方法（理论）**（outline :68-91 Prop2；theory-review-report C2） | H1：各层奇异值谱服从幂律且指数 α 近似相同 ⇒ r*_l ∝ c_l^{2/α} ⇒ 代理 ‖E_l‖_F·I_l | `epsilon_l_report.json.powerlaw`：n_fitted 64，**α_min 0.1815 / α_median 0.4356 / α_max 1.5905（8.8× 跨度）**，r²_median 0.8018 ⇒ 2/α ∈ [1.26, 11.0]，"各层 α 相同"不成立 | ❌ **前提失败** |
| **方法（代理有效性）**（outline :91 Prop3b "实测 ρ=0.9836 ✅"） | 代理与真实边际增益 Spearman ≥0.7，已实测 | ρ=0.9836 出自 `generate_synthetic_ranks.py:40-52`，其 rel_error 是**硬编码 0.025/0.018/0.015**；theory-review-report :203 亦承认"预计算脚本 --self-test 已输出示例值"，真实模型上的 ρ **从未测过** | ❌ **合成当实测** |
| **实验**（outline §4.3 :122-124；framework :88-94） | 六策略消融可分离"误差项/重要性项/交互项" | 实际只跑了 `rel_error` / `importance` / `score(≡‖E‖)` / 常数 / **z-score 加权和** / 随机（`generate_ablation_configs.py:31,33,35,36-50,51-57`）；且 Scheme A 后 **product ≡ importance_only `{4,12}`、error_only ≡ uniform `{8,8}`** ⇒ 6 臂只有 4 种分配，**交互项与误差项都没有被真正检验** | ❌ **设计≠声明** |
| **讨论**（outline :146-148） | "高 ε 层获得更高秩 → 验证方法直觉" | ε 平坦（CV 1.48%）；且 Scheme A 下秩按**模块类型**（q/v 两值）分配，per-layer 秩从未进入 PEFT ⇒ 图4 想画的东西在部署配置里不存在 | ❌ **双重错误** |

### 3.2 结论：方法节现在卖的是"量化误差"，而且卖的公式连自己都没跑

三条独立线索指向同一个结论，必须在正文里合并处理，不能只改标题措辞：

1. **信号层**：rel_error 与 importance 近乎不相关（复算 Spearman(rel, importance) = **0.1056**）。所以"平坦的误差项"**不是**范数的代理，它只是**常数噪声**。⇒ 提案 :66-68 的"若保留 ×rel_error 就说该项近似常数"是**成立且安全**的写法；但更干净的写法是直接承认实现就是 `score ≡ ‖E_l‖_F`，而 ‖E_l‖_F = ε_l·‖W_l‖_F，ε_l 近常数 ⇒ **‖E_l‖_F ∝ ‖W_l‖_F**，即"实现出来的规则本来就是范数规则"。这一步让"名不副实"从缺陷变成**可证明的等价性**，是本评审建议的核心叙事（§8.3）。
2. **理论层**：Prop2 依赖的 H1 在真实 Llama-3-8B 上不成立（α 跨 8.8×）。⇒ 不能把 `r_l ∝ ‖E‖·I` 说成"由 KKT 边际条件导出"，只能说"在 H1 下可导出；我们实测 H1 不成立，因此该规则是启发式，且我们报告其失效条件"。这与 `theory-review-report.md:190`"升格是拒稿级错误，降级反而加分"的自家结论一致——但该报告 §6 未把 H1 的**实测违反**列为待办，`epsilon_l_report.json` 其实已经测出来了，属于**已有产物未被写作计划消费**。
3. **实验层**：由于 Scheme A 塌缩 + 三处公式错位，当前消融**无法**支撑任何关于"误差项/交互项"的结论；它只能支撑"在 126 步优化预算下，4 种实际不同的分配彼此不可区分"。这是一个更弱但**真实**的结论。

### 3.3 一个提案完全没有覆盖的新发现（P0，影响全部三张表的解释）

`checkpoint-*/trainer_state.json` 实读：

| 实验 | 序列长度 | `packing` | **global_step = max_steps** | 3 epoch 总 token | 末段 loss |
|:---|:--:|:--:|:--:|:--:|:--:|
| 8B 主表（12 run） | 2048（`train.py:124` 默认） | True（`train.py:132` 默认，无 run 传 `--no-packing`） | **21** | 593,461（step20 记录） | 2.0102→1.4971 |
| 8B 消融（6 策略×2 种子） | 1024（`run_ablation_schemeA_multiseed.sh:32`） | True | **126** | — | 1.4629→1.1898 |
| 14B（adaq/qlora） | 512（`run_b4_qwen14b.sh:37`） | True | **255** | — | 1.2590→0.9356 |

**含义**：三张表都写"3 epochs / 2000 条 / lr 2e-4"，但真实优化步数相差 **12×**（21 / 126 / 255）。主表在 **21 步**（warmup_ratio 0.03 ⇒ 约 0.6 步预热，cosine 在 step10 已衰减到 1.31e-4、step20 到 4.89e-6）的条件下比较四种秩分配——**这个预算下任何分配策略都不可能有可检出差异**。

由此产生三个必须写进正文的推论：

- **推论 A（削弱主表阴性）**：8B 主表的"无差异"**不能**解读为"秩分配无效"，只能解读为"在 21 步 packed SFT 预算下不可检出"。`claim-revision-norm-aware.md` 的"negative but reproducible"若不加此限定，是**过度概括**，审稿人只要问一句"你训了多少步"就会击穿。
- **推论 B（加强消融阴性）**：消融有 126 步（6× 主表）仍然不可区分，这是**目前功效最高的阴性证据**，应成为论文主证据，而不是主表。
- **推论 C（重解 14B 正向）**：14B 的 +13.3 点 flex 差出现在 255 步（12× 主表）的训练预算下。提案 :114 把它归因为"规模依赖的迹象"——**该归因不成立**：14B 相对 8B 同时改变了模型、序列长度（512 vs 2048）、batch（2×8 vs 4×4）、优化步数（255 vs 21）、以及**秩预算（528 vs 768）**，共 5 个混淆因子。把"步数效应"说成"规模效应"是**无证据的机制归因**，必须改写为"训练预算差异与规模差异均无法排除"。

---

## 4. 问题②：逐节 claim → evidence 映射

标记：✅ 有证据且措辞匹配 / ⚠️ 有证据但措辞过强 / ❌ 无证据或证据反向。

### 4.1 Abstract（outline :16-21 + 提案 :56 替换稿）

| claim | 证据 | 判定 | 措辞上限 |
|:---|:---|:--:|:---|
| "uniform rank 忽视逐层量化误差异质性"（outline :17） | rel_error CV 1.48% | ❌ | 必须删；改为"忽视权重范数/重要性的层间异质性（CV 56.7%）" |
| "precompute NF4 per-layer error → error-driven allocation"（:19） | `adaq_precompute.py:115-122` score≡‖E‖ | ❌ | 改为"a zero-search, data-free precompute over the frozen NF4 base yields a per-site score that is provably proportional to ‖W‖ when relative error is layer-constant" |
| "rigorous theory: perturbation bound + spectral lift + proxy diagnostic"（framework :21b） | Thm1 ✅；Prop1' 效应 +0.45%~0.62%；Prop2 前提失败；Prop3b 合成 | ⚠️ | 删 "rigorous"；改 "a Weyl-type bound, an expectation-level spectral-lift analysis, and an explicit statement of the conditions under which a pre-hoc proxy is and is not identifiable" |
| "parameter-efficiency gain at zero overhead"（:21） | 8B 无增益；14B −26.6% 参数但基线多 45.5% 秩；NF4 口径 adaq 最差 | ❌ | 只保留 "at zero training overhead (a single CPU pass, <5 min)"；参数效率必须绑定 14B 单种子 + 预算不对齐限定 |
| 提案 :56 "six strategies indistinguishable in mean, between-seed reversal" | `results_v2_ablation_multiseed/REPORT.md` + 本次复算 | ✅ | 保留，但补"其中仅 4 种分配实际不同" |
| 提案 :56 "negative but reproducible" | 消融 126 步 ×2 种子；主表仅 21 步 | ⚠️ | 必须补训练预算限定（推论 A/B） |

### 4.2 §1 Introduction

| claim | 证据 | 判定 | 处置 |
|:---|:---|:--:|:---|
| Gap-1 uniform rank 浪费容量 | importance CV 56.65%（真）；但"浪费"需实验证明，主表 21 步、消融不可区分 | ⚠️ | 重锚到范数异质性 + 明示"我们没有证据证明纠正它能带来收益" |
| Gap-2 LoftQ/QA-LoRA 不做秩分配 | 文献层面成立（需按 §8.4 补 AQLoRA/PARA 后重新核查） | ⚠️ | 补 AQLoRA（精度/放置分配）与 PARA（训练后 SVD 分配）后，Gap-2 必须收窄 |
| Gap-3 AdaLoRA 4-bit 重要性不可靠 | **从未测量**；`sota-comparison-protocol.md:90` 预写"结果差是预期证据" | ❌ | 删除或改为可检验假设；实测 adalora flex .4824 vs qlora .4809（AdaLoRA 并不差）⇒ 该 Gap 被自家数据反驳 |
| Gap-4 NF4 逐模块误差相差数倍 | CV 1.48% | ❌ | 删除 |

### 4.3 §3 Method

| claim | 证据 | 判定 | 处置 |
|:---|:---|:--:|:---|
| A1 E_l 近似零均值 | `zero_mean`: 9.81e-06 / max 1.78e-05 / mean_abs 9.53e-04 | ✅ | 升格为已验证，引用实测产物 |
| A2 ε₄≈0.02 | 实测 mean 0.0936（median 0.0932，range 0.0926–0.1007） | ❌ | 全文替换为 0.094（并说明是 ‖E‖_F/‖W‖_F，q/v 两模块、64 位点） |
| A3 块内同分布 | `per_layer` 有 within_var 3.15e-06 / between_var 4.99e-08 / autocorr_lag1 0.007（layer0 q_proj） | ✅ | 可报"块间方差比块内小 63×，lag-1 自相关 0.007" |
| Thm1 Weyl 双侧界 | 数学成立，无额外假设 | ✅ | 保留 |
| Prop1' 平均谱抬升 | `erank_vs_error.json`：baseline erank 1.2020；Δ@rel_err 0.02 = +0.000447；Δ@0.08 = +0.00751（erank 1.2096） | ⚠️ | 必须同时报**相对幅度 ≤0.7%**，并说明合成谱用 α=1.5 而实测 α_median=0.436 |
| Prop2 r*_l ∝ c_l^{2/α} | α∈[0.181,1.591]，r²_median 0.802 | ❌ | 降为"conditional on H1；we measure H1 fails" |
| Prop3a 取整损失 ≤O(L σ̄²/R) | 可证；但 Scheme A 14B 取整后 528≠512（+3.1%） | ⚠️ | 补"Scheme A 分组取整会破坏精确守恒，14B 实测 +3.1%" |
| Prop3b ρ≥0.7 实测 0.9836 | 合成硬编码 | ❌ | 改"synthetic self-test only; not measured on real models" |
| §3.5 "与 QLoRA 唯一差异是秩分配" | 8B ✅（512=512，超参全同）；14B ❌（528 vs 768） | ⚠️ | 分表声明 |
| "预计算 <5 min，零训练开销" | `profile_training.py` 存在但无产物；预计算耗时未见记录 | ⚠️ | 补一次计时产物再写数字 |

### 4.4 §4 Experiments

| claim | 证据 | 判定 | 处置 |
|:---|:---|:--:|:---|
| 主表公平可比（同预算同超参） | 12 run `training_args.bin` 仅种子不同；adaq 32×(12+4)=512=qlora 32×16 | ✅ | 保留，作为方法学优点写出来 |
| 主表 adaq 有效 | flex dora .4877 > adaq .4850 > adalora .4824 > qlora .4809；MMLU dora .6543 > adaq .6540 | ❌ | 只能写"不可区分，DoRA 点估计最高" |
| 消融可分离误差/重要性/交互 | 6 臂 → 4 种分配；product≡importance_only、error_only≡uniform | ❌ | 重写为"4 种实际分配不可区分" |
| 噪声底 4.5 点 | 取自 product/importance_only 同配置一对（s3407 flex .5246 vs .4792）；另一对 error_only/uniform 实测 .08/.83 点 | ⚠️ | 改报区间并说明 n=2 观测、非齐性 |
| 14B 参数效率 −26.5% | 4,620,288 vs 6,291,456（−26.6%）✅ 但秩预算 528 vs 768 | ⚠️ | 只可写"以少 26.6% 参数、少 31.3% 总秩取得 MMLU 平价（+0.04~0.06 点，stderr 0.33 点）" |
| 14B GSM8K flex +13.3 点 | n=1（seed42）；seed2025 仅有 MMLU，seed3407 无结果目录 | ⚠️ | 单种子 + 5 混淆因子；`aggregate_b4_3seeds.py:97-101` 预注册规则本身**未含预算不对齐条款**，需补 |
| NF4 口径下方法占优 | adaq_nf4 GSM8K .4064 = 五者最低，低于无 adapter 基线 .4124 | ❌ | 必须正面写（§8.5） |
| PPL 优势 | adaq 7.4314 最低，但 `sample_len=62`，全距 0.009 | ❌ | 不进正文，或仅附录并标 n=62 |

### 4.5 §5-§7

| claim | 证据 | 判定 | 处置 |
|:---|:---|:--:|:---|
| "高 ε 层获得更高秩" | ε 平坦 + Scheme A 非逐层 | ❌ | 删 |
| "零额外开销下提升参数效率"（framework :152-158 预设 rebuttal） | 8B 无提升；14B 提升但预算不对齐 | ⚠️ | 只保留"零额外开销"，删"提升" |
| 结论"量化误差驱动的秩分配" | 全链条否证 | ❌ | 用 §8.7 替换 |

---

## 5. 补充实测：预算/参数公平性审计（张量级，本次评审新发现）

> ⚠️ **对 §2 第 9 行"主表公平性 ✅"的勘误**：该 ✅ 仅对 qlora / dora / adaq 三者成立。**AdaLoRA 臂未对齐预算**，证据如下。本节全部数字由 `adapter_model.safetensors` 头部（张量形状）直接算出，未经训练框架转述。

### 5.1 8B 主表（每方法 3 种子，三种子完全一致）

| 方法 | 实际 rank_pattern | **总秩** | **可训练参数** | 相对 qlora 参数 | 预算对齐？ |
|:---|:---|:--:|:--:|:--:|:--:|
| qlora | q=8, v=8 | 512 | 3,407,872 | 1.000× | 基准 |
| dora | q=8, v=8 (+64 个 magnitude 向量) | 512 | 3,571,712 | 1.048× | ✅（参数略多，须脚注） |
| **adaq** | **q=12, v=4** | **512** | **3,801,088** | **1.115×** | ✅ 秩对齐 / ❌ **参数多 11.5%** |
| **adalora** | **全部 16（init_r，未剪枝）** | **1024** | **6,816,768** | **2.000×** | ❌ **秩与参数均 2×** |

**AdaLoRA 为何是 1024**：`adapter_config.json` 为 `init_r=16, target_r=8, tinit=20, tfinal=60, total_step=200`；而实际训练 `max_steps=21`（§3.3）。剪枝在 step20 才开始、step60 才完成 ⇒ **训练在剪枝刚启动时就结束了**，三个种子全部停在 rank 16（`lora_A` 形状 [16,4096]，`lora_E` [16,1]×64）。

**直接后果（P0）**：
1. 违反自家协议 `sota-comparison-protocol.md:42`（"所有方法的总秩预算 R 相同，唯一差异是分配方式"）与 :90（"AdaLoRA 全局预算对齐到 R=512"）、:78（❌ 不同预算比较）。
2. **Gap-3"AdaLoRA 在 4-bit 上重要性不可靠"失去唯一支撑**：AdaLoRA 是拿 2× 预算跑的，它没有赢（flex .4824 vs adaq .4850 / dora .4877）说明的是"21 步下连 2× 预算都用不上"，**不是**"重要性分数不可靠"。该 Gap 必须从引言删除或改写为未检验假设。
3. `sota-comparison-protocol.md:191-200` 主表模板写的 "Params(M) 全部 2.7M" **四行全错**：实测 3.41 / 3.57 / 3.80 / 6.82 M。

### 5.2 14B（seed42/2025/3407 三个 adapter 形状完全一致）

| 方法 | rank_pattern | 总秩 | 参数 | 说明 |
|:---|:---|:--:|:--:|:---|
| adaq | q=7, v=4 | **528** | 4,620,288 | Scheme A 取整后 48×(7+4)=528 ≠ 512（**+3.1%**，破坏 `adaq_precompute.py:81` 的守恒断言语义） |
| qlora | q=8, v=8 | **768** | 6,291,456 | **基线多 45.5% 秩、多 36.2% 参数** |

### 5.3 由 5.1/5.2 推出的、比提案更严重的结论：**"参数效率"贡献是预算错配的产物**

在**等秩**条件下把 adaq 与"同总秩的均匀分配"对比（均匀 = 总秩平摊到 96/64 个位点）：

| 模型 | adaq 参数 | 等秩均匀参数 | adaq 相对 |
|:---|:--:|:--:|:--:|
| Llama-3-8B（总秩 512） | 3,801,088 | 3,407,872（r=8/8） | **+11.5%** |
| Qwen2.5-14B（总秩 528） | 4,620,288 | 4,325,376（r=5.5/5.5） | **+6.8%** |

**机制**：范数大的矩阵恰好是**每单位秩更贵**的矩阵（8B：q_proj 每秩 8192 参数 vs v_proj 5120；14B：10240 vs 6144）。因此"按 ‖W‖_F 分配秩"在参数口径上等价于**"优先给贵的矩阵加秩"**，等秩下必然增加参数量。

⇒ **提案 :60-62 与任务书列为"存活贡献"的"参数效率（14B −26.5%）"不能成立**：那 −26.5% 完全来自 qlora 拿了 768 秩而 adaq 只有 528 秩，与分配规则无关。等秩重算后方向**反号**。

⇒ 由此得到一条**可立即检验的新假设**（建议写进 Future Work，并作为返修期最便宜的一个实验）：
**成本归一化分配** `r_l ∝ score_l / cost_l`，其中 `cost_l = d_in(l) + d_out(l)` 为该位点每单位秩的参数量。在 q/v 两类位点上它会把 q:v 的秩比从 12:4 翻转为约 4:6（8B），从而在**等参数**而非等秩口径下实现真正的参数效率。该规则仍保持零搜索、零数据、一次 CPU pass。

---

## 6. 问题③：范数感知重构能否扛住全部证据？

### 6.1 能扛住的部分（可以作为正文的确定性陈述）

| 陈述 | 证据 | 强度 |
|:---|:---|:--:|
| 该规则在数学上就是范数规则 | `score = rel_error × ‖W‖`，rel_error CV 1.48% ⇒ score ∝ ‖W‖；复算 Spearman(score, ‖W‖)=0.9979、Spearman(score, rel_error)=0.1440、Spearman(rel_error, ‖W‖)=0.1056 | ✅ 强 |
| 实现出来的 score 恰为 ‖E_l‖_F，而 ‖E_l‖_F = ε_l·‖W_l‖_F，ε_l 近常数 ⇒ **等价于范数分配（可证明，不只是经验相关）** | `adaq_precompute.py:115-122` + `epsilon_l_report.json` eps min/median/max = 0.0926/0.0932/0.1007 | ✅ 强（**这是比提案更硬的表述**） |
| 重要性确实高度异质 | importance CV 56.65%（13.33–75.27） | ✅ 强 |
| 部署粒度是模块类型而非层 | `ranks.json` Scheme A `{v_proj:4, q_proj:12}`；张量核对 q=12×32, v=4×32 | ✅ 强 |

### 6.2 扛不住、必须在正文里"不回避地写"的部分

**(a) NF4 主场阴性（E1）** —— 必须原样进正文，不能只放附录：

> Under the evaluation path that matches training (NF4-loaded base with double quantization, bf16 compute), the proposed allocation is the **worst of five** configurations on GSM8K (0.4064), below the **no-adapter** NF4 baseline (0.4124) and below QLoRA (0.4124), DoRA (0.4230) and AdaLoRA (0.4185); on MMLU it is 0.6335 vs 0.6364 for the un-adapted base. This is the regime in which a quantization-error-driven rule should have its advantage, and it does not. We report it as a falsification of the error-driven hypothesis rather than as a hyper-parameter artefact, and we did not tune any method on this path.

**(b) 8B 消融阴性（E3）** —— 用**同配置四跑**的版本写，比提案的"4.5 点噪声底"更强也更诚实：

> Six allocation strategies were trained under two seeds. Because the deployed granularity is per module type, the six strategies realise only **four distinct allocations**: `product` and `importance_only` are both {q=12, v=4}, and `error_only` and `uniform` are both {q=8, v=8} (verified from the saved `lora_A` tensor shapes). The four runs of the **identical** {12,4} allocation yield GSM8K flexible-extract accuracies of 0.5246, 0.5011, 0.5186 and 0.4792 — a within-configuration range of **4.5 points** — while the range across all six strategy means is **2.7 points**. Run-to-run variation therefore exceeds the entire between-strategy effect, and no pairwise comparison in this table is interpretable. A second identical-configuration family ({8,8}: 0.5254, 0.5262, 0.5254, 0.5171) gives a much tighter range of 0.9 points, so the noise is **heterogeneous across allocations** and we report it as an interval [0.9, 4.5] rather than a single floor.

（说明：`results_v2_ablation_multiseed/REPORT.md:53` 只发现了 product≡importance_only 这一对重复，**漏了 error_only≡uniform**；而恰恰是漏掉的这一对给出了小得多的噪声估计。因此"噪声底 = 4.5 点"是 n=1 观测，不能作为唯一判据——但**结论方向不变**：即使取最保守的 0.9 点，2.7 点的策略极差也只勉强超出，且其中 4 个臂是两两同配置。）

**(c) 训练预算混淆（§3.3 推论 A/B/C）** —— 每张表脚注必须出现 `global_step`：

> Table 1 (main comparison): 3 epochs over 2,000 packed samples at sequence length 2048 with `packing=True` collapse to **21 optimiser steps** (`max_steps=21`, 593,461 tokens, cosine schedule with 0.6-step warm-up). Table 2 (ablation, sequence length 1024) runs **126 steps**; the 14B experiment (sequence length 512) runs **255 steps**. All three tables are therefore comparisons under different optimisation budgets and their absolute levels are not comparable. Under 21 steps no allocation strategy can express a difference; the ablation at 126 steps is the better-powered negative and we treat it as the primary evidence.

**(d) 理论前提失败（H1/α）** —— 不能藏：

> The continuous-relaxation argument (Proposition 2) requires per-layer singular-value spectra that follow a power law with a **common** exponent α. Measured on the actual 64 target sites of Llama-3-8B, the fitted exponents span α ∈ [0.181, 1.591] (median 0.436, median R² = 0.802), i.e. the implied allocation exponent 2/α ranges over 1.26–11.0. H1 is therefore **violated on the model we study**, and the allocation rule is presented as a heuristic whose derivation conditions we measure and report as unmet.

**(e) 等秩下参数反而更多（§5.3）** —— 必须写，否则审稿人自己算出来就是诚信问题。

### 6.3 重构后的故事是否还成立？

**成立，但重心必须从"方法"移到"诊断 + 边界条件"。** 三条支柱：

1. **诊断（可证明）**：在 NF4 冻结基座上，*相对*重构误差是层级常数（CV 1.48%），因此任何"误差×范数"型打分都退化为范数打分；而范数打分在等秩下增加参数、在等参数下未被检验。→ 这解释了为什么"量化感知秩分配"这一整类想法在 4-bit 上难以奏效，并与 AQLoRA 的独立对照结论一致（§8.4）。
2. **可重复的阴性 + 功效分析（新方法学贡献）**：我们给出该研究方向的**最小可检出条件**——21 步 packed SFT 下噪声主导；126 步下同配置跑间极差 0.9–4.5 点仍大于策略极差 2.7 点。任何后续秩分配研究若不报告 `global_step` 与同配置重复跑，其阴性/阳性都不可解释。
3. **工程配方（弱但真实）**：一次 CPU pass、零校准数据、零训练开销、结果确定性可复现（`ranks.json` 随代码开源）。这一条必须与 AQLoRA 的"zero-search recipe"正面区分（§8.4），否则新颖性不成立。

---

## 7. 问题④：相关工作定位（delta 句，可直接入正文）

**现状核查**：全仓 `grep -rn "PARA|AQLoRA" docs/*.md` **仅 1 处命中**（`sota-comparison-protocol.md:22` 的待核查清单里列了 AQLM/QuaRot，**没有 AQLoRA、没有 PARA**）。`paper-outline.md:37-51` 的 §2.1/§2.2 引用列表里两者均缺席。提案 144 行**完全没有涉及相关工作**。这是 P0 级空白：其中一篇（PARA）与本文标题撞名，另一篇（AQLoRA）已独立做出与本文同方向的阴性判定。

### 7.1 QLoRA（Dettmers et al., 2023）

> QLoRA establishes the frozen-NF4 regime we build on: 4-bit NF4 weights with double quantisation, paged optimisers, and a **uniform** LoRA rank on the adapted sites. We change exactly one thing — the per-site rank pattern — and keep base, quantisation, data, schedule and total rank identical. Our measurement is that this single change is **not detectable** at the optimisation budgets we could afford (21 steps for the main comparison, 126 for the ablation): at equal total rank 512 on Llama-3-8B, QLoRA's uniform pattern reaches GSM8K flexible-extract 0.4809 ± 0.0042 and our norm-driven pattern 0.4850 ± 0.0019 (n = 3 seeds, no valid significance test at n = 3). We therefore position our contribution as a **diagnostic of when such a change can matter**, not as an improvement over QLoRA.

### 7.2 AdaLoRA（Zhang et al., 2023）

> AdaLoRA allocates rank **online**: it over-parameterises (init_r = 16), scores singular triplets during training, and prunes to a target budget on a schedule. Our allocation is **offline**: one CPU pass over the frozen base, no data, no gradient, no training-time cost. The two are complementary rather than competing. We must also report a fairness failure in our own baseline: with `tinit = 20`, `tfinal = 60` and `total_step = 200` configured against an actual `max_steps = 21`, AdaLoRA's pruning **never engaged**, and all three seeds trained at rank 16 on every site (total rank 1024, 6,816,768 trainable parameters — twice the budget of every other row). Any statement about the reliability of AdaLoRA's importance scores on 4-bit bases is therefore **not supported by our experiments**, and we withdraw the corresponding gap claim.

### 7.3 DoRA（Liu et al., 2024，weight-decomposed）

> DoRA decomposes the update into magnitude and direction and adds a trainable magnitude vector per site; it does not redistribute rank. At the same total rank 512 it carries 3,571,712 trainable parameters against QLoRA's 3,407,872 and our 3,801,088. In our main table DoRA has the **highest point estimate on all three reported metrics** (MMLU 0.6543 ± 0.0015, GSM8K flexible 0.4877 ± 0.0072, strict 0.4799 ± 0.0055), which is the opposite of what a rank-allocation story predicts and is why we make no performance claim. **待核查**：检索中出现一篇题为 "DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution"（2026-04）的条目，与本文基线 DoRA（weight-decomposed）**同名不同工作**；投稿前必须核实并在引用中消歧，否则审稿人会认为基线选错。

### 7.4 PARA（Kumaravelu, Gupta & Srijith, 2026；arXiv [2604.27796](https://huggingface.co/papers/2604.27796)）

> PARA shares our problem statement — uniform rank ignores the varying intrinsic dimensionality of layers — and our name space ("adaptive rank allocation"), but inverts the information source and the timing: it is a **post-optimisation**, data-free compression that re-allocates rank from the **SVD of the trained adapter** and reports 75–90% parameter reduction at preserved accuracy. Our rule is **pre-optimisation** and reads only the **frozen base** (‖E_l‖_F and ‖W_l‖_F), never the trained update. This contrast is the most informative result in our paper: the signal PARA exploits (the spectrum of ΔW after training) is precisely the signal that is *unobservable* before training, and the base-side substitutes we can measure — relative NF4 error (CV 1.48%) and weight norm (CV 56.7%) — do not separate allocations at the budgets we tested. **Ablation we did not run and should**: applying PARA-style post-hoc SVD re-allocation to our own trained adapters would test directly whether the failure lies in the *timing* of the signal or in the *budget*.

（行动项：标题必须避开 "Adaptive Rank Allocation" 的裸用，见 §8.7。）

### 7.5 AQLoRA（arXiv [2608.23816](https://arxiv.org/html/2608.23816v1)，2026-08）—— **最危险，也是最有价值的一条**

> AQLoRA is a zero-search recipe for quantised LoRA fine-tuning: one CPU pass ranks layers by **NF4 reconstruction error** and keeps the top-K in fp16 under a memory budget, buying training speed by skipping dequantisation, with no calibration data and no search. It differs from us in *what* is allocated (precision and adapter placement, versus rank budget) and in *why* (speed, versus capacity). It is nevertheless the closest prior work to our framing, and it is prior: it predates our submission.
>
> More importantly, AQLoRA reports a control that **independently reproduces our central negative finding**. Their own words: "the gain comes from protecting *some* layers, not from protecting the *right* ones, and NF4 reconstruction error is not doing the work we attributed to it … it is a function of the budget and not of the selection", and they downgrade their rule to "a way to choose a fixed number of layers cheaply and reproducibly rather than as a sensitivity criterion that has been shown to work". We reach the same conclusion from a different allocation axis, a different model (Llama-3-8B; they report phi-1.5, Llama-3.2-1B/3B, Phi-3-mini, Qwen2.5-7B/14B), a different metric family, and — crucially — from a **mechanism** they do not give: the *relative* NF4 reconstruction error is close to constant across layers (CV 1.48%, range 0.0926–0.1007 over 64 sites), so an error-ranked selection is close to a random or norm-ranked one by construction. Our `error_only` arm, which after module-type grouping is **identical to `uniform`**, is the direct experimental counterpart of their control.
>
> We also adopt their budget-reporting lesson: they show a budget expressed as a *fraction of layers* does not transfer across architectures (4.3% of weights on Llama-3.2-1B versus 16.3% on Phi-3-mini, a factor of fourteen in memory), and that a fraction of *parameters* would. Our own 14B comparison violates this (528 versus 768 total rank), which is why we report parameter counts and total rank for every row and make no scale-generalisation claim.

**必引但未核实（标 GAP，投稿前必须 scholar 核查）**：AQLoRA 自述 "missing is a comparison against the search-based bit- and rank allocators (Zhou et al., 2025a; 2026)" —— 这两条是 bit/rank 分配方向的直接相关工作，本文相关工作必须覆盖；确切题名/出处**待核查**，不得凭记忆填写。

---

## 8. 问题⑤ + ⑥：精确措辞与重写文本

### 8.1 Abstract（重写，英文，替换 outline :16-21）

> Rank allocation for 4-bit quantised LoRA is usually motivated by the observation that NF4 reconstruction error varies across layers, so that high-error layers should receive more capacity. We test that motivation directly and find it does not hold. On Llama-3-8B, the *relative* NF4 reconstruction error of the 64 adapted sites is close to constant (CV 1.48%, range 0.0926–0.1007), while weight-norm importance varies by a factor of 5.6 (CV 56.7%). A score of the form ‖E_l‖_F · ‖W_l‖_F is therefore provably proportional to ‖W_l‖_F alone in this regime, and the rule actually applied is norm-driven rather than error-driven. We report the consequences without selection. Under an NF4-loaded evaluation path matching training, our allocation is the worst of five configurations on GSM8K (0.4064) and falls below the un-adapted base (0.4124). In a six-strategy ablation over two seeds, module-type grouping collapses the strategies to four distinct allocations, and four runs of one identical allocation span 4.5 points of GSM8K flexible-extract accuracy against a 2.7-point spread across all six strategy means, so no pairwise difference is interpretable. In the main comparison at equal total rank (512) over three seeds, QLoRA, DoRA, AdaLoRA and our method are indistinguishable, with DoRA holding the highest point estimates; three seeds admit no valid Wilcoxon test (minimum two-sided p = 0.25). We also disclose a confound that affects this literature generally: with sequence packing, three epochs over 2,000 samples collapse to 21 optimiser steps, while our ablation and 14B runs used 126 and 255 steps, so absolute levels across tables are not comparable. Our contributions are a diagnostic identifying when error-driven allocation is unidentifiable, a Weyl-type bound with an explicit statement of the power-law condition the continuous relaxation needs (measured here to fail, α ∈ [0.18, 1.59]), a zero-search data-free pre-computation recipe, and a reproducible negative that converges with an independently reported control for NF4-error-based layer selection. We do not claim a performance improvement.

### 8.2 贡献 bullet（重写，英文，替换提案 :60-62）

> 1. **A diagnostic for error-driven allocation.** We show that on a frozen NF4 base the relative reconstruction error is close to layer-constant (CV 1.48% over 64 sites of Llama-3-8B; CV 4.00% over 96 sites of Qwen2.5-14B) while weight-norm importance is highly heterogeneous (CV 56.7% / 39.3%). Since ‖E_l‖_F = ε_l · ‖W_l‖_F, any error-times-norm score is proportional to the norm alone under this condition; we quantify it with Spearman ρ = 0.998 against ‖W_l‖_F, ρ = 0.144 against ε_l, and ρ = 0.106 between the two factors. This explains, rather than merely reports, why error-ranked selection behaves like random selection — a conclusion reached independently by concurrent work on NF4-error-based precision allocation.
> 2. **Theory with measured failure conditions.** We give a two-sided Weyl bound on the singular values of the residual update (no distributional assumption), an expectation-level spectral-elevation analysis (relative effective-rank change ≤ 0.7% at the measured error level), and we state the power-law-with-common-exponent condition required to derive a proportional allocation rule — then **measure that it is violated** on the model we study (α ∈ [0.181, 1.591], median 0.436, median R² = 0.802). The allocation rule is presented as a heuristic, not as an optimum.
> 3. **A reproducible negative with a power analysis.** Across six strategies, two seeds and two benchmarks, and across four methods, three seeds and four benchmarks at equal total rank, we find no separable effect. We report identical-configuration replicate runs bounding run-to-run noise at 0.9–4.5 points against a 2.7-point between-strategy spread, we report that n = 3 seeds admits no valid Wilcoxon test, and we report the optimiser-step counts (21 / 126 / 255) that make the three tables non-comparable. This is, to our knowledge, the first published minimum-detectable-effect analysis for rank-allocation studies.
> 4. **A zero-search, data-free pre-computation recipe, with its cost stated.** One CPU pass over the frozen base fixes all ranks in under five minutes with no calibration data and no training-time overhead, and the resulting `ranks.json` is released. We also report its cost: at equal total rank, concentrating rank on high-norm sites raises trainable parameters by 11.5% (Llama-3-8B) and 6.8% (Qwen2.5-14B) relative to a uniform pattern, because those sites are also the more expensive ones per unit of rank. Parameter-efficiency claims for this family should therefore be made at equal parameters, not at equal rank.

### 8.3 §3.3 重写（英文，替换 outline :68-91 的 Allocation Rule 段）

> **Definition (score).** For each adapted site l we compute, in a single CPU pass over the frozen base, the NF4 reconstruction residual E_l = Q₄(W_l) − W_l and the two scalars ε_l = ‖E_l‖_F / ‖W_l‖_F and I_l = ‖W_l‖_F, and we set score_l = ε_l · I_l² = ‖E_l‖_F · I_l. In our implementation the stored quantity is `score = rel_error × importance`, which equals ‖E_l‖_F exactly; we state this explicitly because it determines what the rule can do.
>
> **Lemma (degeneration to a norm rule).** If ε_l is constant across sites, then score_l ∝ I_l and the induced allocation is identical to the norm-aware (importance-aware) allocation. *Proof.* ‖E_l‖_F = ε_l ‖W_l‖_F, so score_l = ε_l ‖W_l‖_F²; with ε_l ≡ ε, score_l = ε ‖W_l‖_F², a monotone function of ‖W_l‖_F, and the water-filling allocation depends on the scores only through their ordering. ∎
>
> **Measurement.** The premise holds to within 1.5% on the model we study: over the 64 adapted sites of Llama-3-8B, ε_l has mean 0.0936, median 0.0932 and range 0.0926–0.1007 (CV 1.48%), whereas I_l ranges over 13.33–75.27 (CV 56.7%). Spearman correlation of the score is 0.998 with I_l and 0.144 with ε_l; ε_l and I_l are themselves nearly uncorrelated (ρ = 0.106), so the flat factor is not acting as a proxy for the norm — it is acting as a constant. **The rule we deploy is therefore a norm-aware rule, and we describe it as such throughout.**
>
> **Status of the derivation.** Under H1 (per-layer singular-value spectra following a power law σ_{l,i} = c_l i^{−α} with a common α) and a continuous relaxation of the rank variables, the KKT marginal condition gives r*_l ∝ c_l^{2/α}, motivating a score proportional to a per-layer scale factor. We measured H1 on the same 64 sites and it **fails**: fitted exponents span α ∈ [0.181, 1.591] with median 0.436 and median fit quality R² = 0.802, so the implied exponent 2/α varies between 1.26 and 11.0 across layers. We consequently present the allocation as a **heuristic rule**, claim neither optimality nor an approximation ratio, and report the measured violation in the method section rather than an appendix.
>
> **Truncation and grouping.** Ranks are clipped to [r_min, r_max] = [4, 128] and renormalised by iterative water-filling so that Σ_l r_l equals the budget exactly. When ranks are exported to PEFT at *module-type* granularity (Scheme A, the granularity used in every experiment reported here), per-site ranks are averaged within each of {q_proj, v_proj}; this preserves the budget on Llama-3-8B (32 × (12 + 4) = 512) but not on Qwen2.5-14B (48 × (7 + 4) = 528 against a budget of 512, +3.1%), and it reduces the allocation to two integers. Both effects are reported wherever Scheme A is used.
>
> **Proxy-quality diagnostic.** We do not report a Spearman correlation between the score and the true marginal gain σ²_{l,r+1} on real models, because the latter requires the singular values of ΔW* − E_l, which are not observable before training. The value ρ = 0.9836 produced by our pre-computation self-test comes from **synthetic** statistics with hard-coded relative errors and must not be cited as a measurement. Establishing proxy quality on real models is future work and requires a post-hoc oracle (e.g. SVD re-allocation of trained adapters).

---

### 8.4 §5 Discussion 重写（英文，替换 outline :146-148）

> **What varies on a frozen 4-bit base.** The premise of error-driven allocation is that quantisation damage is unevenly distributed. In absolute terms it is: ‖E_l‖_F spans 5.7× across the 64 adapted sites. In *relative* terms it is not: ε_l spans 8.9% (CV 1.48%). Because a capacity-allocation rule can only use the ordering of its score, and the score is the product of the two, the ordering is inherited almost entirely from the norm (ρ = 0.998). The heterogeneity that motivated the method is real but sits in the wrong factor.
>
> **Why the error signal cannot be recovered by better measurement.** ε_l is flat because NF4 is a *relative* (block-wise absmax) format: each block of 64 weights is scaled by its own maximum, so reconstruction error scales with local magnitude and the ratio ‖E‖_F/‖W‖_F is normalised away by construction. A rule that wants error information must use a quantity that block-wise scaling does not normalise — for instance per-block outlier counts, or the residual after a second quantisation stage — rather than the relative Frobenius error. This is a design constraint on the whole family, not a limitation of our estimator.
>
> **Convergent evidence.** An independent zero-search recipe that ranks layers by NF4 reconstruction error to decide fp16 *precision* rather than rank reports, from its own control experiment, that the benefit comes from protecting some layers rather than the right ones, and that the outcome is "a function of the budget and not of the selection". Two allocation axes (precision, rank), several model families and two independent groups therefore agree: at 4 bits, NF4 reconstruction error is not a usable selection signal. Our contribution is the mechanism (relative error is layer-constant) and the experimental counterpart of their control (our `error_only` arm is, after module-type grouping, identical to `uniform`).
>
> **The negative is bounded, not open-ended.** Three facts limit how far the null generalises, and we state them as the conditions under which the question should be re-opened. (i) *Optimisation budget*: the main comparison ran for 21 steps; at that budget even a 2× capacity difference (AdaLoRA at total rank 1024) fails to separate from the 512-rank rows, indicating the experiment was insensitive rather than that capacity is irrelevant. (ii) *Granularity*: Scheme A reduces any per-layer rule to two integers, so our test is of module-type allocation, not layer allocation; a per-site test needs at least 64 distinguishable allocations. (iii) *Budget metric*: at equal *rank*, norm-aware allocation costs 6.8–11.5% more parameters because high-norm sites are also the expensive ones; the parameter-efficiency question is consequently untested, and the natural fix — allocating r_l ∝ score_l / (d_in + d_out) — is a one-line change we did not have budget to evaluate.
>
> **What we would need to claim a positive result.** Matched optimiser-step counts across all arms (reporting `global_step`, not epochs), per-site rather than module-type granularity, equal *parameter* budgets with total rank reported alongside, at least six seeds for any paired test, and a post-hoc oracle verifying that the proxy correlates with realised marginal gain. Until then the defensible claim is the diagnostic one.

### 8.5 Limitations（L1–L10，英文，替换 outline :152-156 + 提案 :76-77）

> **L1 Task-agnostic proxy.** I_l = ‖W_l‖_F is computed from the base model only and ignores the downstream task; it is fixed before training and cannot adapt.
> **L2 Power-law condition measured to fail.** Deriving a proportional rule requires per-layer power-law spectra with a common exponent. Measured exponents span α ∈ [0.181, 1.591] (median 0.436, median R² = 0.802) on the 64 adapted sites of Llama-3-8B. The rule is heuristic.
> **L3 Relative-error constancy.** Our measured ε_l (mean 0.0936, CV 1.48%) is ~4.7× larger than the ε₄ ≈ 0.02 value assumed in earlier drafts and is nearly constant across layers, which is what makes the score degenerate to a norm rule. Any bound calibrated at 0.02 must be re-derived.
> **L4 Granularity.** All reported experiments export ranks at module-type granularity (Scheme A), so the deployed allocation is two integers per model, not a per-layer profile. Per-site ranks were computed but never trained.
> **L5 Optimisation budget.** Packing collapses three epochs over 2,000 samples to 21 optimiser steps for the main comparison (126 for the ablation, 255 for 14B). The main comparison is under-powered for any allocation effect, and the three tables are not comparable in absolute level.
> **L6 Baseline budget violations.** AdaLoRA trained at total rank 1024 (2× budget) in all three seeds because its pruning schedule (tinit 20, tfinal 60, total_step 200) never engaged within 21 steps; the 14B QLoRA baseline trained at total rank 768 against our 528. Neither comparison is at equal budget, and we make no claim from either.
> **L7 Statistical power.** n = 3 seeds (main) and n = 2 (ablation) admit no valid Wilcoxon signed-rank test (minimum two-sided p = 0.25 at n = 3). All results are descriptive. Run-to-run noise, measured from identical-configuration replicates, is 0.9–4.5 points on GSM8K flexible-extract and is heterogeneous across allocations.
> **L8 Single-seed 14B signal.** The 14B GSM8K flexible-extract difference (0.6588 vs 0.5254) is a single seed, and that arm differs from the 8B experiments in five respects at once (model, sequence length 512 vs 2048, batch composition, optimiser steps 255 vs 21, total rank). We attribute nothing to scale. Strict-match is unusable on Qwen2.5-14B-Instruct because its output format violates the `#### N` answer pattern, so strict measures format compliance.
> **L9 Evaluation paths differ across tables.** The main table and the ablation evaluate a bf16 base with the adapter attached; the PTQ table and the 14B table evaluate an NF4-loaded base. Numbers from different paths are never placed in one table. Perplexity results use 62 samples and appear nowhere in the main text.
> **L10 Hardware and scope.** All experiments run on a single A10 (24 GB), one instruction dataset (2,000 OpenHermes-2.5 samples), one quantisation format (NF4 with double quantisation) and two model families. No 8-bit robustness study, no rank-budget sweep and no target-module expansion were completed, although the rank-budget sweep was designated mandatory in our own experiment plan.

---

### 8.6 §7 Conclusion（重写，英文）

> We set out to allocate LoRA rank by per-layer NF4 quantisation error and found that the signal does not exist in the form the method requires: relative reconstruction error is layer-constant on a frozen NF4 base, so an error-times-norm score is a norm score, and the resulting allocation is not separable from uniform, random or norm-only allocation at the optimisation budgets we could afford. We report the measurement, the mechanism, the theory together with its measured failure conditions, and a set of identical-configuration replicates that bound the noise above the effect. We also report where our own comparisons are not fair, and the minimum conditions — matched optimiser steps, per-site granularity, equal parameter budgets, at least six seeds, and a post-hoc oracle — under which this question should be re-opened. The recipe itself remains useful for what it costs: one CPU pass, no calibration data, no training overhead, deterministic and released.

### 8.7 标题建议

现标题 `AdaQ-LoRA: Quantization-Aware Adaptive Rank Allocation for PEFT of LLMs`（outline :10）三处不可用：**Quantization-Aware** 被 E1/E2 否证；**Adaptive** 在 Scheme A 下只有两个整数；**Adaptive Rank Allocation** 与 PARA（arXiv 2604.27796）撞名。备选：

1. `When Quantisation Error Is Flat: A Diagnostic and a Reproducible Negative for Error-Driven Rank Allocation in 4-bit LoRA`（推荐，与 §8.1 摘要匹配）
2. `Norm-Driven, Not Error-Driven: What a Zero-Search Rank Allocator Actually Computes on Frozen NF4 Bases`
3. `Identifiability of Rank Allocation in 4-bit LoRA: Flat Relative Error, Optimiser-Step Budgets, and a Bounded Null`

### 8.8 必须删除/改写的表述（在提案 :81-87 之上追加）

| 位置 | 原文 | 处置 | 依据 |
|:---|:---|:---|:---|
| `paper-framework.md:12` | "高误差层需要更大的适配容量来补偿量化偏差，而低误差层的秩是浪费的" | **删** | ε CV 1.48% |
| `paper-framework.md:17-19` Gap-1 | "忽视 ε_l 异质性" | 改为"忽视 ‖W_l‖ 异质性（CV 56.7%）" | 同上 |
| `paper-framework.md:19` Gap-3 | "AdaLoRA 4-bit 重要性不可靠" | **删或降为未检验假设** | AdaLoRA 实跑 2× 预算（rank 1024） |
| `paper-outline.md:84` | `r_l ∝ ‖E_l‖_F · I_l` | 改为 `score_l = ‖E_l‖_F ≡ ε_l‖W_l‖_F ∝ ‖W_l‖_F` | `adaq_precompute.py:115-122` |
| `paper-outline.md:91` | "命题3b 实测 ρ=0.9836 ✅" | 改"合成自测，非实测" | `generate_synthetic_ranks.py:40-52` |
| `paper-outline.md:123` / `framework:88-94` | 六策略公式表 | 改为实际取分式 + 披露只有 4 种不同分配 | `generate_ablation_configs.py:31-57` + 张量核对 |
| `paper-outline.md:146-148` | "高 ε 层获得更高秩 → 验证方法直觉" | **删**，换 §8.4 | ε 平坦 + Scheme A 非逐层 |
| `paper-outline.md:162` | "量化误差驱动的秩分配，零额外开销" | 换 §8.6 | 全链条否证 |
| `sota-comparison-protocol.md:127` | "GSM8K strict-match … AdaQ 优势主战场" | **删** | strict 是格式 artifact；adaq 在 strict 上亦非最优 |
| `sota-comparison-protocol.md:191-200` | 主表 Params(M) 四行全填 2.7M | 改为 3.41 / 3.57 / 3.80 / 6.82 M | 张量实测 |
| `sota-comparison-protocol.md:90` | "AdaLoRA 全局预算对齐到 R=512" | 标注**未达成**（实测 1024） | `adapter_model.safetensors` |
| `theory-review-report.md:140,142` | "含离群通道的层 ε_l 显著偏大…恰好支撑层间异质性动机"；"作为正面证据展示" | **删**（已被实测否证） | ε CV 1.48% |
| `scripts/aggregate_b4_3seeds.py:97-101` | 14B 预注册判定规则 | **补一条**：若 adaq 与 qlora 总秩不同（528 vs 768），任何"正向差异"表述必须同时声明预算不对齐，且只能用于参数效率讨论 | §5.2 |

---

## 9. 提案覆盖度逐条核对（`claim-revision-norm-aware.md` 144 行 vs 全部阴性/矛盾证据）

按行为红线要求，把仓库中**全部**阴性与自相矛盾证据列成 25 项，逐条核对提案是否覆盖。
判定：✅ 已覆盖（有可直接入正文的措辞） / ⚠️ 部分（记录了证据但没给正文措辞，或措辞不够） / ❌ 缺失。

| # | 阴性/矛盾证据 | 提案对应行 | 判定 | 缺口说明 |
|:--:|:---|:---|:--:|:---|
| N1 | NF4 口径 adaq 五者最差（GSM8K .4064 < 无 adapter 基线 .4124） | E1 :12-20 | ⚠️ | 证据齐备，但 §2 替换稿（:56/:60-68/:72/:76）中**无一段可直接放进 §4 或 §5 的正文**；本评审 §6.2(a) 补 |
| N2 | score 排序 99.8% 由范数决定 | E2 :22-34 + :66-68 + :81-87 | ✅ | 覆盖充分 |
| N3 | 双种子六策略不可区分 + 种子反转 + seed42 .4829 未复现 | E3 :36-48 + :72 + :142 + :144 | ✅ | 覆盖充分，禁用词明确 |
| N4 | 噪声底 4.5 点 > 策略极差 2.7 点 | :121-144 | ⚠️ | 4.5 点取自 **n=1** 的同配置配对；漏掉 error_only≡uniform 这对重复配置（实测差 0.08/0.83 点）⇒ 应改报区间 [0.9, 4.5]，见 §6.2(b) |
| N5 | strict-match 是格式 artifact | :114-115 + :142 规则3 | ✅ | 覆盖充分 |
| N6 | 14B 单种子 + 措辞上限 + 禁用句 | :98-117 | ✅ | 覆盖充分（但见 N9/N10） |
| N7 | n=3 无有效 Wilcoxon（最小 p=0.25） | L6 :76 仅提 n=2 消融 | ⚠️ | **主表 n=3 的统计限制未在提案中出现**，需并入 L7 |
| N8 | **AdaLoRA 实跑总秩 1024（2× 预算）、参数 6,816,768（2.0×）**，剪枝从未启动（tinit20/tfinal60 vs max_steps 21） | 无 | ❌ | 张量级实测（三个种子一致）；直接推翻 Gap-3 与协议 :90 的对齐声明 |
| N9 | **14B adaq 总秩 528 vs qlora 768（基线多 45.5%）**；`run_b4_qwen14b.sh:2` 注释"同预算 512"与产物矛盾 | 无 | ❌ | 提案 §4 用了 14B 数字却未核对秩预算；违反协议 :42/:78 |
| N10 | **等秩口径下 adaq 参数反而更多**：8B +11.5%（3,801,088 vs 3,407,872）、14B +6.8%（4,620,288 vs 4,325,376） | 无（:109 仍允许"少 26.5% 参数"） | ❌ | −26.5% 完全来自预算错配，与分配规则无关；"参数效率"这条存活贡献**方向反号** |
| N11 | **训练预算三表不同**：主表 21 步（seq2048+packing）、消融 126 步（seq1024）、14B 255 步（seq512） | 无 | ❌ | `trainer_state.json` 实测；主表阴性被训练预算混淆，14B 正向不能归因于规模 |
| N12 | **H1 幂律实测失败**：α ∈ [0.181, 1.591]，median 0.436，R²_median 0.802 | 无 | ❌ | `epsilon_l_report.json.powerlaw`；Prop2 的推导前提在真实模型上不成立 |
| N13 | **ε 实测 0.0936（min .0926 / max .1007），非假设的 0.02** | 无 | ❌ | 相差 4.7×；theory-review-report A2 与 outline §3.2 仍写 0.02 |
| N14 | **Prop3b ρ=0.9836 来自合成硬编码**，outline :91 标"实测 ✅" | 无 | ❌ | `generate_synthetic_ranks.py:40-52` rel_error 硬编码 0.025/0.018/0.015 |
| N15 | 实现 `score ≡ ‖E_l‖_F` ≠ 文档 `‖E‖·‖W‖` | :66-68 只给措辞 | ⚠️ | 未指出 outline :84 / framework :89-93 / 消融 docstring :5-9 三处公式本身写错 |
| N16 | **Scheme A 把 64 位点塌成 2 个整数**（`{v_proj:4, q_proj:12}`），"逐层自适应"从未被训练 | 无 | ❌ | 张量核对 q=12×32 / v=4×32；per_layer 方案 B 算了但没跑 |
| N17 | **六策略只有 4 种不同分配**（product≡importance_only `{4,12}`、error_only≡uniform `{8,8}`） | 无（仅 REPORT.md:53 提到前者） | ❌ | 交互项与误差项**均未被真正检验**；张量级已确认 |
| N18 | **相关工作缺 PARA（arXiv 2604.27796，撞名）与 AQLoRA（arXiv 2608.23816，独立同向阴性）** | 无（提案无相关工作章节） | ❌ | 全仓 grep 零命中；见 §7 |
| N19 | **秩预算扫描 R∈{128..1024} 标为"必须"（framework :75-82，~50h）但从未跑** | 无 | ❌ | 仓库无对应结果目录；outline §4.4 仍是空表 |
| N20 | PPL 结果 `sample_len=62`，全距 0.009 | 无 | ❌ | 不可进正文；如用须标 n=62 |
| N21 | 14B `peak_vram_gb` = 6.67（adaq 与 qlora 完全相同）< 8B 14.81，物理不可信 | 无 | ❌ | 效率表（outline §4.5 / 协议 :214-219）出数前必须重测 |
| N22 | 主表训练命令**不可复现**（`logs/` 无训练日志、`.bash_history` 无记录）；`train.py:217 save_steps=10` 与主表 checkpoint-{2,7,14,21} 不符 ⇒ 代码已漂移 | 无 | ❌ | 需补脚本入 git 并在论文标注 commit |
| N23 | **adaq 在 8B 主表三指标上均非最优**（flex/strict/MMLU 均为 DoRA 第一） | 无（:81-87 只禁"优于均匀分配"） | ⚠️ | outline :117 "bold best" 会把 DoRA 加粗，正文需明确承认 |
| N24 | Gap-3 被自家数据反驳（adalora flex .4824 ≈ adaq .4850，且它拿 2× 预算） | 无 | ❌ | 与 N8 同源 |
| N25 | 14B seed2025 MMLU 已出（adaq .784005 vs qlora .783649，+0.036 点，stderr .0033）、seed3407 无结果目录；预注册门 `aggregate_b4_3seeds.py:97-101` **无预算对齐条款** | 无 | ❌ | 本次评审首次提取；三种子齐前不得写任何 14B 结论 |

**覆盖度统计：✅ 4 项（N2/N3/N5/N6）· ⚠️ 5 项（N1/N4/N7/N15/N23）· ❌ 16 项。**
即提案对全部阴性/矛盾证据的**完全覆盖率 16%（4/25）**，含部分覆盖为 36%（9/25）。

**结论**：`claim-revision-norm-aware.md` 的**方向正确、措辞纪律优秀**（禁用词清单、措辞上限、预注册式表述都值得保留并推广到全项目），但它是围绕"score 由范数主导"这**一条**证据线写的，**没有覆盖理论前提失败（N12/N13/N14）、实验公平性（N8/N9/N10）、训练预算混淆（N11）、粒度塌缩（N16/N17）、相关工作撞名与独立佐证（N18）、缺失的必须实验（N19）**。按提案改完仍会被 Q1/Q2 审稿人在"你训了多少步""AdaLoRA 为什么是 2× 预算""你的逐层分配怎么只有两个值"这三个问题上击穿。

---

## 10. 目录级修改清单 + 编号修复清单

### 10.1 建议的新目录（与现 `paper-outline.md` 逐节对照）

| 现目录（outline） | 处置 | 新目录 | 关键内容来源 |
|:---|:---|:---|:---|
| Abstract :16-21 | **整体替换** | Abstract | §8.1 |
| §1 Intro ¶1-¶5 :27-31（Gap-1..Gap-4） | **重写** | §1 Introduction：①4-bit LoRA 的容量分配问题 ②"误差驱动"是领域的默认动机 ③**我们测出该动机的前提不成立** ④本文给出诊断+机制+可重复阴性+功效边界 ⑤贡献 4 条 | 删 Gap-4；Gap-1 改锚 importance CV 56.7%；删 Gap-3（N8/N24）；Gap-2 收窄（补 AQLoRA/PARA 后） |
| §2.1 :37-41 / §2.2 :44-46 / §2.3 :49-51 | **补 + 重排** | §2 Related Work：2.1 量化 PEFT（QLoRA/LoftQ/QA-LoRA/LQ-LoRA/IR-QLoRA/**AQLoRA**）2.2 秩分配（AdaLoRA/DyLoRA/FLoRA/**PARA**）2.3 **收敛的阴性证据**（新增小节，正面处理 AQLoRA 对照实验）2.4 矩阵分析工具（Eckart-Young/Weyl） | §7.1-7.5 |
| §3.1-3.2 :56-64 | 改数值 | §3.1 Preliminaries：NF4/E_l 定义 + **A1 已验证**（9.81e-06 vs 9.53e-04）+ **A2 实测 ε=0.0936** | `epsilon_l_report.json` |
| §3.3 :68-91 | **整体替换** | §3.2 Theory：Thm1（Weyl）/ Prop1'（谱抬升，**报 ≤0.7% 幅度**）/ Prop2（**条件命题 + H1 实测违反**）/ Prop3a（取整损失 + **Scheme A 14B +3.1%**）/ Prop3b（**降为合成自测**） | §8.3 |
| §3.4-3.5 :94-104 | 改公式 + 披露粒度 | §3.3 The Allocator：score 定义 → **退化引理** → 水填充分配 → **Scheme A（两个整数）披露** → 零搜索成本 | §8.3 |
| §4.1 :109-114 | **重写** | §4.1 Setup：三张表各自的 seq/packing/**global_step**/预算/参数量/评测精度，**明示三表不可横向比较** | §3.3 表 + §5 |
| §4.2 :117-120 | 改列 + 改措辞 | §4.2 Main comparison（等秩 512；**补 Params 列 3.41/3.57/3.80/6.82M**；**补 flexible-extract 列**；**AdaLoRA 标 2× 预算**）；结论"四方法在种子噪声内不可区分，DoRA 点估计最高" | 本次复算 |
| §4.3 :122-124 | **重写** | §4.3 Ablation：**6 策略 → 4 种实际分配**（附张量证据）+ 同配置重复跑噪声区间 [0.9, 4.5] + 六臂真实取分式 | §6.2(b) |
| §4.4 :127-128（R 扫描） | **补跑或删** | §4.4 Rank-budget sweep（R=128/256/512/768/1024，**等参数口径**）——协议标"必须"，当前无数据 | N19 |
| §4.5 :131-132 | 重测 | §4.5 Cost：预计算耗时实测 + 峰值显存**重测**（14B 6.67GB 不可信）+ **等秩参数开销 +11.5%/+6.8%** | N21/N10 |
| §4.6 :135 | 降级 | §4.6 14B case study：**等参数口径重述**（−26.6% 参数 / −31.3% 秩 → MMLU 平价 +0.036~0.057 点，stderr 0.33 点）；GSM8K flex 单种子**只作观察**；删除一切规模表述 | §5.2/N25 |
| §4.7 :137-142 | **保留（最佳节）** | §4.7 Statistical protocol：n=3 无有效 Wilcoxon、bootstrap CI、最小可检出效应量 | 现文本已规范 |
| §5 :146-148 | **整体替换** | §5 Discussion：什么在变/为什么误差别不出来/**收敛证据**/阴性的三条边界/要主张阳性还需要什么 | §8.4 |
| §6 :152-156 | **整体替换** | §6 Limitations L1–L10 | §8.5 |
| §7 :162-163 | **整体替换** | §7 Conclusion | §8.6 |
| 图表 :169-179 | 重做 | 图1 概览（**改为"score 分解"示意**）/ 图2 ε_l 与 importance 双面板（**替换原"高ε层→高秩"**）/ 图3 同配置重复跑 vs 策略极差（**新增，核心图**）/ 图4 global_step–噪声–效应量三角（**新增**）/ 图5 erank vs ‖E‖（补 PNG + 标 α 失配）/ 表1 主表（补 Params+flex）/ 表2 消融（补实际分配列）/ 表3 成本 / 表4 14B | — |
| — | **新增** | Appendix A：预算与参数审计（张量级）· Appendix B：三表训练配置 · Appendix C：ε/α 实测全表 · Appendix D：预注册判定规则 | §5/§3.3 |

### 10.2 编号修复清单

**P0（不改则拒稿；投稿前必须完成）**

| # | 修复项 | 具体动作 | 验收信号 |
|:--:|:---|:---|:---|
| P0-1 | 撤下"量化误差驱动"全部叙事 | 按 §8.8 表逐条删改（framework :10-12/:17-19、outline :17-21/:27-31/:84/:91/:123/:146-148/:162） | 全仓 grep "误差驱动\|error-driven\|高误差层" 在**论文正文**中零命中（历史文档可留但须标注已废弃） |
| P0-2 | 方法节公式与实现对齐 | outline :84 改为 `score_l = ε_l·I_l² = ‖E_l‖_F ∝ ‖W_l‖_F`；补 §8.3 退化引理 | 公式、`adaq_precompute.py:115-122`、消融取分式三者一致 |
| P0-3 | 披露训练预算（21/126/255 步） | 每张表脚注写 `global_step`、`num_tokens`、seq_len、packing；§4.1 明示三表不可横比；摘要加一句 | 三张表脚注齐全；不再出现"3 epochs"作为唯一预算描述 |
| P0-4 | AdaLoRA 预算违规处置 | 三选一：(a) 重跑 AdaLoRA 使 tinit/tfinal 落在 21 步内（推荐 `total_step=21, tinit=2, tfinal=12`）；(b) 主表删除 AdaLoRA 行；(c) 保留但标 "total rank 1024 (2×), pruning never engaged" 并**移出公平比较**、同步删 Gap-3 | 主表每行总秩列可核；Gap-3 从引言消失 |
| P0-5 | 14B 预算对齐 | 重跑 qlora@rank 5.5（或 adaq 提到 768）使总秩相等；在此之前 14B 只能写"等参数下的平价"并披露 528 vs 768 | `adapter_config.json` 总秩相等，或正文明示不对齐且不作分配效应结论 |
| P0-6 | 撤回"参数效率 −26.5%"作为方法贡献 | 改为等秩口径的 **+11.5% / +6.8% 参数开销**（§5.3），并把"参数效率"降级为"14B 单种子、预算不对齐下的观察" | 贡献 bullet（§8.2 第 4 条）替换提案 :60-62 对应句 |
| P0-7 | 补相关工作 PARA + AQLoRA | 按 §7.4/§7.5 入正文；AQLoRA 的独立对照写进 §2.3 与 §5； scholar 核查 "Zhou et al. 2025a/2026" 与 "DoRA: Dynamic Rank Distribution" 消歧 | 引用表含 2604.27796 与 2608.23816；两条待核查项有明确出处或被删 |
| P0-8 | 补跑或删除秩预算扫描（exp3） | 协议标"必须"（framework :75-82）。若 GPU 不足：删 §4.4 并在 Limitations L10 明写"未完成" | §4.4 有数据，或该节与 L10 一致 |
| P0-9 | 理论前提如实降级 | Prop2 改条件命题并报 α∈[0.181,1.591]/R²=0.802；Prop3b 的 ρ=0.9836 标"synthetic self-test"；ε₄ 全文改 0.0936；Prop1' 补"相对幅度 ≤0.7%"与 α=1.5 vs 实测 0.436 的失配说明 | outline :91 的"实测 ✅"消失；三处数值与 `epsilon_l_report.json` 一致 |

**P1（影响可信度，强烈建议投稿前完成）**

| # | 修复项 | 具体动作 |
|:--:|:---|:---|
| P1-1 | 消融表披露策略塌缩 | 表2 增列"realised rank pattern"，标明 product≡importance_only、error_only≡uniform（附 `lora_A` 形状证据）；六臂真实取分式写进 §4.3 |
| P1-2 | 噪声底改区间 | "4.5 点"改为 [0.9, 4.5] 并说明来自两组同配置重复跑、噪声非齐性；同步修 `aggregate_b4_3seeds.py:97-101` 的判定阈值 |
| P1-3 | 主表补 flexible-extract 列 | 现 `FINAL_REPORT.md` 只报 strict；flex 已可从原始 JSON 聚合（dora .4877 > adaq .4850 > adalora .4824 > qlora .4809） |
| P1-4 | 删"strict 是 AdaQ 优势主战场" | `sota-comparison-protocol.md:127` 删除；主判据统一为 flexible-extract + MMLU 并列描述 |
| P1-5 | 主表 Params 列改真值 | 协议 :191-200 的 2.7M 改为 3.41/3.57/3.80/6.82M；DoRA 标 †（magnitude 向量 163,840 参数） |
| P1-6 | 14B 三种子齐备后再写结论 | seed2025 已有 MMLU（.784005 vs .783649），**缺 GSM8K**；seed3407 **adapter 已训但无结果目录**。跑完后用 `aggregate_b4_3seeds.py` 出 `REPORT_14B.md`，并在预注册规则里补 P0-5 的预算条款 |
| P1-7 | 主表可复现性 | 把 8B 主表训练命令写成脚本入 git（现无脚本、无日志）；论文标 commit hash；解释 checkpoint-{2,7,14,21} 与 `save_steps=10` 的漂移 |
| P1-8 | 显存数据重测 | 14B `peak_vram_gb=6.67`（adaq 与 qlora 完全相同且低于 8B 的 14.81）不可信；效率表出数前用 `memory_budget.py`/`profile_training.py` 重测并记录度量点 |
| P1-9 | PPL 结果处置 | `sample_len=62`、全距 0.009 ⇒ 不进正文；如入附录须标 n=62 与不可解释性 |
| P1-10 | A1/A3 升格 + 引用产物 | A1 零均值已由 `epsilon_l_report.json` 证实（9.81e-06 / max 1.78e-05 vs mean_abs 9.53e-04）；A3 可报 within_var 3.15e-06 vs between_var 4.99e-08、lag-1 自相关 0.007。theory-review-report :198-206 的"待实测"清单应据此更新 |
| P1-11 | 图资产补齐 | `docs/figures/erank_vs_error.json` 有数据无 PNG；图4 需重画为 ε vs importance 双面板（`results_v2/epsilon_distribution.png` 可作基础） |

**P2（提升竞争力，返修期可做）**

| # | 修复项 | 具体动作 |
|:--:|:---|:---|
| P2-1 | **成本归一化分配**（新假设，最便宜的正向实验） | `r_l ∝ score_l / (d_in + d_out)`，在**等参数**口径下对比 uniform；8B 会把 q:v 从 12:4 翻到约 4:6。这是把阴性论文升级为"阴性+可行修正"的关键一步 |
| P2-2 | **per-site（Scheme B）跑一次** | `ranks.json` 的 per_layer_ranks（uniq [4,5,6,11,12,13]）已算好但从未训练；只需 product/uniform/random 三臂 × 2 种子，即可回答"粒度是否是阴性的原因" |
| P2-3 | **匹配步数重跑主表** | 关 packing（`--no-packing`）或提高数据量，使主表达到 ≥126 步，与消融同预算；这是让阴性可解释的最低成本路径 |
| P2-4 | **PARA 式事后 SVD 对照** | 对已训好的 adapter 做事后 SVD 重分配，检验失败在"信号时机"还是"预算" |
| P2-5 | ≥6 种子 | 若坚持任何显著性主张，必须 ≥6 对配对（协议 :170-178 已写明） |
| P2-6 | 8-bit 鲁棒性（exp6） | 协议标"可选"；在 ε 平坦的机制解释下，8-bit 的 ε 更小平坦更甚，可作为机制的**预测性验证**（成本低、说服力强） |

---

## 11. 期刊/会议定位复核

`paper-framework.md:134-148` 的定位（ICLR/NeurIPS ★★★★★ 首选，KBS/ESWA Q1 备选，Neural Networks Q2 底线）**在重构前不成立**，重构后需调整：

| 目标 | 重构后适配度 | 理由 |
|:---|:--:|:---|
| ICLR / NeurIPS 主会 | ★☆☆☆☆ | 主结果阴性 + AQLoRA 已占"zero-search 量化 LoRA 配方"的位 + PARA 已占"adaptive rank allocation"的位；除非完成 P2-1/P2-2/P2-3 拿到正向结果 |
| **TMLR** | **★★★★☆** | 其接收标准正是"claims 与 evidence 匹配、对受众有价值"，明确欢迎可复现阴性与方法学诊断；本文重构后天然契合（**注意：非 JCR 期刊，无影响因子**，需与单位的 JCR 要求核对） |
| **Knowledge-Based Systems（Q1）** | ★★★☆☆ | 可走"诊断 + 方法学协议"路线，但 KBS 审稿人通常期待正向方法贡献；须先完成 P2-1（成本归一化）给出可行修正 |
| **Neural Networks / Neurocomputing（Q1/Q2）** | ★★★★☆ | 分析型/阴性+机制类论文接受度较高；本文的 ε 平坦机制 + 幂律失败 + 功效分析是扎实的"为什么这类方法不工作"贡献 |
| **ESWA（Q1）** | ★★☆☆☆ | 偏应用与系统，阴性结果适配度低 |
| Workshop（NeurIPS ENLSP / ICLR TinyPapers / 负结果专场） | ★★★★☆ | 4 页版本可先发，占位并获取反馈，同时给 AQLoRA 的独立佐证一个正式引用锚点 |

**推荐路线**：先做 **P0-1..P0-9 + P1-1..P1-6**（纯写作与审计，几乎不占 GPU）→ 投 **Neural Networks / Neurocomputing 或 TMLR**；并行用夜间 GPU 做 **P2-1 + P2-2**（成本归一化 + per-site 各 3 臂 × 2 种子）→ 若拿到正向差异，升级到 **KBS** 并重写为"诊断 + 修正"双贡献论文。

---

## 12. 三句话总评

1. **现状态不可投**：三份写作计划仍以"量化误差驱动秩分配"为骨架，而仓库自己的实测数据（rel_error CV 1.48%、Spearman 0.998 对范数、NF4 口径 adaq 五者最差、六策略在 126 步下不可区分）已经把这个骨架证伪，同时方法节写的公式（‖E‖·‖W‖）与实现（≡‖E‖_F）、消融宣称的六臂与实际四种分配、协议宣称的等预算与 AdaLoRA 的 2× 秩 / 14B 的 528 vs 768 秩，三处**描述与产物不一致**——任何一处被审稿人发现都是诚信级问题。

2. **`claim-revision-norm-aware.md` 方向对但只覆盖了 16%（4/25）的阴性证据**：它把"名不副实"这一条处理得干净且纪律优秀（措辞上限 + 禁用词 + 预注册式表述），却完全没触及理论前提失败（α 跨 8.8×、ε 实测 0.0936 而非 0.02、ρ=0.9836 是合成值）、实验公平性（AdaLoRA 1024 秩、14B 768 秩）、训练预算混淆（21/126/255 步）、粒度塌缩（Scheme A 只有两个整数）、以及 PARA 撞名与 AQLoRA 的独立同向阴性——按它改完仍会在审稿第一轮被击穿。

3. **这个项目真正可发表的资产不是方法，而是"机制 + 收敛阴性 + 功效边界"**：相对 NF4 误差因块级 absmax 归一化而天然平坦（这是可证明的、且解释了 AQLoRA 独立观察到的"选哪些层不重要，选多少层才重要"），配合 21/126/255 步的训练预算披露和 0.9–4.5 点的同配置噪声区间，构成一份该研究方向的**最小可检出条件**报告；若再用 P2-1 的"成本归一化分配"（等秩下范数感知反而多花 6.8–11.5% 参数）拿到一个正向修正，论文就能从"阴性报告"升级为"诊断 + 修正"，达到 Q1 分析型期刊或 TMLR 的可投水平。

---

*本评审全部数值可复核：文档行号、脚本行号、`summary.json` / `results_*.json` / `adapter_config.json` / `adapter_model.safetensors`（张量形状）/ `trainer_state.json` / `epsilon_l_report.json` / `erank_vs_error.json` 均为实读；Spearman、CV、总秩、参数量、等秩参数对比为本次评审独立重算。外部文献仅 AQLoRA（arXiv 2608.23816）与 PARA（arXiv 2604.27796）经联网核实，其余标注"待核查"者不得凭记忆填写。*
