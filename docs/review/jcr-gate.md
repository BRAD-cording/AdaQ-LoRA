# AdaQ-LoRA 投稿前总闸门评审（JCR Gate Review）

> **评审人设定**：JCR Q1/Q2 深度学习期刊 reject 倾向审稿人（方法学增量判定 / 致命-可修复缺陷分离 / 拒稿理由预演 / 负结果发表价值裁定）
> **评审基准**：JCR Q2+ / 中科院 2 区「初稿提交后小修」水位
> **评审对象**：证据包（docs/ 7 份文档 + results_v2* 数据文件 + 代码抽查），**非成稿**——本评审裁定的是"若今日成稿投出，证据包能否支撑小修"
> **评审日期**：2026-11-15（UTC+8）
> **评审方式**：SSH 只读实读全部证据文件；每条判断绑定具体文件路径与数字，未读材料不下结论

---

## 0. 实读材料与关键数字（证据闭环清单）

| 材料 | 路径 | 本评审采用的关键数字/内容 |
|---|---|---|
| 8B 主表终报 | `results_v2/FINAL_REPORT.md` | n=3（seed 42/2025/3407）：adaq MMLU .6540±.0007 / GSM8K-strict .4766±.0031 vs qlora .6533±.0006 / .4711±.0046；dora .6543/.4799；adalora .6538/.4774；Wilcoxon 最小双侧 p=0.25（n=3 检验下限）；12/12 run 三 marker 齐全 |
| 双种子消融 | `results_v2_ablation_multiseed/REPORT.md` | 六策略 flex mean：error_only .5254 / uniform .5217 / random .5216 / product .5129 / equal_weight .5069 / importance_only .4989，极差 2.7 点；**噪声底 4.5 点**＝product 与 importance_only rank_pattern 完全相同（同一配置两次独立训练，s3407 flex \|.5246−.4792\|）；seed42 单种子 product .4829 > random .4435 **未复现**；strict↔flex 差 11–15 点为格式 artifact |
| NF4 评测 | `results_v2_ptq_nf4/summary.json` | adaq_nf4 GSM8K .4064＝五方法最差，且低于 baseline_nf4 .4124（纯 NF4 不挂 adapter）；MMLU adaq_nf4 .6335 亦低于 baseline .6364；qlora_nf4 .6334/.4124、dora_nf4 .6343/.4230、adalora_nf4 .6357/.4185 |
| 14B 单种子 | `results_v2_qwen14b/summary.json` | seed42：adaq MMLU .7843 / GSM8K flex .6588 vs qlora .7837 / .5254；strict .1827/.1304（度量格式合规非推理，不可用）；adapter 参数 4,620,288 vs 6,291,456（adaq 少 26.5%）；n_seeds=1，NF4 评测，OpenHermes-2.5[:2000] |
| 14B seed2025 中途 | `results_v2_qwen14b_seed2025/*/results_2026-11-15T15-3*.json` | **仅 MMLU 完成**：adaq .7840 vs qlora .7836（Δ+.0004，持平，与 seed42 模式一致）；GSM8K（判定主终点）未出；seed3407 目录未建 |
| PPL | `results_v2_ppl/summary.json` | adaq word_ppl 7.4314 vs qlora 7.4341（Δ.003）；NF4：adaq_nf4 7.9661 vs qlora_nf4 7.9691；**sample_len=62（全部方法）** |
| 系统指标 | `results_v2/system_metrics.json` | peak VRAM：qlora 14.80 / adaq 14.81 / dora 15.01 / adalora 14.84 GB；adapter 参数：qlora 3,407,872 vs **adaq 3,801,088（8B 上 adaq 多 11.5%）** |
| 主张重构提案 | `docs/claim-revision-norm-aware.md`（144 行） | E1–E3 证据链；rel_error CV 1.47%（0.0926–0.1008，Spearman .144）vs ‖W‖ CV 56.2%（13.33–75.27，Spearman .998）→ score 排序 99.8% 范数驱动；措辞上限/禁止清单；§4 14B 三条 Limitations；§5 交互效应结案（product .5129 低于 error_only .5254，差值均在 4.5 点噪声内） |
| 论文框架 | `docs/paper-framework.md` | 叙事仍为「量化感知秩分配…零额外显存开销下提升参数效率」；Limitations L1–L5；投稿定位首选 ICLR/NeurIPS，底线 Neural Networks(Q2) |
| 逐节大纲 | `docs/paper-outline.md` | 标题 "AdaQ-LoRA: **Quantization-Aware** Adaptive Rank Allocation…"；§4.7 已明令禁写 Wilcoxon/显著性；Allocation Rule 写作 r_l ∝ ‖E_l‖_F·I_l；实验矩阵含实验三（秩预算扫描，标注"必须"） |
| 理论审查报告 | `docs/theory-review-report.md`（206 行） | C1/C3 原有错误已修复：Theorem 1=Weyl 双向扰动界（严格）、Prop 1' 需 H_isotropic、Prop 3b 为经验性（自测 ρ=0.9836 系**合成数据**）；§6 清单未勾项：A2 实测（现已被 E2 覆盖）、Prop 3b 真实模型实测 |
| SOTA 对比协议 | `docs/sota-comparison-protocol.md` | 四个 Gap 检索状态**全部「待 scholar 核查」**；口径 A：总秩 R=512、q_proj+v_proj 64 位点、基线 r̄=8；§5 功效硬约束：n=3 时 Wilcoxon 最小 p=0.25，禁止一切推断性表述；主表模板 Params 写 2.7M（与实测 3.4/3.8M 不符） |
| 评测口径 v2 | `docs/eval-protocol-v2.md` | 主表口径＝**bf16 基座+adapter（非 4-bit）**；14B/NF4 表口径＝NF4 加载——跨表口径不同；v2.0「假表」事故复盘（marker 掩盖 MMLU 失败）已修复至 v2.1；§6：MMLU 对 2000 条 SFT 不敏感，不能作方法有效性主证据 |
| 代码抽查 | `adaq-lora-research-pipeline/scripts/adaq_precompute.py:122` | `stats[name] = {"rel_error": rel, "importance": norm_w, "score": rel * norm_w}` → **score = rel_error × ‖W‖_F**；`allocate_ranks` 迭代水位法、截断 [4,128]、总量守恒 |
| 秩分配产物 | `ranks.json` / `ranks_qwen14b.json` | budget=512、r_min=4、r_max=128；v_proj 层普遍 r=4（下界），与范数驱动一致 |
| 全库检索 | grep（*.md 全库） | `AQLoRA`、`PARA`、`2604.27796`、`2608.23816`、`同号`、`预注册` 均 **0 命中** |

---

## 1. 总判定（Verdict）

# **Major Revision（reject 倾向）**

三个子判定：

1. **今日按现状成稿投 Q2+ → 大概率直接 reject，且属"审稿人不需要争论"的那类 reject**：标题/README/框架层主张（"量化感知…提升参数效率"）被项目自己的 E1–E3 数据证伪（`claim-revision-norm-aware.md`），文实相反是 desk-reject 级缺陷；最危险反例 AQLoRA 全库零引用；阴性结论无任何等价性统计支撑。
2. **科学内核评估**：机制分解（rel_error CV 1.47% → score 99.8% 范数驱动）与噪声底标定（同 rank_pattern 对偶 → 4.5 点）**是真贡献**，质量高于多数 Q2 方法论文的消融节；"方法有效"叙事已死，"诚实证伪+机制解释"叙事活，但当前统计功效撑不起"不可区分"这一等价性断言。
3. **离「初稿后小修」的距离**：§5 的 **P0 六件**（2 件写作/落档 + 4 件需 GPU 机时），完成后才进入小修区间。

---

## 2. 七问逐项判定

### ① 研究问题清晰度 —— 重构后清晰，现状人格分裂

- **重构后的问题**（依 `claim-revision-norm-aware.md` 方向 B）："冻结 NF4 基座上，预计算的误差×范数打分驱动的非均匀秩分配，是否优于均匀/随机分配？"——可证伪、已被回答（8B：否）、答案有机制解释。**这是合格的研究问题。**
- **现状缺陷**：三套问题陈述并存且互斥——`paper-framework.md`/`paper-outline.md`/`README.md` 仍是"利用量化误差层间异质性提升参数效率"（该前提被 E2 直接证伪：rel_error 层间 CV 仅 1.47%，**异质性不存在于相对误差项**）；`claim-revision` 主张反向；标题仍含 "Quantization-Aware"。重构提案状态明写"**提案，未改正文**"。
- **评级**：fatal-mandatory（F1），但修复成本低（纯写作，替换表已在 claim-revision §2 逐段给出）。

### ② 动机-方法-证据对齐（重构后）—— 部分对齐，五处脱节

**对齐的部分**：E1（NF4 主场失效）+ E2（机制分解）+ E3（双种子消融阴性）三线互洽，构成"假设→检验→证伪→机制解释"完整闭环；NF4 最差结果（.4064 < baseline .4124）在旧叙事下是尴尬，在新叙事下恰是**支持性证据**——重构方向正确。

**未对齐的五处**：

| # | 脱节点 | 证据 | 后果 |
|---|---|---|---|
| a | **理论章节悬空**：Theorem 1（Weyl）/Prop 1'（谱抬升）/Prop 2（KKT 边际条件）全部为"误差驱动分配"辩护，重构后没有任何命题服务"范数退化"这一实际结论 | `theory-review-report.md` §3；`paper-outline.md` §3.3 | 理论从资产变装饰；审稿人会问"这些定理支撑了哪个被检验的预测？" |
| b | **Prop 3b 的 ρ=0.9836 是合成数据自测**，真实模型实测至今未做 | `paper-framework.md` L2"自测 ρ=0.9836（合成数据）"；`theory-review-report.md` §6 未勾项 | 若论文引用 ρ=0.9836 暗示真实验证 → 诚信级问题 |
| c | **公式文档-代码不一致**：代码 `score = rel_error × ‖W‖_F`（`adaq_precompute.py:122`）；outline 写 `r_l ∝ ‖E_l‖_F·I_l` = rel_error×‖W‖²，差一个 ‖W‖ 因子 | 两文件对读 | 公开代码后审稿人一眼识破；两版公式的范数主导程度不同，连 E2 的"99.8%"归属都需重新表述 |
| d | **自标"必须"的实验三（秩预算扫描 R=128–1024）未跑**：results* 目录中无任何 budget sweep 产物 | `paper-framework.md` 实验矩阵；顶层 ls 核对 | 丢掉唯一能提升统计功效的实验维度（低预算下分配权重最大）；必配图 2 无来源 |
| e | **"系统收益=推理不劣化"无数据**：只有 peak VRAM（14.80 vs 14.81 GB，本身无差异），无 time/epoch、预计算分钟实测、推理 tok/s | `system_metrics.json` 仅 3 字段 | 效率表（protocol §6）四列缺三列；PPL sample_len=62 也撑不起"不劣化" |

### ③ 对 AQLoRA/PARA 的新颖性定位 —— 窄幅存活，当前零落档

- **核查事实**：全库 grep `AQLoRA`/`PARA`/`2604.27796`/`2608.23816` **0 命中**；`sota-comparison-protocol.md` 四个 Gap 检索状态全部"待 scholar 核查"。即 Gap-0（"冻结量化+预计算误差+零搜索零训练秩分配"无人占据）目前**只存在于会话口径，无可复核落档**——而该协议自己规定"每次正式写作/投稿前重查一次，附检索日期"。
- **AQLoRA 的杀伤面**：其"NF4 reconstruction error 选层 not better than random"已抢先发表"误差信号无效"阴性发现的一半。若论文不主动引用并划清 delta，审稿人一句"see AQLoRA"即可击落新颖性。
- **存活的 delta（三点，均可由现有数据支撑）**：
  1. **机制 vs 现象**：AQLoRA 报告"无效"，本项目给出"为什么无效"的定量分解——rel_error 层间 CV 1.47%（近常数）× ‖W‖ CV 56.2% → Spearman .998 vs .144，乘积打分**数学上退化为范数感知**（E2）。这是对文献的增量知识，不是一句观察。
  2. **对象不同**：AQLoRA 做层**选择**（二元），本项目做预算**分配**（连续水位法+截断守恒，`allocate_ranks`）；PARA 作用于**已训练** LoRA（post-hoc），本项目是**事前零训练**（pre-compute <5 min）。
  3. **评测方法学**：同 rank_pattern 对偶标定噪声底（4.5 点 > 策略极差 2.7 点 → 全表不可区分）+ strict/flex 格式 artifact 定量（11–15 点）——对 PEFT 评测社区有独立工具价值。
- **裁定**：新颖性以"**诊断/机制/评测方法学论文**"身份窄幅存活；以"方法论文"身份不存活。前提条件：Related Work 必须新增 AQLoRA/PARA delta 段 + Gap-0 检索落档（含日期、检索词、库）。

### ④ 阴性结果+机制解释能否独立撑起 Q2 —— 当前不能；补齐三件后能

**当前不能的根因：低功效伪影（low-power artifact）防线未建。** 敌意审稿人的最强攻击是："你在一套**任何方法都不可能赢**的设置里宣布证伪"——且证据全部来自项目自己的文件：

- 2000 条 OpenHermes SFT 动不了 MMLU（`eval-protocol-v2.md` §6 自认，方法间差 ≤.001）；
- IFEval 全员地板（.1263–.1375，`FINAL_REPORT.md` 主表）；
- strict-GSM8K 被格式污染（ablation REPORT 补充节：importance_only strict .3101 / flex .4792）；
- NF4 评测下**所有** adapter 的 MMLU 都低于不挂 adapter 的 baseline（.6334–.6357 vs .6364，`summary.json`）——SFT 本身在该口径下是负收益。

"未检出差异"≠"差异不存在"。要把阴性结论做成可发表的**等价性断言**，必须补：

1. **TOST 等价检验**，预设边界＝噪声底 4.5 点（该边界有实证来源：同配置对偶，非拍脑袋）；
2. **每策略 ≥3 种子**（当前消融 n=2；噪声底本身只有 1 对估计，需 ≥2 对复算带 CI）；
3. **实验三低预算扫描**（R=128/256）：预算越小，分配方式的杠杆越大——这是假设最敏感区；若 R=128 下 product 与 uniform 仍等价，阴性结论才真正"打死了假设"而不是"打死了自己的功效"。

**补齐后能撑起的理由**：机制分解（E2）+ 噪声底方法学（③-3）+ 跨规模证伪（8B n=3 + NF4 口径 + 14B 三种子）构成"诊断-机制-方法学"三位一体，恰好是 TMLR（明确欢迎 negative results）与 Q2 理论+实验型期刊（Neural Networks 类）的收稿形态。**不需要依赖 14B 正向信号**——恰恰相反：把主叙事押在 14B 单种子正向（flex .6588 vs .5254）上是全包最危险的选项，8B seed42 已有先例（product .4829 > random .4435 扩种子后蒸发，ablation REPORT 结论 5）。

### ⑤ 若 14B 三种子阴性，论文还剩什么 —— 故事降档有限，仍可发，档位取决于必修清单

- **剩余故事**＝「**跨规模一致证伪**」：8B（主表 n=3 + 消融 6 策略 + NF4 主场失效）与 14B（n=3）两个规模上，误差/范数驱动分配与均匀/随机不可区分；机制＝NF4 相对误差谱平坦性（CV 1.47%）；附赠两件评测方法学工具（噪声底标定、strict/flex artifact）。这是完整、自洽、可复核的 falsification paper。
- **中途信号**（本评审时点）：seed2025 MMLU adaq .7840 vs qlora .7836——持平，与 seed42 模式一致；GSM8K flex（判定主终点）未出，seed3407 未开跑。按 8B 先例（flex 差全部落在噪声内），"三种子配对差同号且均值>4.5 点"的阳性判定**大概率不成立**。
- **够不够发**：够发"扎实阴性+方法学"论文；现实档位 **TMLR > Q2（Neural Networks 类）> Q1**。硬条件：④的三件（TOST/≥3 种子/低预算扫描）必须完成，否则连阴性故事也只是"描述性观察"。
- **程序性警告（必须立即处理）**：预注册判定规则（配对差三种子同号且均值>4.5 点）**全库无落档**（grep 0 命中），而 seed2025 的 MMLU 已出数——部分揭盲已经发生。规则若不立即带时间戳落档，14B 无论出什么结果都将失去 confirmatory 资格，沦为可被质疑为 HARKing 的探索性分析。

### ⑥ 致命缺陷 vs 可修复缺陷

**FATAL（不解决不可投；▲＝需新实验/GPU 机时）**

| # | 缺陷 | 证据锚点 | 修复路径 |
|---|---|---|---|
| F1 | **文实相反**：标题/README/框架仍卖"量化感知…提升参数效率"，被自家 E1–E3 证伪；重构提案未执行 | `paper-outline.md` Title；`README.md` 核心方法段；`claim-revision` 状态行"提案，未改正文" | 执行 claim-revision §2 全部替换表（含"必须删除/降级表述"五行表）；标题弃 "Quantization-Aware" |
| F2▲ | **无等价性统计**：主表 n=3（Wilcoxon min p=0.25，`FINAL_REPORT.md` §3.3），消融 n=2，噪声底单对估计；"不可区分"仅描述性成立 | `sota-comparison-protocol.md` §5 功效硬约束 | 每策略 ≥3 种子 + TOST（边界=4.5 点）+ 噪声底 ≥2 对复算带 CI |
| F3▲ | **实验三（秩预算扫描）缺失**：自标"必须"未跑；假设最敏感区（低预算）无数据；必配图 2 无来源 | `paper-framework.md` 实验矩阵；results* 目录核对 | R∈{128,256,512,768,1024} × {adaq,uniform,qlora} × ≥2 种子 |
| F4 | **新颖性零落档**：AQLoRA/PARA 0 引用；四 Gap"待 scholar 核查"；Gap-0 无检索记录 | 全库 grep；`sota-comparison-protocol.md` §1 | 按协议 §1 检索词重查+落档（日期/库/命中）；Related Work 增 delta 段（③的三点） |
| F5▲ | **单模型单规模**：核心阴性结论仅系于 Llama-3-8B；14B 未收口（seed2025 半程、seed3407 未跑）；跨模型（实验五）未做 | `results_v2_qwen14b_seed2025/` 状态；outline 实验五"可选" | 最低限：14B 三种子收口+按预注册判定；建议：第二家族（Mistral/Qwen2-7B）至少做消融级（product vs uniform vs random） |

**FIXABLE（必修/高优，但非一票否决）**

| # | 缺陷 | 证据锚点 | 修复 |
|---|---|---|---|
| R1 | 公式文档-代码不一致（rel×‖W‖ vs rel×‖W‖²） | `adaq_precompute.py:122` vs `paper-outline.md` §3.3 | 以代码为准全库统一；E2 的"99.8%"表述随之核对 |
| R2 | PPL sample_len=62，Δ.003 不可解读 | `results_v2_ppl/summary.json` | 扩全量 WikiText-2 重跑，或删除该证据线 |
| R3 | 效率表四缺三：无 time/epoch、预计算实测、推理 tok/s | `system_metrics.json` 仅 params/VRAM/ratio | 补实测；否则"系统不劣化"降为"显存不劣化" |
| R4 | 参数公平口径：8B adaq 3.80M > qlora 3.41M（+11.5%，总秩对齐≠参数对齐）；protocol 模板 2.7M 与实测不符 | `system_metrics.json`；`sota-comparison-protocol.md` §6 | 论文报实际参数+解释形状差异；修正模板 |
| R5 | 跨表口径混杂：8B 主表=bf16+adapter，8B NF4 表与 14B=NF4 加载；8B seq2048 vs 14B max-length 512 | `eval-protocol-v2.md` §1；`claim-revision` §4 | 逐表声明口径；禁止跨表比较；规模故事需承认训练长度混杂 |
| R6 | 理论重排：Theorem1 降为背景；Prop3b 合成 ρ 不得暗示真实验证；建议新增**退化命题**（rel_error∈[c(1±δ)] ⇒ rank(score)=rank(‖W‖)）给机制结论数学形态 | `theory-review-report.md` §5–§6 | 理论节重构为"诊断的理论基础"而非"方法的最优性辩护" |
| R7 | IFEval 地板效应（.126–.138） | `FINAL_REPORT.md` 主表 | 解释（2000 条 SFT+格式漂移）或删除该任务 |
| R8 | **预注册规则未落档**且 seed2025 MMLU 已部分揭盲（此项介于 fixable 与 fatal 之间，拖延即成 fatal） | grep "同号/预注册" 0 命中 | **立即**落档规则全文+时间戳+聚合脚本版本，晚一天贬值一天 |
| R9 | 主表 GSM8K 仅报 strict，而消融已证 strict 格式污染（11–15 点 artifact） | `FINAL_REPORT.md` vs ablation REPORT 补充节 | 主表改双口径（strict+flex），flex 为主判据 |

### ⑦ 投稿前必修排序 —— 见 §5

---

## 3. 拒稿理由预演（敌意审稿人 Top-6 攻击点）

| # | 攻击点（审稿人视角最强表述） | 当前防御 | 修复后防御 |
|---|---|---|---|
| 1 | "你的阴性是低功效伪影：MMLU 动不了（你自己文件承认）、IFEval 地板、strict 被格式污染、NF4 下 SFT 全员负收益——这套设置里**没有任何东西能赢**，你证伪的是你的实验设置，不是假设。" | **无防御**（最致命） | F3 低预算扫描 + F2 TOST 等价 + （建议）一组高功效设置（更大 SFT 或指令敏感任务） |
| 2 | "AQLoRA 已发表 NF4 误差选层≈随机；你的阴性发现被抢跑，机制解释不过是一行观察（误差是常数）。" | **无防御**（零引用） | F4 delta 三点：机制定量化（CV/Spearman 分解）、分配 vs 选择、噪声底方法学 |
| 3 | "所谓方法是范数比例分配——trivial，magnitude-based 分配思想人尽皆知，方法贡献为零。" | 部分（claim-revision 已放弃方法主张） | 明确以诊断论文定位；方法节降格为"被检验的规则"；贡献三条全部指向诊断/理论/阴性实证 |
| 4 | "n=2 消融、n=1 的 14B、n=3 主表配不出任何 p 值；'不可区分'是 absence of evidence。" | 部分（文档已禁写显著性，诚实） | F2：TOST 把 absence of evidence 变成 evidence of absence（边界 4.5 点有实证来源） |
| 5 | "跨规模比较混杂：8B 用 bf16 评测/seq2048，14B 用 NF4 评测/max-length 512；8B 上你的方法参数反而多 11.5%。规模故事不成立。" | 部分（各文件内部已声明口径） | R4+R5：逐表口径声明+参数实测表+承认混杂入 Limitations |
| 6 | "理论节（Weyl/谱抬升/KKT）与任何被检验的预测无关；Prop 3b 的 ρ=0.9836 是合成数据。理论是装饰。" | 部分（theory-review 已降级表述） | R6：理论重排为诊断基础 + 退化命题；Prop3b 补真实模型实测或删除数值 |

---

## 4. 14B 结果两分支预案（预注册判定的写作上限）

**Branch A —— 阳性（三种子配对差同号且均值>4.5 点；按 8B 先例概率低）**：
- 允许写：规模依赖效应的初步证据（8B 阴性 + 14B 阳性），仍不得写"AdaQ 优于 QLoRA"（跨口径混杂 R5）；
- 必须写：三条 Limitations（`claim-revision` §4 已备）+ 8B 阴性全文保留 + 阳性机制解释义务（为什么 14B 出现？‖W‖ 分布随规模变化？需数据）；
- 论文形态：阴性主体 + 规模调节效应节，故事升级为"效应的边界条件"。

**Branch B —— 阴性（大概率）**：
- 允许写：跨规模一致证伪（8B+14B）；
- 禁止写：任何"趋势上 adaq 略优"类措辞（seed42 flex .6588 与 seed2025 MMLU .7840 的单种子/单指标优势不得进入结论层）；
- 论文形态：§2-⑤ 所述 falsification + 方法学论文；档位 TMLR/Q2。

两分支共同前置：**R8 预注册落档必须在 seed2025 GSM8K 出数之前完成**（MMLU 已揭盲，主终点 GSM8K 尚未——现在落档还来得及保住主判定的 confirmatory 资格）。

---

## 5. 投稿前必修排序（P0 阻断 / P1 强烈建议 / P2 建议）

| 优先级 | 项 | 内容 | 验收信号 | 类型 |
|---|---|---|---|---|
| **P0-1** | R8 | 预注册判定规则立即落档（规则全文+时间戳+脚本版本） | docs/ 内可 grep 到规则原文；时间戳早于 seed2025 GSM8K 出数 | 落档，半天 |
| **P0-2** | F1 | 执行主张重构：标题/README/framework/outline/摘要全部替换为 norm-aware+阴性叙事（含 claim-revision §2 删除表） | 全库 grep "量化感知.*提升\|Quantization-Aware" 0 命中（历史文档标注 superseded 除外） | 写作，2–3 天 |
| **P0-3** | F5a | 14B 三种子收口：seed2025 GSM8K + seed3407 全流程，按预注册规则判定，走 §4 分支 | `results_v2_qwen14b/summary.json` 更新为 n_seeds=3 + 判定记录 | GPU，排队中 |
| **P0-4** | F2 | 等价统计升级：消融每策略 ≥3 种子；TOST（边界=4.5 点）；噪声底 ≥2 对复算带 CI | 分析报告含 TOST p 值与等价 CI；噪声底 CI 不覆盖任何策略差 | GPU+分析，约 1–2 周机时 |
| **P0-5** | F3 | 秩预算扫描：R∈{128,256,512,768,1024} × {adaq,uniform,qlora} × ≥2 种子 | 图 2 产出；低预算区等价/差异结论明确 | GPU，约 1 周机时 |
| **P0-6** | F4 | 新颖性落档：Gap-0 检索记录（日期/库/检索词）+ Related Work 增 AQLoRA/PARA delta 段 | docs/ 内可复核检索档案；delta 三点逐点有引文与本文数据对应 | 检索+写作，2–3 天 |
| P1-1 | R1 | 公式统一（以 `adaq_precompute.py:122` 为准） | 全库公式一致，E2 表述复核 | 写作 |
| P1-2 | R4+R5 | 参数实测表 + 逐表口径声明（bf16/NF4、seq 长度、SFT 数据） | 每张表脚注含口径五要素 | 写作 |
| P1-3 | R9 | 主表补 flex 双口径 | summary.json 聚合含 flex | 分析（数据已有则半天） |
| P1-4 | R3 | 效率表补实测（time/epoch、预计算分钟、tok/s） | 表 3 四列齐 | 轻量实测 |
| P1-5 | F5b | 第二模型家族消融级复验（Mistral-7B 或 Qwen2-7B：product/uniform/random × ≥2 种子） | 阴性结论跨家族成立 | GPU |
| P2-1 | R2 | PPL 扩全量或删除 | sample_len ≥ 数千 或正文无 PPL 主张 | 轻量 |
| P2-2 | R6 | 理论重排 + 退化命题 + Prop3b 真实模型实测或删除 | 理论节每个命题对应一个被检验/被使用的断言 | 写作+轻量计算 |
| P2-3 | R7 | IFEval 地板效应解释或删除 | 正文有明确处置 | 写作 |
| P2-4 | — | 方差稳定性观察（product std ±.0032 最小）：作为"可复现地不赢但稳"的次级发现，配 bootstrap 方差比检验 | 有量化支撑再写，无则不写 | 分析 |
| P2-5 | — | 投稿定位重置：`paper-framework.md` "首选 ICLR/NeurIPS"对本证据包不现实；改双轨——TMLR（阴性友好，投稿前核实其当前索引/分区状态）/ Neural Networks 类 Q2（需 P0 全完成）/ workshop 保底（NeurIPS ENLSP 类） | venue 决策文档更新 | 决策 |

---

## 6. 三句话总评

1. 这是一份**科学上诚实、工程上规范、但叙事上错位**的证据包：机制分解（rel_error CV 1.47% / Spearman .998 vs .144）与噪声底标定（同 rank_pattern 对偶 → 4.5 点 > 策略极差 2.7 点）是高于 Q2 平均水位的真贡献，而主文档至今还在出售一个被自家数据证伪的假设。
2. 阴性结论**能够**独立撑起 Q2/TMLR 级论文、且不依赖 14B 正向信号——但前提是完成等价性升级（TOST+≥3 种子）、低预算扫描和第二规模收口，把"未见差异"锻造成"统计等价"；押注 14B 单种子正向（flex .6588）是包内最危险的选项，8B seed42 已演示过单种子阳性如何蒸发。
3. **Verdict：Major Revision（reject 倾向）**——今日投出即 reject；完成 P0 六件（预注册落档、叙事重构、14B 收口、等价统计、预算扫描、新颖性落档，其中四件需 GPU 机时）后，才具备"初稿提交后小修"的入场资格。

---

*评审依据文件清单见 §0；本评审未改动仓库任何既有文件，仅新增本文件（docs/review/jcr-gate.md）。*
*Reviewer: JCR_Q1 深度学习评审专家（reject-leaning persona）· 2026-11-15*
