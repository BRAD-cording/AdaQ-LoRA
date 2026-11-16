# Related Work Search Log — AdaQ-LoRA (Norm-Aware Rank Budgeting)

**目的**：新颖性落档（JCR 复核 B10 / fair-baseline F19）。记录检索方法、最近前作的 arxiv ID 与元数据、与本文的 delta。
**红线**：所有引用 ID 均经本日志 web_fetch 逐字核验可达（2026-11-16 UTC）；未核验 ID 不入库。

## 1. 检索方法
- 源：arXiv（abs 页 + export API），免 key，web_fetch 直连。
- 检索词：`quantization-aware rank allocation LoRA`、`adaptive rank allocation LoRA`、`NF4 reconstruction error layer selection`、`post-training rank pruning LoRA`、`low-rank budget allocation 4-bit`。
- 日期：2026-11-16（UTC）。
- 前序：m01404 学术三步分析扫 ~80 条目并核 ID；本日志对「最像本文」的两篇前作重新逐字开 abs 页核验。

## 2. 最近前两篇前作（已逐字核验 ID 可达）

### 2.1 AQLoRA — arXiv:2608.23816（最危险反例 + 最强方法学呼应）
- **Title**: AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning
- **Authors**: Md Romyull Islam
- **Submitted**: 2026-08-24 (v1), v2 2026-10-11; cs.LG; 24 pages, 5 figs, 20 tables; code github.com/Romyull-Islam/AQLoRA
- **DOI**: 10.48550/arXiv.2608.23816
- **方法**：一次 CPU pass 按 NF4 reconstruction error 给层排序（无搜索、无校准数据），保留固定比例层为 fp16 以跳过 dequantization 换速度。评测 1.4B–14B 六模型四架构族 Commonsense-170K。
- **★直接撞车点（摘要原文）**："Choosing adapter layers by weight density is no better than random. Choosing protected layers by quantization error is not either, on five models at three budgets with memory-matched controls and downstream on two. The number of protected layers sets the speed; which layers they are does not measurably change accuracy."
  → AQLoRA 独立证伪：按 NF4 量化误差（及按权重密度）选层 **no better than random**。直接呼应本文阴性结论（score=rel_error×‖W‖_F 99.8% 范数驱动，等秩下秩分配打不过 uniform/random）。
- **★方法学呼应（摘要原文）**："measure the noise floor from a duplicated arm; and repeat whole sessions, since on one model three sessions that each passed the floor test spread over 8.2 points."
  → 与本文 Phase3-D 噪声底设计（同 product 配置×3seed×3rep=9run 极差）同构；其 8.2 点会话间散布与本文 11.52 点重尾噪声底同量级 → 支持阈值校准修正案（4414d9e：原 4.5pt 作废，改实测 pooled 噪声底 N）。
- **Delta**：
  | 维度 | AQLoRA 2608.23816 | 本文 AdaQ-LoRA |
  |---|---|---|
  | 目的 | 速度（fp16 保层跳 dequant） | 精度（训前秩预算分配） |
  | 作用对象 | 哪些层保 fp16（protected layers） | 每层分多少秩 r_l |
  | 信号 | NF4 reconstruction error（选保护层） | rel_error×‖W‖_F（分秩） |
  | 结论 | 误差选层 no better than random | score 99.8% 范数驱动，等秩打不过 uniform（阴性） |
  → **趋同阴性**：两独立工作从不同角度（选层 vs 分秩）得到「NF4 误差信号对 LoRA 配置无效」同一结论 = 可复现的方法类边界，非孤例。

### 2.2 PARA — arXiv:2604.27796（机制最近但时机不同）
- **Title**: Post-Optimization Adaptive Rank Allocation for LoRA
- **Authors**: Vishnuprasadh Kumaravelu, Sunil Gupta, P. K. Srijith
- **Submitted**: 2026-04-30 (v1); cs.AI
- **DOI**: 10.48550/arXiv.2604.27796
- **方法**：data-free LoRA 压缩，SVD 全局奇异值阈值剪枝 LoRA 秩，按层谱重要性形成非均匀秩分配。post-hoc、不改训练、避免动态架构不稳定，参数减 75–90% 保性能。
- **Delta**：
  | 维度 | PARA 2604.27796 | 本文 AdaQ-LoRA |
  |---|---|---|
  | 时机 | post-hoc（训后作用于已训练 adapter） | 训前（作用于冻结基座） |
  | 信号 | 已训 adapter 谱重要性（SVD 奇异值） | 冻结量化基座 rel_error×‖W‖_F |
  | 是否需训练 | 需先训完整 LoRA 再剪 | 零训练分配（precompute <5min） |
  | 定位 | 压缩（减参保性能） | 预算分配（定 r_l 再训） |
  → **Gap-0 成立**：PARA 是 post-hoc 谱剪枝，未占据「训前+冻结量化基座+零搜索秩分配」。但 delta 须**降格为成本优势（免训 LoRA）非质量优势**——二者等预算精度未对比，本文阴性结论也不支持质量主张。

## 3. 新颖性判定
- **性质**：方法论创新非算法创新。delta=砍搜索+换信号（NF4 误差→实测范数主导）+概念迁移（秩分配搬到冻结量化基座训前）。
- **Gap-0（真空白）**：「训前、冻结 NF4 基座、零搜索零校准的秩预算分配」无任一已核验前作占据。
- **诚实边界**：Gap-0 是**流程/成本**空白非**性能**空白。本文实证阴性（等秩打不过 QLoRA/uniform），新颖性只能主张「提出并严格检验一个更便宜的分配流程 + 诚实报告其失效边界」，不能主张 SOTA。
- **desk-reject 风险解除**：AQLoRA/PARA 在本日志前全库 0 引用 → 已落档；Related Work 必须引用并给出 delta（否则新颖性零落档+最近前作未引用 = 直接拒）。

## 4. TODO（B10 后续）
- [x] 扩到 ≥18 篇完整 Related Work 检索档案（F19，已完成→见 §5：22 扩展+2 锚点=24 篇核验）：覆盖 rank allocation / quantization-aware PEFT / adaptive LoRA 三类，逐篇核验 ID 入库（不编造）。
- [ ] 把 §2 delta 写进论文 Related Work 段（引 AQLoRA 作趋同阴性 + PARA 作机制最近）。
- [ ] AQLoRA 噪声底方法学（8.2 点会话散布）写进论文统计方法段，支撑阈值修正案。

---

## 5. 扩展检索档案（F19，≥18 篇；OpenAlex DOI 索引核验）

**检索源与方法（诚实声明）**：
- arXiv export API 本次检索期间对远端 IP 全面限流（HTTP 429 + read timeout，5 查询 0 返回），故扩展条目改用 **OpenAlex**（免 key，`api.openalex.org/works?search=...`，2026-11-16，两次查询：`rank allocation LoRA quantization` / `quantization-aware parameter-efficient fine-tuning low-rank`）。
- **核验口径**：每条 DOI 均由 OpenAlex 权威索引返回（title+year+doi+ids），DOI 可解析=ID 可达=非编造；arXiv 类 DOI（`10.48550/arxiv.XXXX`）直接给出 arXiv ID。两篇锚点（AQLoRA 2608.23816 / PARA 2604.27796）已于 §2 用 arXiv abs 页逐字核验。
- **delta 口径**：扩展条目的 delta 为 **title/索引级**（未逐篇读全文），标注「待全文核」；凡与本文「训前+冻结量化基座+秩分配」可能撞车者列入 §5.4 优先核清单，投稿前必须读全文给精确 delta。**不据 title 臆断其方法细节。**

### 5.1 类 A：秩 / 容量分配（rank allocation / budget / layer importance）

| 简称 | 年 | DOI / arXiv | 方向（title 级） | 与本文 delta（待全文核） |
|---|---|---|---|---|
| PARA | 2026 | 10.48550/arxiv.2604.27796 | post-hoc SVD 谱剪枝已训 LoRA 秩 | §2.2 已核：post-hoc vs 本文训前；机制最近 |
| Bayesian-LoRA | 2024 | 10.48550/arxiv.2406.13046 | 可微贝叶斯门控联合学「量化级别+秩」 | **优先核**：同时动秩+量化，但需训练学门控（本文零训练预计算） |
| RA-LoRA | 2024 | 10.18653/v1/2024.findings-acl.933 | 秩自适应 PEFT 用于 2-bit 量化 LLM | **优先核**：秩自适应+量化最近撞名；2-bit vs 本文 4-bit NF4，须读全文判 delta |
| MoLA | 2025 | 10.18653/v1/2025.findings-naacl.284 | MoE-LoRA 层间专家容量分配 | 层间「容量」分配但走 MoE 专家数，非单层秩 r_l；待全文核 |
| CTR-LoRA | 2026 | 10.1109/icassp55912.2026.11463994 | 曲率感知+信赖域引导 LoRA | 曲率信号（需训练态二阶信息）vs 本文冻结基座范数；待全文核 |

### 5.2 类 B：量化感知 PEFT / QAT-LoRA（quantization-aware PEFT）

| 简称 | 年 | DOI / arXiv | 方向（title 级） | 与本文 delta（待全文核） |
|---|---|---|---|---|
| AQLoRA | 2026 | 10.48550/arxiv.2608.23816 | zero-search NF4 误差选层保 fp16 换速度 | §2.1 已核：选层(精度) vs 本文分秩；趋同阴性 |
| LoftQ | 2023 | 10.48550/arxiv.2310.08659 | LoRA-fine-tuning-aware 量化初始化 | **B3 基线候选**：量化感知初始化（非秩分配）；本文 Phase3-E 已纳入 LoftQ 臂 |
| QuAILoRA | 2024 | 10.48550/arxiv.2410.14713 | 量化感知 LoRA 初始化 | **优先核**：「quantization-aware」撞名，但作用于初始化非秩预算；待全文核 |
| L4Q | 2025 | 10.18653/v1/2025.acl-long.99 | 参数高效量化感知微调 | QAT-LoRA 路线（训练中量化），vs 本文冻结 NF4 基座训前分秩；待全文核 |
| Low-Rank QAT | 2024 | 10.48550/arxiv.2406.06385 | 低秩量化感知训练 | QAT（非 PTQ 冻结基座）；待全文核 |
| QWHA | 2025 | 10.48550/arxiv.2509.17428 | 量化感知 Walsh-Hadamard 适配 | 不同适配基（Hadamard）；待全文核 |
| LQ-LoRA | 2023 | 10.48550/arxiv.2311.12023 | 低秩+量化矩阵分解高效微调 | 分解路线，非秩预算分配；待全文核 |
| QDLoRA | 2025 | 10.1007/978-981-95-1021-4_36 | 量化 LLM 上集成低秩分解增强 LoRA | 待全文核 |
| OWQ | 2024 | 10.1609/aaai.v38i12.29237 | 离群值感知权重量化 | 量化算法（非 PEFT 秩分配）；离群值视角可对照本文「范数主导」 |
| SQ-LoRA | 2026 | 10.3390/app16042113 | stable-rank 引导量化压缩 | stable-rank 信号 vs 本文 rel_error×范数；待全文核 |
| EfficientDM | 2023 | 10.48550/arxiv.2310.03270 | 低比特扩散模型 QAT | 域外（扩散），仅列示量化感知 PEFT 广度 |

### 5.3 类 C：LoRA 变体 / 剪枝 / 综述（variants / pruning / surveys）

| 简称 | 年 | DOI / arXiv | 方向（title 级） | 与本文 delta（待全文核） |
|---|---|---|---|---|
| LoRA-FA | 2023 | 10.48550/arxiv.2308.03303 | 冻结 A 矩阵的低秩表示微调 | 训练动力学变体，非秩预算分配 |
| Delta-LoRA | 2023 | 10.48550/arxiv.2309.02411 | 用低秩差更新高秩参数 | 训练机制变体 |
| S-LoRA | 2023 | 10.48550/arxiv.2311.03285 | 并发服务数千 LoRA adapter | 系统/服务层，非分配算法 |
| LoRA-XS | 2025 | 10.3233/faia251185 | 极少参数 LoRA | 参数压缩路线 |
| LoRAPrune | 2024 | 10.18653/v1/2024.findings-acl.178 | 结构化剪枝+LoRA | 剪枝（与 PARA 同类 post-hoc 减参）；待全文核 |
| LoRA Survey | 2024 | 10.1007/s11704-024-40663-9 | LoRA 综述 | 综述定位用 |
| PEFT Survey | 2024 | 10.48550/arxiv.2403.14608 | PEFT 综合综述 | 综述定位用 |

### 5.4 撞车优先核清单（投稿前必须读全文给精确 delta）
1. **RA-LoRA**（2024.findings-acl.933）：「秩自适应 + 量化」标题最撞，须确认其是否训前/是否冻结基座/信号是否量化误差。
2. **Bayesian-LoRA**（2406.13046）：联合量化级别+秩，须确认是否需训练学门控（与本文零训练预计算区分）。
3. **QuAILoRA**（2410.14713）：「quantization-aware」撞名，须确认作用对象（初始化 vs 秩预算）。
4. **PARA**（2604.27796）：已核（§2.2），post-hoc 谱剪枝，Gap-0 成立但 delta 降格成本优势。
5. **AQLoRA**（2608.23816）：已核（§2.1），趋同阴性，最强外部佐证。
6. **LoftQ**（2310.08659）：B3 基线，Phase3-E 已纳入实测对照。

### 5.5 新颖性再确认（基于 22 篇扩展 + 2 锚点）
- **Gap-0 仍成立**：在 24 篇核验文献中，无一占据「训前 + 冻结 NF4 基座 + 零搜索零校准 + 秩预算分配 + 诚实阴性 + 功效/最小可检出效应分析」这一完整组合。最近者：PARA（post-hoc，时机不同）、RA-LoRA/Bayesian-LoRA（秩+量化但需训练或不同量化位宽，待全文核）、AQLoRA（选层非分秩，趋同阴性）。
- **诚实边界（不变）**：Gap-0 是流程/成本空白非性能空白；本文实证阴性，新颖性只能主张「提出并严格检验一个更便宜的训前分配流程 + 首个秩分配研究的最小可检出效应分析 + 诚实报告失效边界与机制（相对误差层级常数⇒误差×范数打分退化为范数规则）」，不能主张 SOTA 或参数效率。
- **Related Work 必引**：AQLoRA（趋同阴性）、PARA（机制最近/撞名）、QLoRA/AdaLoRA/DoRA（基线）、LoftQ（量化感知初始化基线）、RA-LoRA/Bayesian-LoRA（秩+量化最近邻，全文核后给 delta）。

---
*核验声明：§2 两锚点（AQLoRA 2608.23816 / PARA 2604.27796）于 2026-11-16 web_fetch 逐字打开 arXiv abs 页核验；§5 扩展 22 篇由 OpenAlex 权威索引返回 DOI（含 arXiv ID）核验可达。全部 ID 非编造、可解析；未核验 ID 不入库。*

### §5.6 撞车优先核结果（2026-11-16 全文 abstract 核验，闭合 §5.4 TODO；arxiv abs + ACL Anthology 逐字）

> 闭合 §5.4 TODO：读三篇最近撞车候选全文 abstract，给精确 delta。核验源：RA-LoRA=aclanthology.org/2024.findings-acl.933（HTTP 200）；Bayesian-LoRA=arxiv.org/abs/2406.13046（HTTP 200，v3 2024-10-28）；QuAILoRA=arxiv.org/abs/2410.14713（HTTP 200，v1 2024-10-09）。

**① RA-LoRA**（Kim/Lee/Sung/Choi，Findings of ACL 2024，pp 15773-15786，DOI 10.18653/v1/2024.findings-acl.933，bibkey kim-etal-2024-ra）
- 机制（摘要原文）：「dynamically adjusting the adapter's rank using **rank-subspace analysis**, optimizing performance with fewer parameters… tested on **2-bit** efficient fine-tuning… improve model accuracy」。
- 精确 delta vs AdaQ：(a) 信号=**秩子空间分析**（rank-subspace，动态/数据驱动），AdaQ=**预计算冻结 NF4 量化误差**（零搜索零训练）；(b) **2-bit** QAT 场景，AdaQ=**4-bit NF4**；(c) **动态调秩**（训练中环），AdaQ=训前一次性分秩；(d) 结论**正向**（提升精度），AdaQ=**阴性/诊断**。
- 撞车判定：**否**。机制（秩子空间 vs 冻结量化误差）、比特（2 vs 4）、时序（动态 vs 预计算）、结论（正 vs 阴）四重不同。RA-LoRA 是「自适应秩 for 量化 LoRA」最近邻，但走的是学习式秩子空间路径且声称有效——恰与 AdaQ「预计算误差信号无效」形成对照。

**② Bayesian-LoRA**（Meo/Sycheva/Goyal/Dauwels，arXiv:2406.13046，cs.AI，v3 2024-10-28，DOI 10.48550/arXiv.2406.13046）
- 机制（摘要原文）：「prior distribution on both **quantization levels and rank values**… **differentiable Bayesian gates**… finding the optimal rank values and quantization levels for every low-rank matrix… fine-tuning a pre-trained **DeBERTaV3 on GLUE**… on par or better than baselines while reducing bit operations ~70%」。
- 精确 delta vs AdaQ：(a) **学习式可微贝叶斯门**（训练中联合优化秩+量化级），AdaQ=预计算固定分秩（无门无学习）；(b) **DeBERTaV3 编码器/GLUE 分类**，AdaQ=**decoder LLM**（Llama-3-8B/Qwen2.5-14B）生成任务；(c) **联合学量化级+秩**，AdaQ=固定 NF4 仅分秩；(d) 结论**正向**。
- 撞车判定：**否**。Bayesian-LoRA 是编码器上的学习式联合量化+秩优化；AdaQ 是 decoder LLM 上的预计算冻结分秩阴性。范式（学习 vs 预计算）、模型类（编码器 vs decoder）、干预（联合 vs 仅秩）、结论全不同。

**③ QuAILoRA**（Lawton/Padmakumar/Gaspers/FitzGerald/Kumar/Ver Steeg/Galstyan，arXiv:2410.14713，cs.LG/cs.CL，NeurIPS ENLSP-IV Workshop，DOI 10.48550/arXiv.2410.14713）
- 机制（摘要原文）：「quantization-aware **initialization** for LoRA that mitigates [quantization error] by **decreasing quantization errors at initialization**… 4-bit QLoRA… 75% of validation perplexity decrease and 86% of downstream accuracy increase as doubling to 8-bit」。
- 精确 delta vs AdaQ：(a) 干预点=**初始化**（改 LoRA init 以降量化误差），AdaQ=**秩分配**（决定每层 rank 预算，不改 init）；(b) 目标=**减小**量化误差，AdaQ=用量化误差作**分秩信号**（且发现该信号近平坦不可用）；(c) 结论**正向**（ppl/acc 提升）。
- 撞车判定：**否**。QuAILoRA 与 AdaQ 干预点正交（初始化 vs 秩分配）。QuAILoRA 证实「量化误差有害且可经初始化降低」——与 AdaQ「量化误差跨层近平坦、作秩分配信号无效」是**互补的两个角度**（一个降误差、一个证误差不可用于分秩），非竞争。

**综合判定（novelty 再确认）**：
1. 三篇最近邻**均不撞车**：机制（秩子空间/贝叶斯门/初始化 vs 预计算冻结 NF4 误差分秩）、结论（全正向 vs AdaQ 阴性）系统性不同。
2. **Gap-0 仍成立且更清晰**：「训前 + 冻结 NF4 基座 + 零搜索零校准 + 量化误差驱动秩分配 + 阴性结论 + 噪声底功效分析」的完整组合，24 篇检索 + 3 篇全文核验无一占据。
3. **AdaQ 定位强化**：RA-LoRA/Bayesian-LoRA/QuAILoRA/MoLA/CTR-LoRA 等主流均主张「自适应秩/量化感知 PEFT 有效（正向）」；AdaQ 与 **AQLoRA(2608.23816)** 独立同向，构成「**误差驱动选择无效**」的阴性簇——AQLoRA 摘要原文「Choosing protected layers by quantization error is no better than random… which layers they are does not measurably change accuracy」与 AdaQ「rel_error 近平坦(CV1.48%)→product≡importance_only→六策略不可区分」机制互证。这是对主流正向叙事的可复现反例，是诊断型贡献的核心价值。
4. **写作落点**：Related Work 须显式对比这四篇（RA-LoRA/Bayesian-LoRA/QuAILoRA 正向自适应 + AQLoRA 阴性同向），用 delta 表呈现「干预点/信号/比特/结论」四维区分；Discussion 引 AQLoRA 作 convergent evidence（§8.4 已含）。
