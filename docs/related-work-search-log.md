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
- [ ] 扩到 ≥18 篇完整 Related Work 检索档案（F19）：覆盖 rank allocation / quantization-aware PEFT / adaptive LoRA 三类，逐篇核验 ID 入库（不编造）。
- [ ] 把 §2 delta 写进论文 Related Work 段（引 AQLoRA 作趋同阴性 + PARA 作机制最近）。
- [ ] AQLoRA 噪声底方法学（8.2 点会话散布）写进论文统计方法段，支撑阈值修正案。

---
*本日志所有 arxiv ID 于 2026-11-16 web_fetch 逐字打开 abs 页核验可达。未核验 ID 不入库。*
