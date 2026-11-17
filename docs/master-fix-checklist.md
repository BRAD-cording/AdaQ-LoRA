# AdaQ-LoRA 主修复清单（8 评审合并 · 达小修路线图）

**来源**：合并 8 份专家评审（docs/review/{jcr-gate,rank-budget,nf4-spectral,quant-protocol,repro-eng,stats-protocol,peft-architecture,fair-baseline}.md）共 ~120 条原子修复项，去重后按**依赖类**组织。
**裁定基线**：8 评审全线 Major Revision（fair-baseline 现状 Reject→大修后 Major）；JCR 复核（jcr-14b-verification.md）确认未达小修。
**目标**（User m05617）：全局达收录小修标准后再写稿。
**状态图例**：✅ 已完成（零GPU） · 🔄 Phase-3 队列覆盖（GPU 自动跑，m05457 拍板 D→B→E→C→A） · ⏳ 需额外 GPU（未拍板，待用户决策） · 📝 写作（待数据或可并行）

---

## 路线图摘要（达小修还差什么）

| 阶段 | 内容 | 状态 | 阻断小修？ |
|---|---|---|---|
| 零GPU 止血 | 叙事统一/新颖性落档/α-r披露/档案闭合/阈值修正案/flex审计工具 | ✅ 全清（6 commit） | 否（已解） |
| Phase-3 队列 | D 噪声底→B adalora→E 基线→C R曲线→A 主表重训 | 🔄 跑中（D 评测 ~07:00 UTC 出，全套 ~60h） | 是（B1/B2/B3/B4/B6-retrain 靠它解） |
| 额外 GPU | B5 chat-template / B7-retrain 14B等预算 / F18 RTN+GPTQ+AWQ / B11 TOST噪声对 | ⏳ **待用户拍板**（未含在 m05457） | 部分是（B5/B11 影响 14B 因果与统计） |
| 写作 | peft §8 重写整合 / claim-revision 扩覆盖 / B13 venue+口径 / Related Work 段 | 📝 待数据稳定后落正文 | 是（最终成稿） |

**结论**：零GPU 能做的已做完。达小修的剩余路径 = ①Phase-3 出数（自动，~60h）解 B1/B2/B3/B4/B6-retrain；②用户拍板是否补额外 GPU（B5/B7-retrain/F18/B11）；③数据稳定后按 peft §8 落正文。**当前不可达小修——硬阻断在 Phase-3 GPU 墙钟（~60h）+ 额外 GPU 决策。**

---

## 主题 1：叙事 / 主张对齐（零GPU）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| 标题改 Norm-Aware（撤 Quantization-Aware/Adaptive Rank Allocation 撞名 PARA） | peft §8.7, B9 | ✅ | paper-outline.md:11 已改；peft 推荐 "When Quantisation Error Is Flat..." |
| 撤「ε_l 可相差数倍」动机→结构性后果（rel_error 层级常数 CV1.48%） | nf4, peft §3.2, fair F16 | ✅ b0d4dde | paper-framework.md 5 处 |
| 撤「参数效率 −26.5%」（等秩反号 +11.5%/+6.8%，是预算错配产物） | peft §5.3, fair F14, rank-budget | ✅ b0d4dde | grep −26.5% 清零 |
| 撤 Gap-3（AdaLoRA 剪枝从未触发，零证据）+ Gap-2 改「未检验秩分配路径，本文填补且结论否定」 | fair F16, peft §8.8 | ✅ b0d4dde | sota-protocol:16 |
| 撤命题3b ✅（ρ=0.9836 是合成自测非实测） | nf4, peft §8.3, rank-budget | ✅ b0d4dde | 改「合成自测，非实测」 |
| A10 单卡 → 2×RTX 4090D 双卡（3+ 处文档） | quant, repro, fair | ✅ 002e750 | README/framework/outline |
| Lemma：‖E_l‖_F=ε_l‖W_l‖_F 且 ε 近常数 ⇒ 可证明等价范数规则（核心叙事，把名不副实变可证等价） | peft §3.2/§6.1/§8.3 | 📝 | 整合进 claim-revision + 正文 §3.3 |
| peft §8.1-8.8 英文重写（Abstract/贡献/§3.3/§5/Limitations L1-L10/Conclusion/标题/必删表）整合进 claim-revision | peft §8 | 📝 | 评审已开方，待落正文（写稿阶段） |

## 主题 2：新颖性 / Related Work（零GPU）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| AQLoRA 2608.23816 + PARA 2604.27796 落档（曾全库 0 引用=desk-reject） | jcr B10, fair F19, peft §7 | ✅ 999d60d | arxiv abs 逐字核验 |
| Related Work 扩 ≥18 篇 | fair F19 | ✅ 7ab7be4 | §5 共 24 篇 OpenAlex DOI 核验 |
| 撞车优先核（RA-LoRA/Bayesian-LoRA/QuAILoRA 投稿前读全文给精确 delta） | fair F19, peft §7 | ⏳ 读全文 | §5.4 清单已列；非 GPU，投稿前人工 |
| Related Work 段写进正文（引 AQLoRA 趋同阴性 + PARA 机制最近 + delta） | peft §8.4 | 📝 | 写稿阶段 |

## 主题 3：α/r 混杂（披露完成 + 重训靠 Phase3-A）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| 披露 α/r 混杂（adaq q1.333/v4.0 vs qlora 2.0，max\|w\| 1.98×）+ 因果降级 | fair F15, rank-budget, jcr B6 | ✅ 0217956 | claim-revision §6 |
| train_phase3.py F15 修 α=2r（per_module_type alpha_pattern={mod:2r}） | fair F15 | ✅ 部署 | Phase-3 全臂干净 |
| 现有 18 adaq adapter α=2r 重训（8B 主表） | fair F15, jcr B6 | 🔄 Phase3-A | A 阶段用 train_phase3 重训即解 |
| 14B adaq α=2r 重训 | jcr B6 | ⏳ 额外 GPU | 14B 不在 Phase3-A；待拍板 |

## 主题 4：预算公平性（F4 已修 + adalora Phase3-B + 14B 额外）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| --budget 真约束（assert sum_ranks==budget 否则 exit2，记 budget_ok） | fair F4, rank-budget | ✅ 部署 | train_phase3.py F4 |
| AdaLoRA 动态调度（total_step=auto，tinit/tfinal/deltaT 比例化，训后断言秩降 target_r） | fair F16/F25, repro | ✅ 部署 | train_phase3.py F25 |
| AdaLoRA 等预算重训（剪枝真触发，秩 1024→512） | fair F16, peft §5.1 | 🔄 Phase3-B | B 阶段 adalora×3seed |
| 14B 等预算 qlora 重训（{q6,v5}→528 对齐 adaq） | fair F14(a), jcr B7 | ⏳ 额外 GPU | 待拍板；(c) 零成本改述已落 b0d4dde |
| F21 peak VRAM 采集修（reset_peak 训前 + resumed 标志；seed42 6.67GB 物理不可能） | fair F21, repro | ✅ 部署 | train_phase3.py F21 |

## 主题 5：噪声底 / 统计（Phase3-D + flex审计 + TOST）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| 阈值校准修正案（原 4.5pt 作废，改实测 pooled 噪声底 N=max(N_seed,N_rep)） | jcr B1, stats | ✅ 4414d9e | pre-unblind 冻结（D results=0） |
| Phase3-D 实测噪声底（product 配置×3seed×3rep=9run GSM8K flex 极差 + 方差分解） | stats, fair F8, jcr B1 | 🔄 Phase3-D | 评测跑中 ~07:00 UTC 出 |
| B1 重判 14B flex +8.87pt（套修正案：Δ>N 且 3/3 同号 且 pending B5） | jcr B1 | 🔄 待 D 出数 | 出数即解 |
| flex 抽取器人工审计（adaq+qlora seed42 各 400 docs 双标 κ≥0.95，查差别抽取偏差） | jcr B8, stats | ✅ 工具 81b539f / 🔄 采样排队 | run_b8_flex_audit.sh waiter 1849739，Phase3 后 GPU1 | ✅已执行(2026-11-17: κ双臂FAIL 0.70/0.75，漏抽率 adaq 18.0%/qlora 26.5% 差异偏差实锤 → docs/b8-flex-audit-results.md)
| TOST 等价检验 + bootstrap CI（MMLU「no difference;underpowered」） | jcr B11, stats | ⏳ 分析（待数据） | 零GPU 但需 Phase3 数据；B11 14B 本地噪声对需 GPU |
| n=3 无有效 Wilcoxon（min p=0.25）→ 全结果降描述性 | stats, jcr | ✅ 落档 | claim-revision §5 + stats §6 英文段 |
| eval seed 口径统一 42（主表曾 seed=训练seed 混合方差） | repro F3, stats F3 | 🔄 Phase3 | run_eval_phase3 固定 seed42 |

## 主题 6：基线补全（Phase3-E + 额外 GPU）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| LoftQ 基线（LoftQConfig bits4 iter1 r8 α16 ×3seed） | fair F17-M2, jcr B3 | 🔄 Phase3-E | E 阶段 qlora_loftq |
| 参数匹配 QLoRA（{q9,v9}→3,833,856 +0.86% ×3seed） | fair F17-M3, jcr B3 | 🔄 Phase3-E | E 阶段 qlorapm |
| baseline_bf16 锚点（METHODS=none，3 评测种子，主表零 BF16 锚） | fair F2/F17-M1 | 🔄 waiter 1663672 | 等 phase3 DONE 后 GPU0 |
| 真 INT4-RTN(g=128) 基线（baseline_nf4 实为 bnb NF4+DQ 非 RTN，命名误导） | fair F18, quant | ⏳ 额外 GPU | 待拍板；命名已更正 base_nf4dq_noadapter |
| GPTQ/AWQ 基线（auto-gptq/autoawq，校准 OpenHermes[:512] 披露） | fair F18 | ⏳ 额外 GPU | 待拍板；Scope 声明「不竞争 PTQ 压缩率」 |

## 主题 7：R-曲线 / 预算扫描（Phase3-C）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| R-曲线预算扫描 R∈{128,512,1024}×{adaq,qlora}×seed{42,3407}=12run | jcr B4, fair F17-M4/F20, peft §4.4 | 🔄 Phase3-C | C 阶段；adaq ranks 离线 allocate_ranks 重生成（已生成 configs/rcurve/） |
| 命题3b 最小可行版（64 位点 ±4 秩扰动测 Δℓ，Spearman(score,−Δℓ)） | fair F20 | ⏳ 额外 GPU | M4 廉价替代；待拍板 |

## 主题 8：主表资格 / 重训（Phase3-A）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| 主表仅 21 优化步（packing 塌缩）→ 对方法比较无功效 | stats F1, repro, fair F3 | 🔄 Phase3-A | A 阶段 --no-packing 375 步重训 |
| 主表重训 4 方法×3seed（qlora/dora/adaq α=2r/adalora 动态调度，统一 375 步） | stats F1, fair F3, jcr B2 | 🔄 Phase3-A | 出数前主表降附录或标 underpowered |
| 三表三口径（seq 2048/1024/512）禁跨表比 + 标注 | quant, stats F2 | ✅ b0d4dde | eval-protocol §9 |

## 主题 9：可复现 / 入库（Phase-2 完成）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| results/脚本/ranks 入 git + results_ledger.csv（8766 行）+ provenance + env lock | repro P0 | ✅ 311c814 | 156 files |
| 每 run provenance.json（基座 sha256+git HEAD+pip freeze） | repro F13, fair F13 | ⏳ 部分 | env lock 已入；逐 run provenance 待 Phase3 产物 |
| resume 数值等价验证（主实验 save_only_model=True 无 optimizer/rng） | repro | ⏳ 额外 | 需重训对照；非阻断小修 |
| peft 0.19.1→0.21.2 rank_pattern 语义漂移（已排除，后缀正则两代一致） | repro | ✅ 排除 | get_pattern_key 核验 |

## 主题 10：评测口径（部分完成 + B5 额外）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| strict/flex 政策（strict 对 instruct 不可用，flex 主判据 + 人工审计） | quant, jcr B8, fair F7 | ✅ b0d4dde | eval-protocol §9 | ✅已执行(2026-11-17: κ双臂FAIL 0.70/0.75，漏抽率 adaq 18.0%/qlora 26.5% 差异偏差实锤 → docs/b8-flex-audit-results.md)
| BF16/NF4 双协议标注 + 禁跨表比 + 补锚 | quant | ✅ b0d4dde | 同 adaq adapter BF16 .4766 vs NF4 .4064 差 7 点纯协议效应 |
| B5 chat-template GSM8K 14B 复评（6×70min，格式漂移 artifact 排除） | jcr B5, fair F11 | ⏳ 额外 GPU | 待拍板；出数前 14B 差值禁进 abstract/intro/conclusion |
| fewshot 机读核查（configs[task].num_fewshot）+ sample_len 断言 | fair F10 | ✅ 部署 | run_eval_phase3 继承 |
| 延迟数字 2.33/5.75 s/it 撤回（全仓库无制品） | quant, fair | ✅ b0d4dde | eval-protocol §9 |

## 主题 11：写作 / 结构（待数据稳定）

| 项 | 来源 | 状态 | 说明 |
|---|---|---|---|
| claim-revision 按 peft §8 骨架扩覆盖 16%→全量 | peft §8 | 📝 | 整合 §8.1-8.8 重写处方 |
| B13 venue TMLR-Q2 双轨 + 口径表头/SD/PPL n=62 标注 | jcr B13 | 📝 | 写作 |
| 统计方法段写进 AQLoRA 噪声底方法学（8.2 点会话散布佐证阈值修正案） | fair F19, stats | 📝 | 写作 |
| 正文按 peft §8 落稿（Abstract/贡献/Method/Discussion/Limitations/Conclusion） | peft §8 | 📝 | **写稿阶段（m05617：达小修后）** |

---

## 待用户拍板：额外 GPU 项（未含在 m05457 的 D→B→E→C→A）

| 项 | GPU 成本 | 阻断小修？ | 建议 |
|---|---|---|---|
| B5 chat-template GSM8K 14B 复评 | ~6×70min=7h | 是（14B 因果干净度） | **建议补**：14B flex 信号是仅存正向，须排除格式 artifact |
| B7-retrain 14B 等预算 qlora {q6,v5}→528 | ~1×35min 训 + 评测 | 否（(c) 改述已落） | 可选：因果干净需等预算 |
| 14B adaq α=2r 重训（B6-14B） | ~1×35min 训 + 评测 | 否（披露已落） | 可选：与 B7 合并跑 |
| F18 真 INT4-RTN(g=128) | ~3 评测 run | 部分（RTN 正确性） | **建议补**：fair-baseline 指 baseline_nf4 命名误导，真 RTN 才是 PTQ 文献对照 |
| F18 GPTQ/AWQ | 装库 + 校准 + ~6run | 否（Scope 声明可降格） | 可选：不竞争 PTQ 压缩率则非必需 |
| B11 TOST 14B 本地噪声对 | ~4 评测 run | 部分（统计严谨） | 建议补：MMLU 等价主张需 TOST |
| F20 命题3b 最小可行版 | ~1 run | 否（可删命题3b） | 可选：删命题3b 更省 |

**最小达小修额外 GPU 建议**：B5（7h）+ F18-RTN（~2h）+ B11-TOST（~3h）≈ 12h，接在 Phase-3（~60h）后。GPTQ/AWQ、14B 等预算重训、命题3b 可选/可降格。

---
*本清单由 8 评审 ~120 原子项去重合并；状态截至 2026-11-16 01:50 UTC。Phase-3 进度见 logs/phase3/chain_phase3.log；额外 GPU 项待用户拍板后排期。*
