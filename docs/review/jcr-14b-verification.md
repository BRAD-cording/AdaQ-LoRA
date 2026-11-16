# AdaQ-LoRA · 14B 三种子结果独立复核与小修门槛裁定（JCR 对抗性验证）

> **评审人设定**：JCR Q1 / 中科院 2 区资深审稿人（PEFT/LoRA + 低比特量化 NF4/QLoRA）
> **评审日期**：2026-11-16（服务器 UTC ≈ 00:30，`date -u` 实测 Mon Nov 16 00:28:43 UTC 2026）
> **评审基准**：JCR Q2+ / 中科院 2 区「初稿提交后小修（minor revision）」水位
> **评审方式**：SSH 只读实读 + CPU 独立重算（不复用项目聚合脚本，从原始 `results_*.json` 重算全部数字）；每条结论绑定文件路径；区分「已独立验证 ✓」与「推测/待核查 ◇」
> **本评审足迹**：只新建本文件（`docs/review/jcr-14b-verification.md`），不改任何共享文件，不 commit，留给主代理统一处理
> **评审对象**：`results_v2_qwen14b*/` 三种子评测数据、`docs/preregistration-14b-3seeds.md`（commit 220fd60）、`docs/review/{fair-baseline,stats-protocol,jcr-gate}.md`、`results_v2_phase3_noise/`（Phase3-D 状态）、git 提交史与文件 mtime 时间线

---

## 0. 总裁定（VERDICT）

# **Major Revision —— 未达小修门槛**（与项目自有 `jcr-gate.md` §1、`fair-baseline.md` 总判定方向一致）

三句话版：

1. **数据包本身是干净的**：本评审从 12 份原始 `results_*.json` 独立重算，全部数字与 `summary_3seeds.json`、`REPORT_14B.md` 逐位一致（§1）；预注册时间线闭合——判定规则冻结（脚本 mtime 13:37:14 + commit 220fd60 16:14:08）早于主判定终点揭盲（18:28–22:30），confirmatory 资格成立（§2）。这是项目内少见的合格档案链。
2. **但按预注册字面触发的「正向」判定，站在一个已被项目自己改写的校准参数上**：门槛 4.5pt 系单对 flex 观测（4.55/1.75pt），而 Phase-1 止血（commit b0d4dde）已把 `claim-revision-norm-aware.md` §5 噪声底改为**区间 [0.9, 11.5]**——均值 +8.87pt 落在该区间**之内**；叠加三重混杂（非等预算 528/768、α/r 失配 2.29/4.0 vs 2.0/2.0、无 chat-template 的失效评测口径）一个都未排除，+8.87pt 至多写 *"consistent with a positive effect under a pre-registered descriptive criterion"*，且 `jcr-gate.md` §4 Branch A 要求的阳性机制解释义务完全未履行（§3）。
3. **jcr-gate P0 六件：2 件已闭合（预注册落档、14B 收口），4 件未闭合**（叙事重构半途——README/framework 仍在卖「Quantization-Aware 提升参数效率」、等价统计、预算扫描、新颖性落档）；其中叙事统一、flex 抽取器审计、新颖性落档**不需要 GPU**，是零机时成本却仍敞开的 desk-reject 级缺陷（§6）。

---

## 1. 独立重算结果（已验证 ✓）

方法：CPU python 直接解析 12 份原始 `results_*.json`（GSM8K 键 `gsm8k → exact_match,strict-match` / `exact_match,flexible-extract`；MMLU 键 `mmlu → acc,none`），不经过 `aggregate_b4_3seeds.py`。数据源路径：

- seed42：`results_v2_qwen14b/{adaq,qlora}/__home__ubuntu__AdaQ-LoRA__outputs__adapters_qwen14b__{adaq,qlora}_seed42/results_*.json`（注意：**不存在** `results_v2_qwen14b_seed42/` 目录）
- seed2025：`results_v2_qwen14b_seed2025/{adaq,qlora}/results_*.json`
- seed3407：`results_v2_qwen14b_seed3407/{adaq,qlora}/results_*.json`

### 1.1 MMLU（acc,none；57 子任务全量 ✓，n=14042）

| seed | adaq | qlora | 配对差 |
|---|---|---|---|
| 42 | 0.7842899871813132 | 0.7837202677681242 | +0.0005697 |
| 2025 | 0.7840051274747187 | 0.7836490528414756 | +0.0003561 |
| 3407 | 0.7845748468879077 | 0.7841475573280160 | +0.0004273 |
| **mean** | — | — | **+0.00045103（+0.045pt），三种子同号** |

### 1.2 GSM8K strict（exact_match,strict-match；n=1319 全量 ✓）

| seed | adaq | qlora | 配对差 |
|---|---|---|---|
| 42 | 0.18271417740712662 | 0.13040181956027294 | +5.23pt |
| 2025 | 0.19029567854435178 | 0.12585291887793784 | +6.44pt |
| 3407 | 0.2699014404852161 | 0.22441243366186506 | +4.55pt |
| **mean** | — | — | **+5.41pt（样本 std 0.96pt）** |

### 1.3 GSM8K flex（exact_match,flexible-extract；n=1319 全量 ✓）——主判定终点

| seed | adaq | qlora | 配对差 |
|---|---|---|---|
| 42 | 0.6588324488248674 | 0.5253980288097043 | **+13.34pt** |
| 2025 | 0.5117513267626990 | 0.42153146322971946 | **+9.02pt** |
| 3407 | 0.5602729340409401 | 0.5178165276724791 | **+4.25pt** |
| **mean** | 0.5769522365428355 | 0.4882486732373010 | **+8.87pt（0.08870356330553453），样本 std 4.55pt，all_positive=True** |

### 1.4 与聚合产物核对

- `results_v2_qwen14b/summary_3seeds.json`：全部 per_seed / mean / std / paired_diff 字段与上述重算**逐位一致**（含 float 尾数）✓
- `results_v2_qwen14b/REPORT_14B.md`：表格数字与重算一致（舍入位内），内嵌同一判定规则文本 ✓
- `results_v2_qwen14b/summary.json`（seed42 单种子，已 commit c313131）：同值 ✓
- **结论：聚合链无数值错误。**（档案性瑕疵：seed2025/3407 的原始 JSON、`summary_3seeds.json`、`REPORT_14B.md` 目前均 untracked（`git status` 显示 `??`）——Phase-2 入库承诺「results 全量入 git」未覆盖这批最新产物，见 B12。）

### 1.5 参数量与秩预算独立验证

- adaq 14B `rank_pattern={q_proj:7, v_proj:4}`（`ranks_qwen14b.json` 键 `per_module_type_ranks (方案A, PEFT 直接使用)`）：(7×10240 + 4×6144)×48 = **4,620,288** ✓（q_proj 5120×5120 → 每秩 10240；v_proj 5120×1024 GQA → 每秩 6144）
- qlora r=8 全模块：(8×10240 + 8×6144)×48 = **6,291,456** ✓
- adapter safetensors 顶层文件字节数：adaq 三种子均 18,506,656 B ≈ 4,620,288×4（F32 存储）+25.5KB header；qlora 均 25,191,384 B ✓；比值 0.734 → **adaq 少 26.5% 参数** ✓
- sum-of-ranks：adaq (7+4)×48 = **528**（超 `--budget 512` 死参数 +16）；qlora (8+8)×48 = **768**（+256）；**qlora 获 1.45× 秩预算** ✓（F14 独立证实；`per_layer_ranks` 96 条 sum=512 分布 {4:47,5:4,6:11,7:34} 为方案 B 产物，实际训练用方案 A）

---

## 2. 预注册时间线核验（已验证 ✓）

`docs/preregistration-14b-3seeds.md` 全文（`git show 220fd60`，43 行，唯一 commit，此后未改）+ 文件系统 mtime + git log + chain 日志交叉验证，全部时间为 UTC：

| 时刻 | 事件 | 证据 |
|---|---|---|
| 11-14 19:39 / 20:27 | seed42 MMLU adaq/qlora 落档 | results JSON mtime |
| 11-14 21:25 / 23:28 | seed42 GSM8K adaq/qlora 落档（**冻结前全部已知**） | results JSON mtime |
| 11-14 23:37:54 | commit c313131（14B 单种子 + 噪声分析） | git log |
| **11-15 13:37:14.586534594** | `scripts/aggregate_b4_3seeds.py` 最后修改——判定规则四条硬编码于 98–101 行（`sed -n '80,120p'` 实读确认） | 文件 mtime，与 prereg §1 声称**逐位一致** |
| 11-15 13:37:16.400528213 | `scripts/run_b4_qwen14b_seeds.sh` 最后修改 | 文件 mtime |
| 11-15 13:37:22 | 训练链启动（SEEDS=2025 3407） | `logs/chain_b4_seeds.log` START 行 |
| 11-15 14:44:20 | 6/6 adapter 训练完成，Peak VRAM 17.60–17.62GB | prereg §6；b4_*.log mtime（14:10/14:43/14:44）吻合 |
| 11-15 15:32:33 / 15:33 | seed2025 MMLU 落档（次要终点） | results JSON mtime |
| **11-15 16:14:08** | **commit 220fd60：prereg 43 行落档冻结** | git log |
| 11-15 18:28:11 / 18:42:07 | **seed2025 GSM8K adaq/qlora 落档（主判定终点首次揭盲，晚于冻结 +2h14m）** | results JSON mtime |
| 11-15 21:59:22 / 22:30:33 | seed3407 GSM8K adaq/qlora 落档 | results JSON mtime |

**结论：主判定终点（gsm8k flex，seeds 2025/3407）全部在规则冻结（脚本 mtime + commit 双保险）之后揭盲，confirmatory 资格成立。** jcr-gate.md L100 的程序性警告（「规则不落档则 14B 失去 confirmatory 资格」）已被 220fd60 在时限内解除——落档（16:14）早于 seed2025 GSM8K 揭盲（18:28）✓。

seed42 的处理：其全部结果在冻结前已知，prereg §4 如实披露并定位为「探索性前例，不构成确证证据，但三种子规则对其同样生效」——披露诚实，处理得当 ✓。seed2025 MMLU 在冻结后、prereg commit 前落档（15:32 < 16:14），属次要终点，prereg §4 亦披露 ✓。

---

## 3. 裁定 (a)：flex「正向」是真实信号，还是噪声底未定下的过早结论？

### 3.1 字面触发：事实，且程序合法

三种子配对差同号（+13.34/+9.02/+4.25）且均值 +8.87pt > 4.5pt → prereg §2 规则① H1 分支字面触发；`jcr-gate.md` §4 Branch A（阳性预案）激活。规则冻结早于揭盲（§2 已验证），执行无作弊痕迹（聚合器只输出数据与规则文本，本评审重算一致）。

### 3.2 但门槛的校准参数已失效——项目自己的文档就是证据

1. **4.5pt 的出身**：8B 同配置对（product vs importance_only，schemeA，rank_pattern 均 {q_proj:12, v_proj:4}，safetensors 128/128 张量字节级不同）仅**两对** flex 观测：s3407 |.5246−.4792| = 4.55pt、s2025 |.5011−.5186| = 1.75pt，取大者。prereg §3 自declared「仅一对 run 估计，保守但高方差」。
2. **同配置对的 strict 口径给出 11.52pt**（fair-baseline F8：product strict 0.42532221379833207 vs importance_only 0.3100833965125095，152/1319 题）；另一对 {q8,v8}（error_only/uniform）strict Δ 仅 1.06pt——**两对相差 10.9×，单点/成对噪声估计不可用，噪声重尾**（F8 结论，本评审采纳）。
3. **口径注记（对任务书转述的两处更正，见附录 B）**：11.52pt 是 **strict** 口径，4.5pt 门槛是 **flex** 口径——「4.5pt 被低估 2.6×」是跨口径比较；flex 口径的直接证据是 4.55 / 1.75 / 0.08pt（第三对 error_only/uniform s3407 flex）三个观测。同样，「seed42 +13.34pt 落在 11.52pt 之内」不成立：13.34 > 11.52，且跨口径。**但这不救 4.5pt**：σ̂_d(flex, df=2) = √((4.55²+1.75²)/2) ≈ 3.45pt，95% CI（χ²₂）≈ **[1.8, 21.7]pt**（stats-protocol §5.2 实读复核）——真实 flex 噪声底可在 1.8 到 21.7pt 之间任何位置，4.5pt 无校准依据；门槛 4.5pt = 1.31σ̂ → 正态近似单侧 p≈0.10，本就不是 0.05 级门槛。
4. **决定性事实——Phase-1 已改写项目自己的噪声口径**：commit b0d4dde（11-15 16:57:55）提交说明明载「claim-revision §5 噪声底改区间 **[0.9, 11.5]**（11.52 同种子对实测）」。即项目当前官方立场：噪声底是区间，上界 11.5pt。**均值 +8.87pt 落在 [0.9, 11.5] 之内**——按冻结门槛判「正向」，按项目自己修正后的口径判「不能排除噪声」。同一份数据、两个都「合法」的口径、相反的结论：**这正是「判定规则校准参数失效」的教科书案例**。prereg §5 允许的最强措辞「三种子一致**超噪声底**正向差异」中的「超噪声底」三字，在 [0.9,11.5] 口径下已不成立。

### 3.3 三种子配对差的统计结构

- **单种子扛均值**：seed3407 +4.25pt **本身低于 4.5pt 门槛**；规则触发完全依赖 seed42 +13.34pt 把均值拉过线。均值 +8.87pt 亦低于 stats-protocol §5.2 自设的「任何 <9pt（2×上界）的单 run flex 差不可信」线的量级。
- **事后 t 检验（仅记录，不作判据）**：t = 8.8704/(4.5508/√3) = **3.38**，df=2，双侧 p≈**0.077**，单侧 p≈0.039——未过双侧 0.05；n=3 时引入 t 检验本身就是预注册要规避的事后行为（HARKing），且 stats §5.1 已钉死：三种子同号 = 符号检验 FWER P=2×(1/2)³=**0.25**，「同号」几乎无证据力。
- **8B 先例反向**：8B flex 上 product（=adaq 同源机制）从未赢过 uniform/random/error_only（8B REPORT：product .5129 vs uniform .5217 vs error_only .5254；六策略极差 2.7pt < 噪声）；seed42 单种子阳性（.4829>.4435）扩种子后**蒸发**。jcr-gate L98 据先例预测「阳性判定大概率不成立」——实际却触发了。**意外反转 + 替代解释未排除 = 应当更保守，而不是更兴奋。**

### 3.4 未排除的替代解释（按可能性排序；均 ◇ 推测，均可检验）

1. **格式漂移 artifact（F11，最强替代解释）**：基座是 Qwen2.5-14B-**Instruct**，评测用 4-shot raw-completion **无 chat template**；strict .1827/.1304 对官方 ≈0.94（F11 引文，本评审未独立核查外部数值 ◇）→ 评测口径已失效。F11 原话：「flexible 差值可能主要由答案格式漂移驱动」。机制猜想 ◇：adaq 与 qlora 的 α/r 缩放不同（下条），可系统改变输出风格，使 flex（正则抓数字）与 strict（格式合规）**同时**上升而无能力增益——与观测模式（strict +5.41pt、flex +8.87pt 同向）完全相容。检验 = chat-template 评测（6 次评测，adapter 已有，纯 GPU 评测 ~70min/次）。
2. **α/r 混杂（F15，已实测确认存在，效应大小 ◇）**：14B 6 份 adaq `adapter_config.json` 均 `alpha_pattern={}` + `lora_alpha=16` → α/r = 16/7≈**2.29**（q_proj）/ 16/4=**4.0**（v_proj）；qlora 统一 **2.0/2.0**。LoRA 前向缩放 (α/r)·BA → 一组**未声明的逐模块有效学习率差异与秩分配处理完全混杂**。旁证（F15 实测）：max|w| adaq 0.2471 vs qlora 0.1250 ≈ 1.98×，与 v_proj 2× 放大吻合。14B 上「正向」无法归因于秩分配而非缩放。
3. **预算不对称（F14，方向上反而利于结论）**：qlora 拿 1.45× 秩预算（768 vs 528）、1.36× 参数（6.29M vs 4.62M）**仍输**（若信号为真）——预算混杂不能解释 adaq 赢，这一点使「若是真信号则更强」；但同样提示：更少容量反而赢，更符合优化动力学差异（α/r）而非「分配智慧」。
4. **14B 噪声底未知**：Phase3-D 是 **8B**（`scripts/run_phase3_train.sh:11` MODEL=Meta-Llama-3-8B，已验证），9 runs = product 配置 × 3 seeds × 3 reps——即使出数也只回答 8B 噪声；14B 本地噪声对（stats P1-9：qlora×2 同 seed 复训）未排期 ◇。8B→14B 外推方向未知（stats §5.2.3：seq512/255步/NF4/instruct 全部不同）。

### 3.5 裁定结论 (a)

**「预注册字面触发的描述性判定，程序合法、执行诚实；但其校准参数（4.5pt）已被项目自身证据（F8 重尾 + [0.9,11.5] 区间改写）实质废黜。当前不能把 +8.87pt 认定为真实信号，也不能宣布预注册无效——正确处置是保留触发记录 + 在新数据揭盲前冻结一个阈值校准修正案。」**

不算 HARKing 的修订路径（四步，缺一不可）：

1. **保留字面触发记录**：不删不改 prereg §2 原判与 `summary_3seeds.json` 的判定输出（git 已有 confirmatory 审计轨迹）。
2. **在 Phase3-D 首个 flex 评测落档之前**，冻结「阈值校准修正案」（新文档 + 时间戳 + commit）：预写重判规则——「以 D 的 9-run pooled flex 噪声（极差与 2σ̂ 取大者）为新门槛重判 14B 三种子配对差：若均值 +8.87pt > 新门槛，措辞升级为 *positive effect replicated above the measured noise floor*；若 ≤ 新门槛，降级为『原预注册规则曾触发，但其门槛基于被低估的单点噪声估计（4.5pt，单对观测；项目后修正区间 [0.9,11.5]）；按实测噪声底重判后，未观察到一致差异』」。**14B 数据已固定不可变，修正案只改校准参数、且在新数据揭盲前冻结 → 这是阈值校准（threshold recalibration），不是挑选结果（HARKing）。** 时间窗口：D 训练 6/9 已完成、s2025 r0/r1 进行中（chain log 00:08:44 启动，~31min/run），评测未跑——修正案必须抢在评测揭盲前，**窗口以小时计，本条最紧急**。
3. **补 14B 本地噪声对**（qlora×2 同配置同 seed 复训 + flex 评测，stats P1-9）：D 是 8B，外推问题只有本地对能解。
4. **chat-template 评测（F11）出数前，14B GSM8K 差值不得进 abstract/intro/conclusion/图表标题**（F11 原要求）；Branch A 的阳性机制解释义务（jcr-gate §4：「为什么 14B 出现？‖W‖ 分布随规模变化？需数据」）当前**零履行**——若最终要写 Branch A，机制节是硬义务，不是可选项。

---

## 4. 裁定 (b)：MMLU +0.05pt——14B 的价值主张到底是什么？

- +0.045pt = 14042 题中约 6 题的量级；三种子同号（+0.06/+0.04/+0.04）但幅度无实际意义。跨种子一致性提示可能存在微小系统差 ◇（α/r 或容量不对称所致？未检验，不建议写入论文）。
- MMLU 对 2000 条 SFT 本就不敏感（`eval-protocol-v2.md` §6 项目自认）→ 14B 表中 MMLU 只能当「未劣化」护栏，不能当有效性证据。
- **14B 上唯一可辩护的价值主张**：①**参数效率描述**——「以 73.4% 的 adapter 参数（4.62M vs 6.29M）取得 MMLU 持平」，参数是真实资源口径，此主张不需要等预算前提，但必须随附「非预算对齐比较」声明（§5）；②**可检测效应边界证据**——服务论文新叙事（Detectable-Effect Boundaries）：即使在出现同号差的地方，差值也处在噪声分辨率边缘，这正是「机制近常数 → 分配策略间差异低于可检测限」主线的一部分。
- **措辞红线**：n=3 无 CI、TOST 未做 → 只能写「未观察到差异；underpowered to establish equivalence」（stats §3.2.8/P0-3），**不能**写「统计上不可区分」「等价」「打平」——「持平」一词在中文语境已属边缘，英文稿禁用 *equivalent / indistinguishable / as good as*（stats §6.5 对照表）。

---

## 5. 裁定 (c)：非等预算下「用更少资源打平/略胜」是否成立？

事实（已验证，§1.5）：sum-of-ranks 528 vs 768（1.45×）、params 4.62M vs 6.29M（0.734×）；`run_b4_qwen14b.sh` 的「同预算 512」标签是死参数（Phase-1 已更正注释，b0d4dde）。

**分层裁定：**

1. **「以少 26.5% 参数取得 MMLU 持平」——描述性成立。** 参数量是合法的资源口径；方向对 adaq 有利；claim-revision §4 已备好合规措辞（「在 QLoRA 获得 1.45× 秩预算（768 vs 528）的条件下……并明确声明这不是预算对齐比较」）。`REPORT_14B.md` 已按规则④同报参数量 ✓。
2. **「略胜」（GSM8K）——当前不成立。** 不是因为预算方向（预算不对称反而无法解释 adaq 赢），而是因为 §3.4 的格式 artifact + α/r 混杂使差值**无法归因于任何有意设计**。「赢」的可能是有效学习率差异和输出风格漂移，不是秩分配。
3. **作为因果比较（分配策略 → 性能）——无效。** 三重混杂（预算 528/768、α/r 2.29/4.0 vs 2.0/2.0、无 chat-template 口径）一个都未控制。14B 表目前只能定位为「自然主义观察」，不是受控实验。
4. **修复路径**（F14 三选一已给全）：立即执行 (c) 零成本改述；等预算重训排 (a) qlora{q6,v5}→528（params 4,423,680）或 (b) adaq{q11,v5}→768，各 3 种子；**并且必须与 F15（α=2r 重训）合并做**——只修预算不修 α/r，混杂仍在。
5. **波及范围警告**：F15 的 α/r 混杂同样污染 **8B 全部 adaq 臂**（12 份 adapter 同病）→ 8B 主表与消融的「product 不赢」阴性结论**方向上稳健**（阴性不怕混杂抬高 adaq），但任何「adaq 与 X 无差异」的等价表述同样受累。这是全包性问题。

---

## 6. 全局小修门槛裁定

### 6.1 jcr-gate P0 六件现状（本评审逐项核验）

| gate P0 | 内容 | 状态（核验证据） |
|---|---|---|
| P0-1 预注册落档 | 规则+时间戳+脚本版本 | **DONE ✓**（220fd60；时间线 §2 闭合） |
| P0-2 叙事重构 | 全库 grep "Quantization-Aware/量化感知.*提升" 0 命中 | **半 DONE ◐**：outline 标题已改 Norm-Aware（paper-outline.md:11 评审裁定改题注记）、framework 已撤「ε_l 可相差数倍」/Gap-3/命题3b（b0d4dde）；**但 README.md:1/5/179 仍写「量化感知……零额外显存开销下提升参数效率」+ BibTeX 旧题，paper-framework.md:44 仍「3.3 量化感知秩分配」，claim-revision 状态行仍「提案，未改正文」（:3/:94）**→ 验收信号未达标 |
| P0-3 14B 收口 | 三种子 + 按预注册判定 | **DONE ✓**（数据齐、判定触发、本评审重算一致）——但 Branch A 附带义务（机制解释/Limitations 落地/chat 口径）未履行 |
| P0-4 等价统计 | ≥3 种子、TOST、噪声底 ≥2 对带 CI | **NOT DONE ✗**（Phase3-D 训练 6/9 rc=0、s2025 r0/r1 进行中、评测未跑、`results_v2_phase3_noise/summary.json` 全 null——本评审实读；TOST 未做；且 D 为 8B，14B 本地噪声对无排期） |
| P0-5 预算扫描 | R∈{128..1024}×{adaq,uniform,qlora}×≥2 种子 | **NOT DONE ✗**（Phase3-C 计划中，未出数） |
| P0-6 新颖性落档 | AQLoRA/PARA/Gap-0 检索档案 + Related Work delta 段 | **NOT DONE ✗**（gate 记录全库 grep 0 命中；此后无新检索档案入库 ◇ 以 git log 为据） |

### 6.2 裁定

**未达小修门槛。当前证据包 = Major Revision。** 距小修的距离已不是「缺数据」——最贵的机时件（D 噪声底、A 主表重训、C R-曲线、B/E 基线）都已在跑；**缺的是 4 件 P0 闭合 + 14B Branch A 义务履行 + 替代解释排除**。其中叙事统一（README/framework）、flex 抽取器审计、新颖性落档、阈值校准修正案**均为零 GPU 成本**，应在 Phase3 出数前立即完成——特别是修正案有时间窗口（§3.5.2）。

### 6.3 编号阻断项清单

**①类：Phase3 出数即可解（机时已在跑）**

| # | 严重度 | 内容 | 验收信号 |
|---|---|---|---|
| B1 | **P0** | 实测噪声底 + 修正案重判：D（8B product 9 runs）评测出数 → pooled flex 噪声（极差/2σ̂）→ 按 §3.5 预冻结修正案重判 14B；同步完成噪声底区间化报告（stats P0-5） | `results_v2_phase3_noise/summary.json` 非 null；修正案 commit 时间戳早于 D 首个评测落档；重判结论写入 REPORT_14B 附录 |
| B2 | **P0** | 8B 主表资格：Phase3-A 重训（足步数）出数后主表才可进主文；在此之前主表只能按 stats §6.3 降级模板进附录；eval-seed 统一（stats P0-2）并入重训链 | 重训后 `results_v2/summary.json` 步数/token 对账通过；主文表引用新数据 |
| B3 | **P1** | 基线补齐：LoftQ、param-matched QLoRA（Phase3-B/E；baseline_bf16 锚点行已入 e5143fc ✓ 部分完成） | 主表含 loftq / qlora_parammatched 行，n_seeds=3 |
| B4 | **P0** | R-曲线（Phase3-C = gate P0-5 预算扫描）：唯一自标「必须」且可能产出正面结论的实验维度；图 2 来源 | `results_v2_phase3_R*/summary.json` 含 R×策略网格 |

**②类：需补实验或改写（Phase3 解不了）**

| # | 严重度 | 内容 | 验收信号 |
|---|---|---|---|
| B5 | **P0** | chat-template GSM8K 14B 评测（F11）——阳性判定的最强替代解释排除；纯评测（adapter 已有），6×~70min GPU | `results_v2_qwen14b_chat/summary.json` 含 strict+flex 双列；出数前 GSM8K 差值不进 abstract/intro/conclusion/图表标题 |
| B6 | **P0** | α/r 混杂处置（F15）：α=2r 重训全部 adaq adapter（8B 12 + 14B 6），或全量披露 + 所有因果措辞降级为「分配与缩放的联合效应」；与 B7 等预算重训合并做性价比最高 | 重训后 adapter_config `alpha_pattern` 逐模块 = 2r；或 Limitations 有专段 + 措辞审查记录 |
| B7 | **P0** | 14B 等预算比较（F14）：(c) 零成本改述**立即**执行（措辞已备）；(a)/(b) 等预算重训排 P1 机时 | 全库「同预算 512」类表述 0 命中（run_b4 注释已改 ✓）；14B 表脚注含预算不对称声明；重训后 sum-of-ranks 对齐 |
| B8 | **P0** | flex 抽取器人工审计（stats P0-4）：主判定指标的仪器未校准——分层 400 条（flex对∧strict错 100 + flex错 100，×2 模型）双人标注，precision/recall/Cohen's κ；precision≥0.95 才可维持 flex 主判据 | `docs/review/flex-audit.md` 存在且含 κ；零 GPU，可立即并行 |
| B9 | **P0** | 叙事统一收尾（gate P0-2 余项）：README.md:1/5/179 与 paper-framework.md:44 删改「Quantization-Aware/提升参数效率」；claim-revision 状态行从「提案」改为「已执行」；BibTeX 题同步 | gate P0-2 验收：全库 grep 0 命中（历史文档标 superseded 除外）；纯写作半天 |
| B10 | **P0** | 新颖性落档（gate P0-6）：AQLoRA/PARA/Gap-0 检索记录（日期/库/检索词）+ Related Work delta 段 | docs/ 内可复核检索档案；纯检索+写作 2-3 天 |
| B11 | **P1** | 等价统计升级（gate P0-4 分析侧）：TOST + margin（取 B1 实测噪声底）、item-level 配对 bootstrap CI（stats P1-8）、14B 本地噪声对（stats P1-9，可并入 B6 重训窗口） | 分析报告含 TOST p 与等价 CI；n=3 功效不足处如实写 underpowered |
| B12 | **P1** | 档案闭合：seed2025/3407 原始 JSON + `summary_3seeds.json` + `REPORT_14B.md` 当前 untracked → 入 git（本评审不 commit，留给主代理）；溯源链机读化（F13/Phase2 的 provenance.json + sha256 审计仍缺，`main-run-provenance.md` 已诚实落档缺失事实 ✓） | `git status` 干净；新 run 起 provenance.json 随产 |
| B13 | **P2** | 措辞与图表润色：口径四元组表头（stats P1-12）、sample-SD(ddof=1) 标注、PPL n=62 删除或扩全量、跨表比较禁令执行、venue 重定位（gate P2-5：ICLR/NeurIPS 不现实 → TMLR/Neural Networks 类 Q2 双轨） | 逐表自查通过 |

### 6.4 通往小修的最小路径（依赖序）

1. **立即（零 GPU，今天）**：§3.5 修正案冻结（**抢在 D 评测揭盲前，窗口以小时计**）→ B9 叙事统一 → B10 新颖性落档 → B8 flex 审计启动 → B7(c) 改述 + B6 披露性措辞 → B12 untracked 入库。
2. **Phase3 出数（~1-2 天，已在跑）**：B1 重判 → B2 主表资格 → B4 R-曲线 → B3 基线。
3. **GPU 排队**：B5 chat 评测（最便宜的替代解释排除，优先）→ 视 B5/B1 结果决定 B6/B7 重训范围 → B11 TOST。
4. 以上闭合后，剩余仅措辞/图表/溯源润色（B13）= **小修区间**。

---

## 7. 三句话总评

1. 这份 14B 三种子包**数据干净、时间线干净、执行诚实**：独立重算逐位一致，预注册冻结早于主终点揭盲，seed42 探索性地位自我披露——档案链质量高于项目内 8B 主表（后者仍有 21 步 + eval-seed 混口径问题在 Phase3-A 待解）。
2. 但它触发的「正向」站在一个已被项目自己改写的噪声口径上：+8.87pt 介于冻结门槛 4.5pt 与自修区间上界 11.5pt 之间，单种子（+13.34pt）扛起全部均值，seed3407（+4.25pt）自身不过线，而格式 artifact（F11）与 α/r 混杂（F15）两个替代解释一个都没排除——**在 chat-template 与实测噪声底出数前把它写成真实信号，就是在 14B 尺度上重演 8B seed42 单种子阳性蒸发的老路**。
3. **Verdict：Major Revision**——小修入场费 = 4 件未闭合 P0（叙事统一/等价统计/预算扫描/新颖性落档）+ B5/B6/B8 替代解释排除 + §3.5 阈值校准修正案（唯一有时间窗口的一件，须抢在 Phase3-D 评测揭盲前冻结）；好消息是最贵的机时件都已在跑，坏消息是最紧急的一件（修正案）是零成本的，却只剩几小时窗口。

---

## 附录 A：独立验证清单（claim → 证据 → 结果）

| # | 声称 | 验证方式 | 结果 |
|---|---|---|---|
| A1 | 12 份原始 JSON 数字与转述/聚合一致 | CPU python 重算（不经过聚合脚本） | ✓ 逐位一致 |
| A2 | gsm8k n=1319 全量、mmlu 57 子任务全量 | JSON `n-samples`/`configs` 字段 | ✓ |
| A3 | 判定规则硬编码于 aggregate_b4_3seeds.py 98-101 行 | `sed -n '80,120p'` 实读 + mtime 13:37:14.586534594 | ✓ 与 prereg §1 声称逐位一致 |
| A4 | prereg 冻结早于主终点揭盲 | git log 220fd60（16:14:08）vs results mtime（18:28–22:30） | ✓ |
| A5 | seed42 揭盲早于冻结且已披露 | mtime 11-14 19:39–23:28 + prereg §4 | ✓ 披露诚实 |
| A6 | 参数量 4,620,288 / 6,291,456 | 解析式独立计算 + safetensors 字节数（×4 F32） | ✓ |
| A7 | sum-of-ranks 528 vs 768 | ranks_qwen14b.json 方案A 键 + adapter_config | ✓（F14 证实） |
| A8 | F8：11.52pt 为 strict 口径、同配置对 128/128 张量字节级不同 | fair-baseline.md L700-705 实读 + 8B REPORT flex 表交叉 | ✓（并发现任务书转述两处口径混淆，附录 B） |
| A9 | Phase3-D 状态 | summary.json（全 null）+ chain_phase3.log（6/9 rc=0，s2025 r0/r1 在跑）+ run_phase3_train.sh:11（MODEL=8B） | ✓ 评测未跑、D=8B |
| A10 | Phase-1 已改噪声底为区间 [0.9,11.5] | git show b0d4dde 提交说明 | ✓ |
| A11 | README/framework 仍卖旧叙事 | grep 实测（README.md:1/5/179、paper-framework.md:44） | ✓ gate P0-2 未闭合 |
| A12 | jcr-gate 总判定 = Major Revision、Branch A 义务 | jcr-gate.md §1/§4/§5 实读 | ✓ |
| A13 | flex σ̂_d=3.45pt、CI [1.8,21.7]、符号检验 FWER 0.25 | stats-protocol.md §5.1/§5.2 实读 + 本评审复算 t=3.38/p≈0.077 | ✓ |

## 附录 B：对父代理任务书转述的更正（事实驱动红线）

1. **「seed42 的 +13.34pt 落在 8B 同配置噪声 11.52pt 之内」——不成立**：13.34 > 11.52；且 13.34 为 flex 口径、11.52 为 strict 口径，跨口径直接比较本身不合法。正确表述：seed42 差值**超出**现有任何单点噪声观测，但 flex 噪声 σ 的 95% CI 上界达 21.7pt，故仍不能排除噪声。
2. **「F8 证 4.5pt 噪声底被低估约 2.6×（真实可达 11.52pt）」——口径需拆分**：11.52/4.5≈2.6 是 strict 观测对 flex 门槛的跨口径比值；flex 口径的直接证据是 4.55/1.75/0.08pt 三个观测（重尾成立、单点估计不可用的结论不变）。F8 自身亦声明：成因查清前 11.5 只能作「噪声底上界证据」，非估计值。

## 附录 C：未验证 / 推测项（◇ 汇总）

- Phase3-A/B/C/E 的具体配置与进度：本评审只实证了 D（8B product 9 runs）与 bf16 锚点行（e5143fc）；A/B/C/E 状态转述自父代理任务书，未逐一读 chain 日志 ◇。
- Qwen2.5-14B-Instruct 官方 GSM8K ≈0.94：引自 F11，未独立核查外部文献 ◇。
- 「MMLU 三种子一致微弱正号可能源于 α/r 系统差」：推测，未检验 ◇。
- 「格式漂移是 flex 正向的最强替代解释」：基于 F11 口径失效证据的机制推断，chat-template 评测出数前不能定论 ◇。
- claim-revision §5 区间 [0.9,11.5] 中 0.9 的来源观测：未逐行核对 claim-revision 现文本（仅核对 commit 说明）◇。

*本评审未运行任何 GPU 任务；未修改任何共享文件；未 commit。本文件为本次评审唯一新建产物。*