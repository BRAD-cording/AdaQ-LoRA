# AdaQ-LoRA 投稿前评审 — NF4 量化误差与谱分析

| 字段 | 内容 |
| --- | --- |
| 评审角色 | NF4 量化误差与谱分析专家（E_l=Q₄(W_l)−W_l 逐层建模、ε_l 分布、零均值/块独立性假设实测边界、反量化路径一致性、跨模型异质性） |
| 评审日期 | 2026-11-16 |
| 方法 | 远程只读审计（未触碰 GPU/进程）+ 代码全读 + 从原始 JSON 独立重算 + 原始评测产物核对 + 图目视 |
| Verdict | **major-revision** |

## 0. 证据清单（下列数字均为本次实测；路径相对 /home/ubuntu/AdaQ-LoRA）

| 证据 | 路径 | 实测值 |
| --- | --- | --- |
| 8B 秩与层统计 | ranks.json | 64 层 sum=512；方案A={q_proj:12,v_proj:4}；rel_error mean .093608、样本 CV 1.4847%（总体 1.47%）、min .092642 max .100753；importance mean 44.52、CV 56.65%；Spearman(rel,score)=.1440、Spearman(imp,score)=.9979；Var(log·) 分解 importance 占 99.92%；max\|score−rel·imp\|=0 |
| 14B 秩 | ranks_qwen14b.json | 96 层 sum=512；方案A={q7,v4}；rel mean .094280、CV 4.00%、max .122659（离群层）；Spearman(rel,score)=−.1967 |
| ε 诊断 | results_v2/epsilon_l_report.json | eps med .093197、spread 8.71%；零均值 bias mean 9.81e-6、max 1.78e-5，对照 mean_abs_e 9.53e-4（≈1.0–1.5%）；between/(between+within) med 1.56%；块均值 lag-1 autocorr \|·\|≤.0074；α med .436（q med .663 / v med .252）、r² med .802 |
| 图 | results_v2/epsilon_distribution.png | 三子图目视：panel1 layer-0 离群(.1007/.0999)；panel2 q/v 两条 α 带+单 median 线；panel3 log 尺度 \|mean\| 无尺度参照 |
| NF4 主表 | results_v2_ptq_nf4/summary.json | adaq_nf4 GSM8K .406368 五方法最差且 < baseline_nf4 .412434 |
| 8B 主表 | results_v2/summary.json、results_v2/FINAL_REPORT.md | n=3；significance_test 自曝 Wilcoxon n_pairs=3 无效 |
| 消融 | results_v2_ablation_multiseed/REPORT.md、configs/ablation_ranks/ | 方案A 下六臂塌缩为 4 配置；error_only md5 ≡ uniform（8940e8dbaf65）；product pmt ≡ importance_only {12/4} |
| adapter 配置 | outputs/**/adapter_config.json | 全部方案A 运行 alpha_pattern={}（α=16 固定）→ v_proj α/r=4.0 vs qlora 2.0 |
| 训练日志 | logs/ablation_schemeA_s3407_*.log | 三臂 loss 轨迹重合（差 ~4e-4）；importance_only 日志含 step6 中断拼接（无 resume） |
| 理论文档 | docs/theory-review-report.md、docs/paper-framework.md、docs/paper-outline.md | C2 推导需 H1 单 α；outline §3.3 载「实测 ρ=0.9836」 |
| Prop1' 验证 | docs/figures/erank_vs_error.json | 合成 α=1.5+各向同性 Gauss；monotone PASS；ε=.08 时 Δerank=.0075(+0.62%)，ε=.093 外推 ≈+0.87%≈ε² |
| 14B | results_v2_qwen14b/summary.json、results_v2_qwen14b_seed2025/ | seed42 adaq MMLU .7843 / flex .6588 vs qlora .7837 / .5254；seed2025 仅 MMLU .7840 vs .7836（未入 summary） |
| 实验矩阵 | configs/experiment_matrix.json L63 | budgets [128,256,512,768,1024] 列为必须实验三；results_v2* 无对应目录 |

## Q1 研究问题是否绝对清晰？ — 否：经验问题清晰，动机前提为假且双叙事并存

1. 经验问题本身清晰且可证伪：「冻结 NF4 QLoRA 下，按预计算量化误差做逐层秩分配能否提升下游性能/参数效率？」全部实验围绕它。
2. 但论文意图文档写出的动机前提被项目自测数据否证：docs/paper-framework.md 叙事主线「NF4 对不同模块类型的量化误差 ε_l 可相差数倍」、docs/paper-outline.md Gap-4 同句。实测分模块 ε 均值 q .09353 vs v .09364（差 0.15%），逐层 max/min=1.0876（除 layer0：q .09260–.09502、v .09265–.09483）。「数倍」与实测差约一个数量级。
3. docs/claim-revision-norm-aware.md 已提出 norm-aware 重构并标注「提案、正文未改」→ repo 现存两套互斥叙事（framework/outline 的误差异质性 vs claim-revision 的 norm-aware），按现状投稿必内不一致。
4. 方法对象含糊：理论与算法节（paper-outline §3.4、adaq_precompute.py allocate_ranks）描述逐层分配（方案B），但全部已跑实验为方案A（per_module_type 两个 median）：outputs/adapters/adaq_seed42/adapter_config.json rank_pattern={q_proj:12,v_proj:4}，消融各臂同。逐层信号进实验前被压成 2 个值，「理论对象≠被测对象」。
结论：major-revision 级。修复=应用 claim-revision + 声明方案A/B 边界（修复清单 #1、#2）。

## Q2 动机是否完全被算法与方法论支持？ — 否：三重脱节

1. 动机 vs 数据：rel_error CV 1.47%（8B）/4.0%（14B 含 1 离群层）；Spearman(rel,score)=.144（8B）、−.197（14B）；Var(log score) 分解 importance 占 99.92%。「误差信号」在两个模型的排序中几乎不提供信息。
2. 消融 vs 动机：方案A 下六臂塌缩为 4 配置——error_only 与 uniform 的 per-layer 秩文件逐字节相同（md5 8940e8dbaf65，均 {q8,v8}）；product 与 importance_only 的 per_module_type 相同（{q12,v4}）。故「仅误差 vs 均匀」是字面同一处理，其阴性对误差信号无信息；「product vs importance_only」在方案A 粒度同样分不出乘积项。REPORT.md 的结论在配置层成立，被解读为信号层结论则过度。
3. 算法 vs 公平性：全部方案A 运行（8B 主 adaq、六臂消融、14B adaq）非均匀秩+固定 α=16（alpha_pattern={}）：v_proj α/r=4.0、q_proj 1.33、均匀对照 2.0。scripts/train.py L103-104 注释自认「α 须随 r 缩放(α_l=2·r_l)保证各层有效学习率统一」，而方案A 分支 L87-89 显式 alpha_pattern=None。方法对比因此混杂「秩分配 × 逐模块有效学习率」。
结论：动机不被支持；已跑算法测的是混杂且粒度塌缩的变体。

## Q3 rel_error 估计量与反量化路径实现是否正确？ — 实现正确；两个文档/谱系缺陷

正确点（adaq-lora-research-pipeline/scripts/adaq_precompute.py compute_quant_error L86-127；scripts/measure_epsilon_l.py 同路径）：
1. 量化配置与训练（scripts/train.py）逐字一致：nf4 + double_quant=True + compute_dtype bf16 → 测的 E_l 即训练实际承受的冻结量化误差。
2. bnb.functional.dequantize_4bit(q_param.data, q_param.quant_state) 为官方反量化路径，正确处理 DoubleQuantState 嵌套 absmax（二级 FP8 absmax 逐层还原），无手写反量化。
3. 做差前 .float() 避免低精度舍入污染；fp16 参考（CPU）在 fp16 值域内是 bf16 原权重的无损表示（Llama 权重 O(1)–O(10)，无溢出）。
4. rel=‖w_q−w_fp‖_F/(‖w_fp‖_F+1e-8) 定义标准；恒等式 score≡rel×‖W‖_F≡‖E‖_F 在 64 层精确成立（max|score−rel·imp|=0）。
缺陷：
- D1 谱系裂缝：ranks.json 记录 model=/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B（该路径在本机不存在），而 epsilon_l_report.json 与全部训练/评测用 models/Meta-Llama-3-8B；两产物同层 rel 第 4 位小数不同（.092642 vs .092598）→ 秩分配与 ε 诊断来自两份不同权重拷贝。方案A median 对此差异稳健，但论文须声明或统一重算。
- D2 scripts/generate_ablation_configs.py docstring 与实现矛盾：docstring 写 error_only 为 r∝‖E_l‖_F，代码 L31 实为 rel_error（相对误差）；product 行描述与代码（score）一致。须修（修复 #3）。
结论：估计量与反量化路径正确可复现；D1/D2 为文档/谱系级。
## Q4 ε_l 平坦 / 幂律 α / 零均值三诊断是否成立；代码与图是否自洽？ — 平坦与零均值成立（带边界），H1 不成立；图-数据自洽、图-主张不自洽

1. ε_l 平坦：成立。除 layer0 外 q .09260–.09502、v .09265–.09483，样本 CV 1.4877%；8.71% spread 主要由 layer-0 离群贡献（q .100716、v .0999）。图 panel1 与 JSON 一致（median 线 0.0932、标题 8.7% 均真），但标题 "flat" 未标注 layer-0 结构。
2. 幂律 α（H1）：不成立。实测 α（svdvals(W)、k=0.8n、log-log lstsq）：q_proj med .663（除 layer0 .541–.936）、v_proj med .252（.181–.330）——模块类型间 2.63× 系统性分离；layer-0 q=1.591 离群；r² med .802（min .739）拟合中庸。图 panel2 画出的正是这两条带，却只给单 median 线 α=0.436 且标题 "(H1)"，视觉上暗示假设成立——图与数据自洽、图与主张不自洽。理论链（docs/theory-review-report.md C2：H1 → r*_l ∝ c_l^{2/α}）要求共享预算的各层 α_l≈α：α_q≈.66、α_v≈.25 时指数 2/α 差约 2.6×，单 α 推导不成立。另 H1 的目标对象是 ΔW*−E 的谱（B1 自认不可观测），以 W 的谱替代是未验证代理。
3. 零均值：近似成立、严格不成立。层均值 bias 9.81e-6（max 1.78e-5）≈ mean_abs_e 9.53e-4 的 1.0–1.5%；但对 SE≈3.6e-7（q_proj n≈16.7M、元素 std≈1.47e-3，由 within-block var 2.13e-6 估）bias 为 25–50×SE → 统计显著的小正偏。theory-review A1 的 "approximately zero-mean" 措辞正确；图 panel3 只画 log 尺度绝对值、不给 mean_abs_e 或 SE 参照，视觉过干净。
4. 块独立性（A3，附验）：between/(between+within) med 1.56%、块均值 lag-1 autocorr mean .0001（|·|≤.0074）→ 经验成立；但 measure_epsilon_l.py 的 summary 不聚合 A3（仅 per_layer），论文表需手工聚合。
5. 代码-图一致性：plot_epsilon_distribution.py 数据管线与 JSON 匹配；但两个子图标题硬编码（"flat, ~8.7% spread"、"|mean(E_l)| ~ 10^{-5}"）——对当前数据为真，重生成数据时会静默过期。
结论：三诊断中两项成立（带边界条件）、H1 被否证；图须从「确认式」改「诊断式」呈现（修复 #6、#9）。

## Q5 rel_error 平坦是否物理上判死「量化误差感知」动机？ — 判死「相对误差异质性」版；判不死「绝对误差/范数感知」版；且理论自预言该阴性

1. 物理机制：NF4 按 64-block absmax 归一化尺度，相对 Frobenius 误差是块内权重分布形状（峰度/尾部）的泛函，同质 transformer 各层形状高度一致 → ε_l 近乎常数（8B CV 1.47%）。存活的是尺度异质性：‖W‖_F CV 56.2%。
2. 但非普适律：Qwen2.5-14B rel CV 4.00%、含离群层 .122659（较 median .094280 +30%）→ 异质性可模型条件存在；「数倍」作为普适主张为假，作为模型条件主张可存活。
3. 关键恒等：score = rel×‖W‖_F ≡ ‖E‖_F（实测 max 偏差 0）。故「绝对量化误差感知」与「范数感知」在数值上是同一分配——方法的物理内容在 norm-aware 叙事下完整存活；死的只是相对误差异质性叙事与「乘积项提供增量信息」的主张（方案A 粒度 product≡importance_only）。
4. 理论自预言阴性：命题 1' 的谱抬升在层尺度即能量恒等式 ‖M‖_F²=‖ΔW*‖_F²+‖E‖_F²；项目自做合成验证（docs/figures/erank_vs_error.json）在 ε=.08 给 Δerank=+.0075（+0.62%），外推实测 ε=.093 ≈ +0.87%≈ε²，且 ε 均匀 ⇒ 抬升均匀 ⇒ 无层间差分信号。「ε 平坦 → 按 ε 分配无增益」是理论+测量的自洽推论，而非实验意外。
结论：相对误差版动机判死（且其死法可预言）；绝对误差/范数感知版存活；论文应把阴性结果改写为理论一致性验证。

## Q6 范数感知重构是否唯一存活主张？ — 否：至少五条可 salvage 资产

1. norm/绝对误差感知分配 + 诚实阴性叙事（claim-revision 提案）：8B n=3 四方法差异落在种子波动内（FINAL_REPORT.md），NF4 口径 adaq 最差（results_v2_ptq_nf4/summary.json）——按阴性结果如实写，措辞上限照 claim-revision §5。
2. NF4 逐层诊断数据集（测量贡献）：ε_l 分布含 layer-0 离群；统计显著的 ~1% RMS 零均值偏置（修正「零均值」教科书假设）；块独立性经验验证（between-var 1.56%）；分模块 α 带（q .663 / v .252）；ε₄≈0.02 → 实测 0.093 的修正。独立于方法成败的可复用证据。
3. 同配置噪声底方法论：product vs importance_only rank_pattern 完全相同（{q12,v4}），s3407 flex 差 4.5 点、strict 差 11.5 点（raw results_2026-11-14T11-39-00 / 11-43-02 json：.5246/.4253 vs .4792/.3101）→ 直接量化「训练+greedy 解码」的 run-to-run 噪声。注意另一同配置对（error_only vs uniform）flex 差仅 .0008 → 噪声重尾，「4.5 点噪声底」是 n=1 对观察（修复 #10）。
4. 14B 参数效率线索：adaq 4.62M vs qlora 6.29M（−26.5%），seed42 MMLU .7843 vs .7837、GSM8K flex .6588 vs .5254；seed2025 MMLU raw .7840 vs .7836（results_v2_qwen14b_seed2025/，未入 summary）；seed2025/3407 adapter 已训完（logs/b4_*_s3407.log）但 GSM8K 评测缺 → 补种后可升为 n=3 描述性结论，现仅单种子措辞。
5. α/r 混杂方法论警示：非均匀秩 LoRA 对比中固定 α 使 α/r 随 r 变化 3×（本项目实锤：4.0/1.33 vs 2.0）；可写为 methods caveat 或补 α=2r 重跑短消融。
不存活：相对误差异质性动机、product 优越性、命题 3b（见 F1）、单 α 推导、「数倍 ε」主张。

## 附加发现（六问之外）

- F1 命题 3b 循环验证：paper-outline.md §3.3「命题 3b …（实测: ρ=0.9836 ✅）」；但 adaq_precompute.py self_test L165-169 计算的是 spearmanr(scores, ranks)——score 与分配秩的相关，而秩是 score 的单调水位函数（偏离 1 仅来自截断/取整），循环统计量；且 3b 目标量 σ²_{l,r_l+1}(ΔW*−E) 被 theory-review B1 自认不可观测。3b 当前无有效测量。
- F2 Prop1' 验证域错配：erank_vs_error.json 用合成 α=1.5 + 各向同性 Gauss E；实测为分类型 α 带（.663/.252）、E 带块结构（between-var 1.56%）与 ~1% 偏置 → H_isotropic 被项目自测 E 否证；且实测 ε=.093 处抬升仅 ≈ε²≈+0.9%。只可作 illustrative。
- F3 必须实验三缺失：configs/experiment_matrix.json L63 定义 budgets [128,256,512,768,1024]（framework 标必须），results_v2* 无对应目录 → R-性能曲线缺失，「参数效率」叙事无曲线支撑。
- F4 importance_only s3407 日志含中断拼接：首跑死在 step6/126（首个 checkpoint step10 之前），二跑完整（grep resume=0）→ 最终 adapter 来自单次完整训练、无续训污染；日志体积差（13164B vs 11729B）由此解释，复现文档应注明防误读。
- F5 results_v2/summary.json significance_test 全部条目自注「仅 3 对配对值；Wilcoxon 需 >=6 对才有效」——良好实践，保留并原文进论文。
- F6 14B seed2025 评测仅 MMLU（adaq .7840 / qlora .7836）、无 GSM8K；seed3407 评测目录不存在。

## 编号修复清单

1. docs/paper-framework.md 叙事主线段、docs/paper-outline.md Gap-1/Gap-4：删除「忽视量化误差的层间异质性／ε_l 可相差数倍」。framework 叙事主线替换为：「现有 PEFT 方法（QLoRA、DoRA、AdaLoRA）在 4-bit 量化基座上微调时对所有层分配均匀秩。我们的实测（Llama-3-8B，64 层）表明：相对量化误差 ε_l=‖E_l‖_F/‖W_l‖_F 跨层近乎常数（CV=1.47%，除 layer 0），而绝对量化误差 ‖E_l‖_F=ε_l·‖W_l‖_F 随层范数 ‖W_l‖_F 跨层变化显著（CV=56.2%）。AdaQ-LoRA 按绝对误差（等价地，范数感知）分配秩预算。」outline Gap-4 替换为：「实测 NF4 相对误差 ε_l 跨层近乎常数（Llama-3-8B CV 1.47%），而绝对误差 ‖E_l‖_F 随层范数以 CV 56.2% 变化——Gap-4（修订：绝对误差/范数异质性，而非相对误差异质性）。」并把 docs/claim-revision-norm-aware.md §2/§3/§5 措辞应用到摘要/引言/结论（改名 norm-aware allocation；摘要写阴性但可复核结论）。
2. 论文 §3.4 与 §4.1 加方法对象声明：「正文全部实验使用方案A（per_module_type）聚合：逐层 score 压为模块类型 median（8B budget 512 下 {q_proj:12, v_proj:4}）；方案B（逐层）分配仅为算法能力描述，未进主实验。」（证据：outputs/adapters/adaq_seed42/adapter_config.json rank_pattern。）
3. scripts/generate_ablation_configs.py docstring：error_only 行改为「error_only: r ∝ rel_error_l = ‖E_l‖_F/‖W_l‖_F（相对误差；方案A 下该臂秩文件与 uniform 逐字节相同，md5 8940e8dbaf65）」；product 行改为「product: r ∝ score_l = rel_error_l·‖W_l‖_F = ‖E_l‖_F（完整 AdaQ）」。
4. results_v2_ablation_multiseed/REPORT.md 与论文 §4.3 加设计局限段：「方案A 下六策略塌缩为 4 个不同秩配置：error_only≡uniform（per-layer 秩文件逐字节相同，md5 8940e8dbaf65）、product≡importance_only（per_module_type 均 {q:12,v:4}）。本消融检验的是 4 个配置而非 6 个信号；误差信号的阴性结论应在配置层解释，信号层分离需方案B 重跑。」
5. scripts/train.py 方案A 分支（L87-89）：alpha_pattern=None 改为 alpha_pattern={m: 2*r for m,r in rank_pattern.items()}（与 L103-104 自述原则一致）；论文 Limitations 加：「现有方案A 实验（8B 主表、消融、14B）均固定 α=16 配非均匀秩：v_proj α/r=4.0、q_proj 1.33，均匀对照 2.0，秩分配与逐模块有效学习率混杂；α_l=2r_l 重跑为 future work。」投稿前无法重跑则该披露为必须条件。
6. 理论 H1：docs/theory-review-report.md C2 与 paper-outline §3.3 命题 2 推导链改分模块 α（实测 α_q≈0.66、α_v≈0.25，results_v2/epsilon_l_report.json），或删除 r*_l ∝ c_l^{2/α} 推导、仅保留 Allocation Rule 启发式定位。
7. 命题 3b：删除 paper-outline.md §3.3「（实测: ρ=0.9836 ✅）」与 framework L2 对应引用；3b 改可观测替代验证（score 与逐层下游敏感度、或与 Ŵ 边际奇异值的 Spearman）或整体删除；adaq_precompute.py self_test L165-169 加注释「ρ(score, ranks) 为循环统计量，仅作分配自检，不作命题 3b 证据」。
8. 命题 1'：docs/figures/erank_vs_error.json 与论文图注标 illustrative（合成 α=1.5、各向同性 Gauss E）；补跑实测分类型 α + 实测 E 二阶统计（between/within=1.56%、bias≈1% RMS）一组，或移附录；图注写明实测 ε=.093 处抬升幅度 ≈ε²≈+0.9% erank。
9. scripts/plot_epsilon_distribution.py：两个硬编码标题改由 JSON 计算（f-string）；panel1 标注 layer-0 离群；panel2 改分类型 median 线（q .663 / v .252）、标题改「α per layer (H1 rejected: module-type bands)」；panel3 加 mean_abs_e=9.53e-4 参照线或改纵轴为 |mean(E_l)|/mean|E_l|（median≈1.0%）。
10. 噪声底措辞：results_v2_ablation_multiseed/REPORT.md 的「噪声底 4.5 点」改为「同配置（rank_pattern 相同）单种子对观察：s3407 flex 差 4.5 点、strict 差 11.5 点；另一同配置对（error_only vs uniform，{q8,v8}）flex 差仅 0.08 点 → run-to-run 噪声重尾；故策略间 ≤2.7 点差异不可归因，但 4.5 点不得引用为均匀噪声下界。」
11. 谱系：用训练/评测实际权重拷贝（models/Meta-Llama-3-8B；Qwen snapshot 路径）重算 ranks.json / ranks_qwen14b.json；或在论文 appendix 加 provenance 小节：ranks.json 来自 /mnt/workspace modelscope 拷贝、ε 诊断与训练来自 models/Meta-Llama-3-8B，同层 rel 第 4 位小数差（.092642 vs .092598），方案A median 对该差异不敏感。
12. 零均值措辞：论文 H0 按 theory-review A1 写「approximately zero-mean，带统计显著的小正偏：|mean(E_l)| ≤ 1.78e-5 ≈ mean|e|=9.53e-4 的 1.5%，约 25–50×SE（SE≈3.6e-7）」。
13. 14B：补 seed2025/3407 的 GSM8K 评测（adapter 已就绪）并把 seed2025 MMLU（.7840/.7836）并入 summary 后，方可把参数效率主张升为 n=3 描述性；保留 strict-match 排除说明（Qwen-Instruct 格式）。
14. 实验三：跑预算扫描（configs/experiment_matrix.json L63 budgets [128..1024]）或从矩阵删除并把「参数效率」主张限于单预算（512）证据 + 14B 两点证据。
15. 保留诚实阴性：NF4 五方法表（含 adaq_nf4 GSM8K .406368 < baseline_nf4 .412434）与 significance_test 无效性自曝（n_pairs=3）原文保留，不得删减或软化。

## 评审边界声明

- GPU 被 14B 补种训练占用，本评审零重算/零训练/零推理；全部重算为 CPU 离线对既有 JSON/raw 产物完成。
- docs/figures/erank_vs_error.json 已逐条读取（monotone=true、verdict=PASS、Δerank 数值如文引）；14B seed3407 评测缺失经目录列举确认。
- 「25–50×SE」为评审者估算（within/between-block var 与层元素数 n≈16.7M → SE≈3.6e-7），非项目提供数字。
- 本文件为唯一新建文件（docs/review/nf4-spectral.md），未改动任何共享文件。