# 预注册：P2-1 成本归一化秩分配实验（cost-normalized rank allocation）

> 状态：**冻结于看数前**（2026-11-16，P2-1 任何训练/评测结果尚未产生）。
> 本预注册遵循 docs/preregistration-14b-3seeds.md（220fd60）与 docs/preregistration-14b-noise-amendment.md（4414d9e）同一纪律：规则先于数据冻结，揭盲后只填实测值不改判据，防 HARKing。
> 目标 venue：**Knowledge-Based Systems（KBS，Elsevier，JCR Q1，中科院 2 区）**，需 P2-1 正向「诊断+修正」双贡献支撑（用户拍板 m06070）。

## 1. 背景与动机（为何这是「修正」候选）

诊断链（已落档，见 claim-revision-norm-aware.md §1-§7 + docs/review/）：
- AdaQ 分秩信号 score_l = rel_error_l × importance_l = ‖E_l‖_F（量化误差 Frobenius 范数）。
- 实测：rel_error 跨层近平坦（CV 1.48%，range 0.0926-0.1007），importance=‖W_l‖_F 跨层差异大（CV 56.7%）。
- ⇒ score 排序 99.8% 由 ‖W_l‖_F 决定（Spearman score-‖W‖ 0.998 vs score-ε 0.144）→ AdaQ 实为**范数感知**非量化误差感知。
- ⇒ 原始分配 r_l ∝ score_l（∝‖W_l‖）在 8B/14B 均未稳定胜过 uniform（六策略不可区分，差异落噪声底内）。

**P2-1 假设（peft §8.4/§12 提出的「最便宜正向修正」）**：原始分配按**秩**正比于 score，忽略了「在宽层（q_proj，d_in+d_out=8192）上加一单位秩比窄层（v_proj，5120）更贵」。成本归一化分配 r_l ∝ score_l/(d_in+d_out) 把秩从宽层转向窄层，在**等秩和**下可能更优。这是对「范数信号」的一个自然修正，若胜过 uniform → 论文从「纯阴性」升级为「诊断 + 可行修正」双贡献。

## 2.  arms（实验组）

全部 8B Meta-Llama-3-8B（models/Meta-Llama-3-8B，4 safetensors 分片），Scheme B **per-layer**（守恒，rank-budget 评审认定 per_module_type median 聚合破坏守恒，故用 per_layer），train_phase3.py（含 F15 α=2r 修 + F4 预算断言 + F21 VRAM reset）。

| arm | 分配 score | 说明 |
|:---|:---|:---|
| **cost_norm** | score_l / c_l，c_l=d_in+d_out（q:8192, v:5120） | P2-1 主修正臂 |
| **raw_score** | score_l（=‖E_l‖_F） | 原 AdaQ per-layer（对照，等价 product 臂） |
| **uniform** | 1.0（每 site 等秩） | 基线（QLoRA 等秩） |

- 三臂均 allocate_ranks(·, budget=512, r_min=4, r_max=128) 守恒 sum-of-ranks=512（复用 adaq-lora-research-pipeline/scripts/adaq_precompute.py 的 allocate_ranks，仅改传入 score）。
- 秩来源：ranks.json 的 layer_stats（64 site × {rel_error, importance, score}），与主实验同源。
- **参数量须实测报告**（c_l 不同 → 等秩和不等参数）：uniform r=8 → 3,407,872；raw_score/cost_norm 待 gen 后实测。若 cost_norm 与 uniform 参数差 >5%，追加 **uniform_pm**（param-matched uniform，调 r 使参数≈cost_norm）作公平对照（fair-baseline F6/M3）。

## 3. 训练配方（与 Phase3 正典一致，修三表三口径）

`--max-length 1024 --batch-size 4 --grad-accum 4 --epochs 3 --dataset-size 2000 --lr 2e-4 --no-packing`（→375 优化步）；NF4（load_in_4bit+nf4+double_quant+bf16）；--rank-scheme per_layer；--budget 512（train_phase3 F4 断言 sum_init_r==512）；MODEL=models/Meta-Llama-3-8B；HF_ENDPOINT=hf-mirror。

## 4. 种子与重复（功效）

- **3 seeds × {42, 3407, 2025}**，每 arm 每 seed 1 run = 3 arm × 3 seed = **9 run**（与 Phase3-D 噪声底测量同规模）。
- 噪声底依据：Phase3-D 实测 8B gsm8k-flex **N=max(N_seed 0.30, N_rep 4.09)=4.09pt**（9run 同 product 配置）。P2-1 效应判据直接绑此实测 N（不另设阈值）。
- 诚实功效声明：n=3 seeds 欠功效（stats 评审 F-系列），P2-1 为**描述性判定**（同号 + 效应 vs 实测噪声底），非频率学显著性检验；若需 TOST/bootstrap CI 升 B11（额外 GPU）。

## 5. 评测（与主实验同口径）

- run_eval_phase3 wrapper 模式：lm_eval 0.4.13，--model hf，--model_args pretrained={BASE},peft={adapter},dtype=bfloat16（BF16 干净基座评测，与主表同；非 NF4），--seed 42（固定评测种子，stats P0-2），mmlu num_fewshot=5 + gsm8k num_fewshot=4，--batch_size auto，--log_samples（供 B11/审计）。
- **主判据任务=GSM8K flexible-extract**（MMLU 无区分度，六策略极差 0.003；GSM8K-flex 是 8B 唯一有方差任务）。strict-match 仅并列报告，**永不作判据**（Qwen/Llama 格式 artifact，预注册 220fd60 R4 沿用）。
- 8B Meta-Llama-3-8B（base 非 instruct）：**不用 chat-template**（与主表/消融同口径；B5 chat-template 仅 14B instruct 适用）。

## 6. 成功判据（冻结，揭盲后只填值不改规则）

设 cost_norm/uniform/raw_score 的 GSM8K-flex 各种子值为 c_s/u_s/r_s（s∈{42,3407,2025}），配对差 d_s = c_s − u_s。

- **R1（正向，支撑 KBS「诊断+修正」）**：mean(d_s) > N（=4.09pt，Phase3-D 实测） **且** 3/3 种子 d_s 同号（全正） **且** cost_norm 参数 ≤ uniform×1.05（等参数或更省）。→ 判「成本归一化修正在等参数下稳定胜过均匀分配 +Δpt（>实测噪声底）」。
- **R2（阴性，强化诊断）**：mean(d_s) ≤ N **或** 符号不一致。→ 判「成本归一化修正未观察到超出噪声底的一致增益；范数信号的宽度修正不足以产生可检出效应」→ 论文维持「机制+阴性+功效边界」四件套，venue 退 TMLR/Neural Networks/Neurocomputing（中科院2区/JCR Q1-Q2 但接受纯阴性）。
- **R3（vs raw_score）**：并列报告 cost_norm − raw_score（检验「成本归一化是否优于原始范数分配」），同 R1/R2 判据（次要，描述性）。
- **R4（参数效率）**：若 R1 成立且 cost_norm 参数 < uniform，可报「等性能更省参数」；若 cost_norm 参数 > uniform 而性能仅持平，**禁**报参数效率（fair-baseline F6 等秩反号教训）。
- **R5（强制披露）**：任何结论须同报 三臂参数量 + 秩预算（sum-of-ranks=512）+ 噪声底 N=4.09pt 来源（Phase3-D）+ n=3 欠功效声明。

## 7. 红线（措辞）

- 禁「优于/等价/打平/显著」除非 R1 字面命中（mean(d)>4.09 且 3/3 同号 且 等参数）。
- 禁跨口径比（BF16 vs NF4 vs 14B vs 8B 不同表不横比）。
- 禁单种子扛均值（若 mean(d) 由单种子主导，如 14B seed42 教训，须标注并降级）。
- P2-1 出数前，cost_norm 任何差值禁进 abstract/intro/conclusion（仅附录/预注册）。
- 若 R2（阴性），诚实写「连成本归一化修正也未检出效应」——这本身强化「平坦误差信号下秩分配路径整体无效」的诊断结论，是 KBS 之外的可发表价值。

## 8. 产物与归档

- ranks：configs/p21/ranks_{cost_norm,raw_score,uniform}.json（gen_p21_ranks.py 生成，记录每臂 sum_ranks/params/per-site）。
- adapters：outputs/p21/{cost_norm,raw_score,uniform}_s{42,3407,2025}/。
- 评测：results_v2_p21/{arm}_s{seed}/（run_eval_p21.py，wrapper 复用 run_eval_ablation）。
- 聚合：results_v2_p21/summary.json + REPORT_p21.md（aggregate_p21.py，三臂 × 三种子表 + 配对差 + R1/R2/R3 判定 + 参数对照 + 噪声底引用）。
- 链：scripts/chain_p21.sh（flock，gate "[phase3] DONE"，gen→train→eval→aggregate→台账→commit→"[p21] DONE"）。
- 全部入 git（results json/md/.done，adapters 仅 config+stats，遵循 Phase-2 入库口径 311c814）。

## 9. 时间线（冻结点声明）

- 本预注册 commit 时间 = P2-1 冻结点。冻结时：Phase3-E 评测进行中（baselines .done=10），Phase3-C/A 未跑，P2-1 ranks 未生成、训练未启动、results_v2_p21/ 不存在。揭盲（读 cost_norm flex 值）必晚于此 commit。
- 揭盲后：只填 §6 的 c_s/u_s/r_s 实测值 + 套 R1/R2/R3，不改 N、不改判据、不改臂定义。

---

## 10. Pre-unblind 修正案（2026-11-16，冻结点 4107552 后、任何训练前）

> 合法性：本修正案在 P2-1 **训练数据产生前**（仅 ranks 已生成，results_v2_p21/ 仍不存在）细化 §2 预先声明的条件臂与 §6 判据，非看数后改规则。冻结点 4107552 的 §2 已预声明「若 cost_norm 与 uniform 参数差 >5% 追加 uniform_pm」——gen_p21_ranks.py 实跑确认条件触发，本修正案执行该预声明。

### 10.1 条件触发确认（gen_p21_ranks.py 实测，configs/p21/）

| 臂 | 分配 score | budget | sum_ranks | params | q:v 中位 | 参数 vs uniform |
|:---|:---|:--:|:--:|:--:|:--:|:--:|
| cost_norm | score_l/c_l | 512 | 512 | **3,693,568** | 11:5 | 1.0838× |
| raw_score | score_l | 512 | 512 | 3,782,656 | 12:4 | 1.1100× |
| uniform | 1.0 | 512 | 512 | 3,407,872 | 8:8 | 1.0000× |
| **uniform_pm** | 1.0 | **576** | 576 | **3,833,856** | 9:9 | 1.1250× |

- 触发：cost_norm 1.0838 > 1.05、raw_score 1.1100 > 1.05 → §2 条件臂 **uniform_pm 启用**（等秩 r=9，budget 576）。
- ★关键公平性质：**cost_norm params 3,693,568 = 0.9634× uniform_pm 3,833,856**（cost_norm 比等参数对照**更省 3.7%**）。⇒ cost_norm vs uniform_pm 是「**更少参数 vs 更多参数的 flat 分配**」对照——若 cost_norm 性能 ≥ uniform_pm，即「等（更省）参数下，成本归一化集中分配胜过均匀分配」=分配本身生效的强阳性，排除参数量混杂（fair-baseline F6/M3 铁律满足）。
- 成本归一化效果实证：q:v 中位从 raw_score 的 12:4 → cost_norm 的 11:5（秩从宽层 q 转向窄层 v，如 §1 假设；peft 预估「翻到约 4:6」偏大，实测为温和再分配，因 per-layer score 仍 q 主导）。

### 10.2 §6 判据精化（绑死等参数对照 + 两阶段确认）

设四种子值（GSM8K flexible-extract，s∈{42,3407,2025}）：c_s=cost_norm, p_s=uniform_pm, u_s=uniform, r_s=raw_score。配对差：dPM_s=c_s−p_s（等参数主对照），dU_s=c_s−u_s（等秩和次对照）。噪声底 N=4.09pt（Phase3-D 实测 N_rep）。

- **R1（阳性，支撑 KBS「诊断+修正」，须同时满足）**：
  1. **3/3 种子 dPM_s > 0**（cost_norm 各种子均胜 uniform_pm，同号；sign-test 单侧 p=0.125）；**且**
  2. **mean(dPM_s) > N（4.09pt）**（效应踩过实测噪声底，非噪声可解释）；**且**
  3. cost_norm params < uniform_pm params（已满足，0.9634×，更省参数）。
  → 判「成本归一化分配在**更省 3.7% 参数**下，GSM8K-flex 稳定胜过等参数均匀分配 +mean(dPM)pt（>实测噪声底 4.09），3/3 种子同号」=分配集中度在等参数口径生效的阳性证据。
- **R1-confirm（两阶段，防 14B n=3 覆辙）**：R1 命中后**不直接定稿 KBS**——n=3 欠功效（stats 评审），须 pre-register 扩 **6 seeds**（追加 3 seed）复跑 cost_norm+uniform_pm，6/6 或 ≥5/6 同号且 mean(dPM) 仍 >N 才升「确证阳性」。R1 命中但 confirm 失败 → 降「初步阳性，n=3 提示，需更多种子」（不进 abstract 主claim）。
- **R2（阴性）**：R1 任一条不满足（mean(dPM)≤N 或 符号不一致）→ 判「成本归一化修正未观察到超出噪声底的一致增益；范数信号的宽度修正不足以产生可检出效应」→ 维持「机制+阴性+功效边界」四件套，venue 退 TMLR/Neural Networks/Neurocomputing（接受纯阴性）。
- **R1b（次，等秩和）**：mean(dU_s)>N 且 3/3 dU_s>0 → 佐证（但 dU 含参数量差 cost_norm 1.0838×uniform，须与 R1 等参数结论一致才可引；若 R1 阴 R1b 阳，归因参数量非分配，禁作阳性）。
- **R3（vs raw_score）**：mean(c_s−r_s) 描述性报告「成本归一化 vs 原始范数分配」，同 R1 判据（次要）。
- **R4/R5**：沿用 §6（参数效率红线 + 强制披露 params/秩预算/N来源/n=3欠功效）。

### 10.3 臂数与成本更新

- 4 臂（cost_norm/raw_score/uniform/uniform_pm）× 3 seeds = **12 run**（原 §2 估 9 run + uniform_pm 3 run）。
- 训练 ~40min/run（375步）× 12 ÷ 2卡 ≈ 4h；评测 mmlu(~25min)+gsm8k-flex(~70min) × 12 ÷ 2卡 ≈ 9.5h → 合计 ~13-15h（略超原估 6-10h，因加 uniform_pm 臂 + 全 mmlu/gsm8k）。
- 若 R1 命中触发 R1-confirm：再 +2 臂（cost_norm/uniform_pm）× 3 新 seed = 6 run ≈ +7h。

### 10.4 产物路径（4 臂）

- ranks：configs/p21/ranks_{cost_norm,raw_score,uniform,uniform_pm}.json（✅ 已生成，本修正案同 commit 入库）。
- adapters：outputs/p21/{arm}_s{42,3407,2025}/。
- 评测：results_v2_p21/{arm}_s{seed}/。
- 聚合：results_v2_p21/summary.json + REPORT_p21.md（aggregate_p21.py，4 臂×3 种子表 + dPM/dU/dR 配对差 + R1/R1b/R2/R3 判定 + 参数对照 + 噪声底 N=4.09 引用 + n=3 欠功效声明 + R1-confirm 触发标志）。


### 10.5 R1-confirm 确认种子冻结（揭盲前，防 HARKing）

冻结时间：2026-11-16（P2-1 主跑 outputs/p21 尚未生成、results_v2_p21 空，仍 pre-unblind）。本条仅指定 §10.2 R1-confirm 已预注册协议的具体种子与判定阈值，不新增/不修改任何判据逻辑。

- **确认种子（追加 3 个，与主跑 {42,3407,2025} 互斥）**：`{1337, 7, 2024}`。合计 6 seeds = {42, 3407, 2025, 1337, 7, 2024}。
- **确认臂**：仅 `cost_norm` + `uniform_pm`（等参数主对照 R1 的两臂）；raw_score/uniform 不重训（confirm 只验主对照稳健性）。= 2 臂 × 3 新 seed = **6 run**（≈+7h，接 P2-1 主跑后）。
- **确认判定（机械套用 §10.2 R1-confirm，只填数）**：设 6 seed 的 dPM_s = cost_norm_flex_s − uniform_pm_flex_s：
  - **确证阳性（→ KBS Q1 主claim）**：≥5/6 种子 dPM_s > 0 **且** mean(dPM over 6) > N(4.09pt) **且** cost_norm params < uniform_pm params（恒成立 0.9634×）。
  - **初步阳性（n=3 提示，降级）**：R1（n=3）命中但 confirm 失败（≤4/6 同号 或 mean(dPM over 6) ≤ N）→ 判「初步阳性，n=3 提示，需更多种子；不进 abstract/intro 主claim，仅附录」。
  - 噪声底 N 仍取 Phase3-D 实测 4.09pt（confirm 不重估 N；若届时 Phase3-D 数据有更新，以更新值为准并记录）。
- **触发条件**：仅当主跑 aggregate_p21.py 输出 `"R1": true` 时自动触发（chain_p21_confirm.sh gate）；R2 阴性则 confirm 永不启动（should_run 门控），省 GPU。
- **产物**：outputs/p21/{cost_norm,uniform_pm}_s{1337,7,2024}/；results_v2_p21/ 追加；aggregate_p21_confirm.py → results_v2_p21/summary_confirm.json + REPORT_p21_confirm.md。
- **诚信声明**：confirm 是 §10.2 预注册的两阶段设计，种子在此冻结于揭盲前；无论 confirm 结果如何均如实报告，阴性/降级不隐藏（防 14B n=3 单种子扛均值覆辙）。
