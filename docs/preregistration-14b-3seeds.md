# P1-B4 14B 三种子实验预注册（Pre-registration）

- **规则冻结时间**：2026-11-15 ~13:30 UTC —— 判定规则固化于 `scripts/aggregate_b4_3seeds.py` 第 98–101 行，脚本 mtime：`2026-11-15 13:37:14.586534594 +0000`（配套训练脚本 `run_b4_qwen14b_seeds.sh` mtime：`2026-11-15 13:37:16.400528213 +0000`），早于 seed2025 首个评测结果产生时间（15:32 UTC）。
- **落档时间**：2026-11-15 16:14 UTC（本文件 commit 时刻）。
- **状态**：主判定指标（seed2025/3407 的 gsm8k_flex）**尚未揭盲**，本预注册构成确证性分析依据。

## 1. 假设

- H1：在 Qwen2.5-14B-Instruct（NF4 评测协议）上，adaq − qlora 的 GSM8K flexible-extract 配对差在三种子（42/2025/3407）下一致为正且均值超过噪声底。
- H0：配对差符号不一致，或均值落在 ±4.5 点以内。

## 2. 主指标与判定规则（冻结，不得事后修改）

主指标：**gsm8k exact_match,flexible-extract**（全量 1319 条、4-shot、NF4 加载 load_in_4bit+nf4+double_quant+bf16 compute、评测 --seed 42 固定、HF_HUB_OFFLINE=1）。

1. 若配对差三种子**同号**且**均值 > 4.5 点** → 方可写「14B 规模上 AdaQ 相对 QLoRA 有正向差异」。
2. 若符号不一致，或均值落在 ±4.5 点内 → 只能写「未观察到一致差异」。
3. **strict-match 永不作为判定依据**（Qwen-Instruct 输出不满足 #### N 格式约束，seed42 实测 strict .18/.13 vs flex .66/.53，strict 度量的是格式合规性）。
4. 任何对比必须**同时报告 adapter 参数量**（adaq 4,620,288 vs qlora 6,291,456，AdaQ 少 26.5%）。

## 3. 噪声底依据与局限

4.5 点来自 8B 同配置双跑经验值：product 与 importance_only 在两种子下 rank_pattern 完全相同（r∈{4,12}），即同一配置的两次独立训练，flex 差 |.5246−.4792| = 4.5 点，大于六策略真实极差 2.7 点（见 results_v2_ablation_multiseed/REPORT.md）。
**局限（如实声明）**：噪声底仅由一对同配置 run 估计，是保守但高方差的阈值；后续将以 ≥2 对复算并给出 CI（见 docs/review/jcr-gate.md P0-4）。

## 4. 揭盲状态披露

| 数据 | 揭盲状态（相对规则冻结/本落档） |
| --- | --- |
| seed42 全部（flex adaq .6588 / qlora .5254） | 规则冻结前已知 → **定位为探索性前例**，不构成确证证据；三种子规则对其同样生效 |
| seed2025 MMLU（.7840 vs .7836，15:32 UTC） | 规则冻结后、本落档前揭盲；MMLU 为次要指标，不参与主判定 |
| seed2025/3407 gsm8k_flex | **本落档时未揭盲** —— 主判定干净 |

## 5. 出数后的写作上限（两分支预置）

- **H1 成立**：允许「在 14B 规模上，AdaQ 以少 26.5% 的 adapter 参数取得与 QLoRA 持平的 MMLU，并在 GSM8K flexible-extract 上取得三种子一致、超过噪声底的正向差异（配对差均值 X 点）」；仍禁止写「显著优于」「具备规模可扩展性」，须附单模型家族限制。
- **H0 成立**：必须写「未观察到一致差异」，与 8B 双种子阴性并列为**跨规模一致证伪**，叙事转向机制解释（score 99.8% 范数驱动）与 falsification 贡献；禁止淡化或只报有利种子。

## 6. 执行记录

- 链路启动：2026-11-15 13:37:22 UTC（logs/chain_b4_seeds.log，chain PID 1196644，flock /tmp/adaq_b4_seeds.lock）
- 训练完成：2026-11-15 14:44:20 UTC，6/6 adapter 落盘（adaq/qlora × seed42/2025/3407），Peak VRAM 17.60–17.62 GB
- 聚合与判定：scripts/aggregate_b4_3seeds.py → results_v2_qwen14b/summary_3seeds.json + REPORT_14B.md（自动生成，判定逻辑与本文件第 2 节一致）
