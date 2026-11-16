# B8：GSM8K flexible-extract 抽取器人工审计（计划 + 状态更正）

## 状态更正（重要）
JCR 复核（docs/review/jcr-14b-verification.md）把 B8 列为「零 GPU」，**前提不成立**：
全仓库 `samples_*.jsonl` 计数 = 0（14B / ablation / phase3_noise 全部评测从未用 `--log_samples`），
模型原始生成文本已丢弃 → 无法直接审计 flex 抽取。B8 实为 **GPU 任务**（须重跑捕获 generations）。
这同时是 repro-eng 指出的样本级溯源缺口；Phase-3 起评测应按需开 `--log_samples`。

## flex 指标为何是审计对象
14B 全部正向结论建立在 lm_eval gsm8k 的 **flexible-extract（flex）** 上：
- strict-match 正则 `#### (\-?[0-9\.\,]+)`：要求模型输出 `#### N`，Qwen2.5-Instruct 不输出 → strict 失真（.18 vs 官方≈.94）。
- flexible-extract 正则 `(-?[$0-9.,]{2,})|(-?[0-9]+)`，`group_select=-1`：取生成中**最后一个**数字串。
  instruct 模型自由生成里「最后一个数字」可能是推理步骤号/中间量而非最终答案 → 抽取有效性存疑。
- 14B 信号 = adaq−qlora flex 配对差 +8.87pt（3 seed 全正）。若 adaq 臂 flex 抽取系统性「过度捕获」
  （更易抓到碰巧等于答案的数字），可制造假阳性 → **必须双臂审计差别抽取偏差**。

## 已交付工具（零 GPU 部分已完成）
- `scripts/audit_flex_extract.py`（py_compile OK + 单元/端到端自测 ALL PASS）：
  - 忠实复刻 lm_eval flex/strict 正则 + `norm_num`（去 `$`/`,`/尾点，与 lm_eval exact_match 一致）+ Cohen's κ。
  - `inspect <samples.jsonl>`：打印首条记录键+样例（**适配 lm_eval 0.4.13 真实 schema，采样后先跑这个**）。
  - `worksheet <dir> --arm NAME --out TSV`：生成双标注工作表（doc_id/问题/gold/原始输出/flex抽取/strict抽取/自动比对 + annotA/annotB 空白列）。
  - `kappa <annotated.tsv>`：算 κ(A vs B) + flex==gold 准确率 + verdict（κ≥0.95 PASS）。
- `scripts/run_b8_flex_audit.sh`（bash -n OK，已启动 waiter PID 1849739，gate 在 phase3 DONE，GPU1，@reboot cron）：
  - 逐字复刻 run_eval_b4_qwen14b_seeds.py 的 14B NF4 评测命令 + `--limit 400 --log_samples`。
  - 审计 adaq_seed42 + qlora_seed42 两臂各 400 docs → `results_v2_audit_flex/{arm}_seed42/`。
  - 采样后自动跑 inspect + worksheet。

## 审计设计
- 双标注：annotA / annotB 各独立判「flex_extracted 是否捕获模型意图答案」（Y/N/UNCLEAR）。
- 阈值：Cohen's **κ ≥ 0.95**（标注任务定义清晰、flex 抽取可审计）。
- 同时报告：flex==gold 自动准确率（「答对率」）+ κ（「可标注性」）+ 双臂差别（adaq vs qlora 抽取偏差）。
- 样本量：每臂 400，共 800 标注点，远超 κ 估计所需。

## 待办（GPU 排队，phase3 之后）
1. phase3 DONE → run_b8_flex_audit.sh 自动采样（adaq+qlora seed42，~1h GPU1）。
2. 采样后先 `inspect` 确认真实 samples schema 字段映射（lm_eval 版本差异；get_raw_gen/get_doc 已防御式多键探测，不符则改字段映射）。
3. `worksheet` 生成 TSV → 分派两名标注员双标 → `kappa` 出 κ。
4. κ≥0.95 且双臂抽取无系统偏差 → B8 闭合，flex 结论可保留（仍受 F11 chat-template + F15 α/r 两替代解释约束，见 claim-revision §6）。
