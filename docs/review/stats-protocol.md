# AdaQ-LoRA 投稿前评审 —— 评测口径与统计显著性（Stats & Eval-Protocol Review）

- **评审人设定**：评测口径与统计显著性专家（全量评测 / 任务齐备性校验 / 多种子方差与显著性检验 / 效应量与结论措辞匹配 / 防止把噪声当方法差异）
- **日期**：2026-11-16　**项目**：/home/ubuntu/AdaQ-LoRA
- **评审方式**：SSH 只读实读全部结果文件、训练产物（trainer_state/adapter_config/safetensors）、评测与聚合脚本源码；每个引用数字绑定文件路径；未读材料不下结论
- **Verdict**：**Major Revision** —— 统计诚实性基建是真实强项（无伪 p 值、预注册写死在脚本、噪声底前提经权重级验证）；但主表仅 21 个优化步且 AdaLoRA 剪枝调度未执行、三张表三套训练/评测口径、噪声底仅 2 对观测且从 8B 外推到 14B——当前证据包不能支撑任何方法差异结论（正向或负向）以 Q2 统计标准成稿

---

## 0. 证据闭环清单（本评审亲自实读）

| 材料 | 路径 | 本评审实读的关键数字 |
|---|---|---|
| 8B 主表终报 | `results_v2/FINAL_REPORT.md` + `summary.json` | n=3（seed 42/2025/3407）：qlora MMLU .6533±.0006 / GSM8K-strict .4711±.0046；dora .6543±.0015 / .4799±.0055；adalora .6538±.0012 / .4774±.0009；adaq .6540±.0007 / .4766±.0031。`significance_test` 每任务标注「仅 3 对配对值；Wilcoxon 需 >=6 对才有效」（n_pairs=3，未报任何 p 值）。样本数 [14042]/[1319]/[817]/[541]，12/12 run 三 marker 齐全 |
| 主表训练产物 | `outputs/adapters/adaq_seed42/checkpoint-21/trainer_state.json` | **global_step=21, epoch=3.0, max_steps=21**；log：step10 loss 2.0102 / lr 1e-4，step20 loss 1.4971 / lr 0.0，**num_tokens=593,461（累计）**；loss 末段斜率 ≈0.05/step 仍在下降。checkpoints = 2/7/14/21；adalora_seed42 同为 2/7/14/21 |
| 消融训练产物 | `outputs/ablation_adapters_schemeA_s3407/product/checkpoint-126/trainer_state.json` | global_step=126, epoch=3.0；loss 1.4629→1.1898 |
| 14B 训练产物 | `outputs/adapters_qwen14b/adaq_seed42/checkpoint-255/trainer_state.json` | global_step=255, epoch=3.0 |
| 训练调用 | `scripts/batch_train.py`（主表→train.py 默认：max-length 2048、batch4×accum4）；`scripts/run_ablation_schemeA_multiseed.sh:33-35`（--max-length 1024 --batch-size 4 --rank-scheme per_module_type --epochs 3 --dataset-size 2000）；`scripts/run_b4_qwen14b.sh:35-43`（--max-length 512 --batch-size 2 --grad-accum 8） | 三张表 = 三套训练序列长度（2048/1024/512），lr 均 2e-4、3 epochs、OpenHermes-2.5[:2000] |
| AdaLoRA 调度 | `scripts/train.py`（AdaLoraConfig） | init_r=16, target_r=8, **tinit=20, tfinal=60, deltaT=5, total_step=200（硬编码）**；主表实际 21 步 → 剪枝调度基本未执行 |
| 评测调用 | `scripts/run_eval_missing.py:130`（`"--seed", job["seed"]` = **训练种子**）；`scripts/run_eval_ablation.py:78`（`--seed 42`）；`scripts/run_eval_b4_qwen14b_seeds.py:39`（`--seed 42`） | 主表评测种子随训练种子漂移；消融/14B 固定 42 → 协议不一致。主表/消融 = BF16 基座+adapter（dtype=bfloat16，无 load_in_4bit）；14B = NF4（load_in_4bit=True, nf4, double_quant, compute bf16） |
| 双种子消融 | `results_v2_ablation_multiseed/summary.json` + `REPORT.md` + 原始 `results_*.json`（本评审用 python 逐一重算核对） | 主表 metric=strict-match（`aggregate_results.py:25` 优先级）。strict mean±std（n=2）：error_only .6457±.0013/.4189±.0359；importance_only .6456±.0012/.3700±.0847；product .6464±.0032/.4230±.0032；uniform .6454±.0003/.4147±.0450；equal_weight .6456±.0004/.4124±.0032；random .6464±.0020/.4321±.0375。**flex（手工附加表，本评审从原始 json 全部复核一致）**：s3407 = error .5254 / importance .4792 / product .5246 / uniform .5262 / equal .5224 / random .5269；s2025 = .5254 / .5186 / .5011 / .5171 / .4913 / .5163（均 n=1319 全量） |
| 噪声底前提 | `outputs/ablation_adapters_schemeA_s{3407,2025}/{product,importance_only}/adapter_config.json` + `adapter_model.safetensors` | product 与 importance_only 两种子 rank_pattern 均完全相同 `{'q_proj':12,'v_proj':4}`；s3407 两 run safetensors 128/128 张量全不同（首张量 max abs diff=0.1695）→ 确为同配置独立复训，flex 差 4.55pt 是真实 run 噪声 |
| 14B 结果 | `results_v2_qwen14b/summary.json`；`results_v2_qwen14b_seed2025/{adaq,qlora}/` | seed42：adaq mmlu .78429 / strict .18271 / **flex .65883**，qlora .78372 / .13040 / **.52540**；params 4,620,288 vs 6,291,456。seed2025 仅 MMLU 完成（adaq .7840051 / qlora .7836491，Δ+.036pt）；**GSM8K 正在 GPU 运行**（本评审实查：两个 lm_eval 进程 = adaq_seed2025 + qlora_seed2025）；seed3407 目录不存在 |
| 预注册规则 | `scripts/aggregate_b4_3seeds.py` | 规则 1-4 写死在报告模板文本：①flex 配对差三种子同号且均值>4.5pt（来自 8B 同配置复训经验值）才可写正向；②否则「未观察到一致差异」；③strict 一律不作判据；④必须同时报参数量（−26.5%） |
| 聚合脚本 | `scripts/aggregate_ablation_multiseed.py` | SEEDS=[3407,2025]；自动生成部分只含表格+审计；「结论与边界」与 flex 表为手工附加（本评审已独立复核其数字） |
| 系统指标 | `results_v2/system_metrics.json` | adapter_params：qlora 3,407,872 / dora 3,571,712 / adalora **6,816,768** / adaq **3,801,088**（8B 上 adaq 比 qlora **多 11.5%**）；peak VRAM 14.80–15.01GB |
| PPL | `results_v2_ppl/summary.json` | **sample_len=62**；word_ppl：adaq 7.4314 / qlora 7.4341 / dora 7.4357 / adalora 7.4402（极差 0.009） |
| NF4 表 | `results_v2_ptq_nf4/summary.json` | baseline_nf4 .6364/.4124；adaq_nf4 .6335/**.4064（五方法最低，低于不挂 adapter）**；qlora_nf4 .6334/.4124；dora_nf4 .6343/.4230；adalora_nf4 .6357/.4185 |
| 秩方案产物 | `ranks.json`、`configs/ablation_ranks/*`、12 份 `adapter_config.json` | 所有已训 adaq adapter 均为 per_module_type（类型级 2 组秩），逐层 per_layer_ranks（sum=512 精确）未被任何已训 adapter 使用；v_proj 全层压下界 r=4 |
| random 臂生成 | `scripts/generate_ablation_configs.py:55-57` | `import random as _rnd; scores = {name: _rnd.random() ...}`，**无 _rnd.seed** → 单次抽取，config 固化后两个训练种子复用同一 random 秩模式（与脚本第 10 行注释「多 seed 平均」意图不符） |
| 方向 B 提案 | `docs/claim-revision-norm-aware.md` | E1-E3 证据链；rel_error CV 1.47%（0.0926–0.1008）vs importance CV 56.2%，Spearman .144 vs **.998**；其 E3 表数字与本评审实读的 summary.json 逐项一致 |
| 既有评审 | `docs/review/jcr-gate.md`（Major Revision，reject 倾向）、`docs/review/rank-budget.md`（Major Revision：equal_weight 448/random 480 预算不守恒、α/r 混杂、product 比 uniform 参数多 11.5%） | 本评审与其 flex 均值/噪声底/参数量数字互相吻合；本评审新增发现见 §1 |
| 版本 | 评测日志 | lm_eval 0.4.13、peft 0.21.2、transformers 4.57.6 |

---

## 1. 总判定与新增发现

**Verdict: Major Revision**，与 jcr-gate、rank-budget 两份既有评审一致。本评审新增 4 项两份既有评审均未覆盖的发现：

- **F1（最重）主表仅 21 个优化步，AdaLoRA 剪枝调度未执行** → 8B 主表无检验方法差异的功效，adalora 基线被系统性削弱（详见 §2.2）。
- **F2 三张表三套训练口径**：seq len 2048（主）/1024（消融）/512（14B）；主表 num_tokens≈593K（step20 累计）与消融 ≈2.06M packed tokens（126×16×1024）不可调和——同名「2000 样本×3 epochs」下主表实际消耗数据量存疑，必须解释或重训。
- **F3 主表评测种子协议漂移**：`run_eval_missing.py:130` 用训练种子做评测 `--seed`，而消融/14B 固定 42。评测种子影响 few-shot 示例抽样 → 主表的「种子效应」混杂了 few-shot 变动。
- **F4 random 臂 = 单次未播种抽取**：`generate_ablation_configs.py:55-57` 无 `_rnd.seed`，配置一次生成、两个训练种子复用 → 「随机基线」只实现了 1 个秩抽取，其种子方差只含训练噪声、不含抽取噪声，与脚本注释「多 seed 平均」不符。

---

## 2. Q1 裁决：超参数 / 种子 / 轮数边界是否合理、是否足以支撑结论

### 2.1 宣称配置 vs 实测配置

| 维度 | 8B 主表 | 8B 消融 | 14B |
|---|---|---|---|
| 训练入口 | `batch_train.py`→`train.py` 默认 | `run_ablation_schemeA_multiseed.sh` | `run_b4_qwen14b.sh` |
| max_length | **2048** | **1024** | **512** |
| batch×accum | 4×4 | 4×4 | 2×8 |
| 实测优化步（trainer_state） | **21** | 126 | 255 |
| num_tokens | **593,461（step20 累计，主表独有字段）** | ≈2.06M（trainer_state 无该字段，按 126×16×1024 推算） | ≈2.09M（按 255×16×512 推算） |
| 末段 loss | **1.4971，仍在陡降（前 10 步 2.0102），grad_norm 1.55** | 1.1898（step120，grad_norm 0.43，已趋收敛） | 0.9356（step250，grad_norm 0.30，已趋收敛） |
| 评测基座精度 | BF16 | BF16 | NF4 |
| 评测 --seed | **=训练种子** | 42 | 42 |
| n seeds | 3 | 2（3407/2025） | 1 完成 + 1 在跑 + 1 未启动 |

任务简报所称「3 epoch / 2000 样本 / max-len 512 / lr 2e-4 / batch2×accum8」**只是 14B 口径**；论文实验设置必须按表分别报告七元组（max_len, batch×accum, 优化步数, num_tokens, 评测精度, 评测种子, n seeds）。

### 2.2 F1：主表 21 步问题 —— 对主表结论致命

证据链（全部实读）：
1. `checkpoint-21/trainer_state.json`：max_steps=21、epoch=3.0 —— 3 个 epoch 总共只有 21 个优化步（7 步/epoch）。
2. log_history：step10 loss=2.0102（lr=1e-4，与 21 步 cosine 一致），step20 loss=1.4971（lr=0.0）。**终态 loss ≈ 消融首段 loss（1.4629）**，末 10 步平均下降 0.05/step，未见平台 → 训练在下降段被截停。
3. warmup_ratio 0.03 × 21 步 = 0.63 步 → 无有效 warmup。
4. `train.py` AdaLoraConfig：tinit=20、tfinal=60、deltaT=5、total_step=200（硬编码）→ 21 步内剪枝窗口刚触到 tinit；adalora adapter 参数 6.82M（`system_metrics.json`，约为其他方法 2 倍）说明秩基本停在 init_r=16 未压缩 → **主表 adalora 是「半配置」基线**，其 .4774/.6538 不能解释为「AdaLoRA 方法的表现」。
5. 后果：主表四方法 MMLU 极差 0.10pt（.6533–.6543）、GSM8K-strict 极差 0.88pt（.4711–.4799），全部落在欠训练平台+种子噪声内。**主表既不能支撑「adaq 更好」，也不能支撑「adaq 不如 dora」——它对任何方法差异命题都无功效**；欠训练还会系统性压扁自适应方法（DoRA/AdaLoRA）与静态方法的差异。

**裁决：不合理，不足以支撑结论。** 修复见 P0-1。

### 2.3 其余边界参数

- **3 epochs × 2000 样本**：作为小样本 SFT 探针预算可接受（消融 126 步、14B 255 步是正常运行规模），但主表实际 token 消耗（≈0.6M）与消融（≈2.1M）在同名预算下差 3.4 倍 → 主表数据管道与当前 `train.py` 行为不可调和（可能是历史版本产物），**按现状不可复现**，必须重建（git 哈希+环境）或重训。
- **lr 2e-4 cosine、有效 batch 16**：常规选择，无问题；问题是步数（§2.2）。
- **r_min=4/r_max=128**：`ranks.json` 显示 v_proj 全部 32 层压下界 4 → 逐层分配在 v_proj 上退化为常数，实际分化只发生在 q_proj；论文方法节须如实说明（P2-14）。
- **budget 512 / 64 位点 / r̄=8**：与 qlora r8 基线在 8B 名义对齐 ✓；但类型级 median 塌缩后 equal_weight 有效总秩 448（−12.5%）、random 480（−6.25%）、14B adaq 528（+3.1%）（rank-budget 评审实测，本评审复核其算术一致）→ 预算不守恒使 random/equal_weight 基线容量吃亏。注意方向性：**容量更少的 random 仍赢了 product（strict .4321 vs .4230），所以「product 不占优」的负结论反而更稳**；但论文不能同时宣称「公平预算对比」。
- **种子 42/2025/3407**：选择本身无问题；但 F3（评测种子漂移）与 F4（random 单抽取）使「多种子」的语义打折。

### 2.4 充分性总结

- **消融表**（n=2，126 步）：可支撑「未观察到超出 run 噪声的差异」的描述性负结论；不可支撑「等价/不可区分」（§3.2）。
- **14B 表**（n=1 完成）：当前不能支撑任何方向性结论；seed42 flex +13.34pt 只能写「单种子观察，待复现」。
- **主表**（21 步）：不能支撑任何方法比较结论，须重训或降级重定位。

---

## 3. Q2 裁决：统计报告是否达 JCR Q2 级

### 3.1 做对了的（修订时不得倒退）

1. `results_v2/summary.json` 的 `significance_test` 对全部 5 任务统一标注「仅 3 对配对值；Wilcoxon 需 >=6 对才有效」（n_pairs=3），**没有伪造任何 p 值** ✓
2. `FINAL_REPORT.md` 明写「无有效显著性检验：3 种子 Wilcoxon 最小双侧 p=0.25（n=6 才 0.03125）」并禁用「显著优于/SOTA/p<0.05」✓ —— 数学核对：n=3 全同号时精确双侧 p = 2×(1/2)³ = 0.25；n=6 时 = 2×(1/2)⁶ = 0.03125，**表述诚实且精确** ✓
3. `aggregate_b4_3seeds.py` 把 14B 判定规则 1-4 预注册写死在报告模板 ✓
4. 消融 REPORT「结论与边界」段：product 不占优、种子间反转、「可重复地不赢」、原 seed42 结论未复现、效应量<种子噪声 —— 措辞强度与证据匹配 ✓
5. 表格由聚合脚本自动生成（无手抄），sample_len 审计 `warnings=[]` ✓

### 3.2 不达标的（Q2 差距清单）

1. **n=2 的「描述性 trend」可以接受，但有四个硬条件**：(a) 每处显式标注 *descriptive, n=2 seeds, no inferential statistics*；(b) 逐种子值全部列出（已做 ✓）；(c) 不得使用暗示方向稳定的「trend」措辞，除非有方差机制解释；(d) 区分两种噪声源并分别量化（见第 2 条）。
2. **必须区分「训练噪声」与「评测精度」两个噪声源**：
   - 训练噪声（主导）：同配置复训对 flex 差 1.75–4.55pt（n=2 对，§5）；
   - 评测精度：GSM8K 全量 1319 条在 p≈.52 处二项 SE=√(p(1−p)/1319)≈**1.37pt**；MMLU 14042 条在 p≈.646 处 SE≈**0.40pt**。全量评测=对「这个测试集」无抽样误差，但对「GSM8K 类问题」的推断仍受此精度限制。
   - 修复：消融聚合脚本增加**配对 item-level bootstrap 95% CI**（10k 重采样），并在论文注明该 CI 只含评测精度、不含训练噪声（P1-8）。
3. **「不可区分」措辞过强**：claim-revision 摘要草稿写「该分配与随机/均匀分配在均值上不可区分」。实算：product−random 的 flex 配对差（逐种子）= {−0.23, −1.52}pt，均值 −0.88pt，sd 0.91，df=1，t₀.₉₇₅=12.706 → **95% CI = [−9.1, +7.3]pt**。n=2 下连「等价」都无法统计确立。正确措辞：*「未观察到超出 run 噪声底的差异；该设计不足以确立统计等价」*（英文文本见 §6.2）。
4. **Wilcoxon n=3 p=0.25 的表述诚实 ✓，但建议干脆不跑检验**：报告一个 p=0.25 会诱导读者把它当「无差异的证据」（absence of evidence ≠ evidence of absence）。保留现版「声明检验无效」的写法，不要退化成报 p 值。
5. **多重比较**：主表 4 方法×5 任务、消融 6 策略×2 任务，无任何校正框架。因已全面禁止推断性表述，暂不强制 FDR；但一旦引入 bootstrap CI 或等价性检验，必须声明 per-comparison、未校正。
6. **PPL 表（sample_len=62）无统计意义**：word_ppl 极差 0.009（7.4314–7.4402），62 条样本下远小于任何合理 CI → 从论文删除，或扩到 ≥1000 条并报 CI（P1-11）。
7. **std 口径未注明**：n=2 时样本 sd（ddof=1）与总体 sd 差 √2 倍，REPORT/论文所有 mean±std 处必须注明 *sample SD (ddof=1), n seeds*。
8. **阴性结论的正规化路径**：若论文要正式主张「AdaQ≈QLoRA（参数效率）」，正确工具是 TOST 等价性检验 + 预置 margin（可取噪声底 4.5pt）；n=2-3 种子下功效不足 → 如实写 *underpowered for equivalence*，把结论降级为描述性。

### 3.3 裁决

**现状低于 Q2**（主表欠训练、无 CI、PPL n=62、std 口径缺失、「不可区分」过强）。完成 §3.2 修复后，消融+14B 的阴性结果报告可达 Q2 水位；主表在重训前不得进入论文主文。

---

## 4. Q3 裁决：strict 弃用 / flex 主判据 / 抽取器可靠性

1. **14B 用 flex 作主判据、strict 弃用：正确 ✓**。Qwen2.5-14B-Instruct 不产 `#### N` 格式，strict（adaq .1827 / qlora .1304）度量的是格式合规而非数学能力，且两值都远低于该模型正常 GSM8K 水平；flex（.6588/.5254）量级合理。`aggregate_b4_3seeds.py` 规则 3「strict 一律不作判据」与预注册一致 ✓
2. **但 flex 抽取器可靠性必须人工验证（当前缺失，P0 级）**。lm-eval 0.4.13 的 flexible-extract 用正则抓生成文本中的数字，instruct 模型输出含中间量/多数字时存在错抓风险；flex 一旦成为主判据，抽取器就是测量仪器，仪器必须先校准。修复：分层人工审计——从 14B seed42 与 8B 消融的 gsm8k 原始输出各抽「flex=对∧strict=错」100 条 +「flex=错」100 条，双人独立标注，报告抽取器 precision/recall + Cohen's κ；**precision ≥ 0.95 方可维持 flex 主判据地位**，审计脚本与标注结果随论文发布（P0-4）。
3. **8B 上 strict↔flex 差 11–15pt 同样是格式 artifact**（importance_only s3407：strict .3101 / flex .4792，本评审实读复核）→ 8B 消融的结论段已正确地以 flex 为准（REPORT 附加 flex 表）；但 `summary.json` 主表 metric 仍是 strict-match —— 论文引用时必须写明「8B 消融结论以 flexible-extract 为准，strict 仅存档」。
4. **每表单一主判据 + 四元组表头**：主表=BF16+strict、消融=BF16+flex（结论）、14B=NF4+flex、NF4 表=NF4+strict——四套口径并存是事实，无法事后统一；论文每张表的表头必须标注（基座精度, 主判据 metric, 评测种子, n seeds）四元组，**正文禁止任何跨表数值比较**（例如 14B flex .6588 与 8B strict .4766 之间没有可比性）。`eval-protocol-v2.md` 的「一次口径一个 OUT_DIR」在数据层已落实 ✓，表级继承即可（P1-12）。
5. **生成确定性**：gsm8k generate_until 需确认 greedy（temperature=0）并在论文注明；评测 `--seed` 影响 few-shot 抽样 → 全部统一为固定 42（修复 F3，P0-2）。

---

## 5. Q4 裁决：预注册判定规则的方法学成立性

规则原文（`aggregate_b4_3seeds.py` 写死）：①gsm8k_flex 配对差三种子同号且均值 > 4.5pt（来自 8B 同配置复训经验值）才可写正向差异；②符号不一致或均值在 ±4.5pt 内 → 「未观察到一致差异」；③strict 一律不作判据；④必须同时报参数量。

### 5.1 规则的统计性质 —— 作为「预注册描述性判定规则」成立，作为「检验」不成立

- 「三种子同号」= 符号检验：H₀（无差异、符号随机）下 P(3 种子全同号) = 2×(1/2)³ = **0.25** → 若当假设检验用，family-wise α=0.25，远超常规 0.05，**不是校准检验**。
- 叠加「均值 > 4.5pt」的最小重要差（MID）门槛后，联合规则的假阳性率取决于噪声分布，无解析校准 → 论文必须把它表述为 *pre-registered descriptive decision criterion*，**不得**写成 significance test / statistically significant（§6.4 英文文本已按此措辞）。
- 在 n=3 物理上无法做出 p<0.05 的约束下（§3.1.2），用「符号一致性 + 效应量门槛」替代无效检验，是**当前样本量下诚实且可辩护的次优选择** ✓。预注册时点（评测完成前写死在聚合脚本）可信 ✓。

### 5.2 噪声底 4.5pt 的可靠性 —— 全链条最薄弱环节

1. **前提成立（本评审权重级验证 ✓）**：product 与 importance_only 两臂 rank_pattern 在两个种子下都完全相同（`{'q_proj':12,'v_proj':4}`），s3407 两 run 的 safetensors 128/128 张量全不同（首张量 max|Δ|=0.1695）→ 确为同配置两次独立训练，其 flex 差是纯 run 噪声（GPU 非确定性+数据顺序）。附带说明：两臂模式相同并非偶然——rel_error 层间近乎常数（CV≈1.47%，claim-revision E2；rank-budget 评审独立复核 q_proj CV=1.54%/v_proj CV=1.40%）使 score=rel×‖W‖ 与 ‖W‖ 单独排序在 median 塌缩后同型，这本身就是 E2 的一个佐证。
2. **估计量只有 2 对观测**：s3407 |.5246−.4792| = **4.55pt**；s2025 |.5011−.5186| = **1.75pt**。REPORT 取 4.5 = 取大者（保守方向正确 ✓），但：
   - σ̂_d = √((4.55²+1.75²)/2) ≈ **3.45pt**（df=2）→ σ_d 的 95% CI（χ²₂）≈ **[1.8, 21.7]pt**：噪声底本身有数量级不确定性；
   - 门槛 4.5pt = 1.31σ̂_d → 正态近似下单次比较单侧 p≈**0.10**，不是 0.05；
   - 两对观测差 2.6 倍（4.55 vs 1.75），说明该噪声尺度本身高度不稳定。
3. **跨设定外推未验证**：噪声底产自 8B / seq1024 / 126 步 / BF16 评测 / base 模型；14B 是 seq512 / 255 步 / NF4 评测 / instruct 基座。instruct 模型输出格式更稳定，run 噪声可能更小（门槛过保守）也可能因生成路径更长而更大（门槛过松）——**方向未知，无实证**。
4. **附带观察**：run 噪声（1.75–4.55pt）≥ 种子间效应（product 两种子 flex 差 2.35pt）→ 在该训练规模下单 run 的 GSM8K flex 测量精度就是 ±3-4pt 量级，**任何 <9pt（2×上界）的单 run 差异都不可信**；这也解释了为何 strict 表上 importance_only 的 ±.0847 巨大 std 不应过度解读。
5. **修复阶梯**（按成本）：
   - (a) 最低成本（P0-5，必做）：论文与 REPORT 把噪声底如实报告为「两对同配置复训给出 4.55 / 1.75pt，取保守上界 4.5pt 作门槛；该阈值未在 14B 设定重估」；
   - (b) 中成本（P1-9，强烈建议）：14B 追加一对同配置 qlora×2（seed42）复训+GSM8K 评测（≈2 训练 + 2×70min 评测），把 14B 噪声底升级为本地 n≥3 对估计；
   - (c) 彻底：每个（模型, 口径）单元独立估噪声底，≥3 对复训。

### 5.3 规则执行现状

- seed42：flex 配对差 **+13.34pt**（.65883−.52540）> 4.5 ✓（同号 1/1）；
- seed2025：MMLU 已完成（Δ+.036pt≈噪声），**GSM8K 正在 GPU 上运行**（本评审实查进程）；
- seed3407：目录不存在，未启动。
- → **预注册规则当前不可判定；14B 任何方向性结论在论文中冻结**，直到三种子齐备并跑 `aggregate_b4_3seeds.py` 产出 `summary_3seeds.json`（P0-7）。即便届时三种子同号且均值超阈，措辞上限也是「与预注册判据下的正向效应一致」，并同时报参数量差（adaq 少 26.56%，即文件所称 26.5%）与噪声底外推限制。
- 顺带（P2-13）：error_only flex 两种子恰同为 .5254（=693/1319 两次）；在 p≈.525、n=1319 下两次同计数的巧合概率 ≈4.5%，属可接受巧合，建议加脚注预防审稿人质疑数据复用。

---

## 6. Q5：论文统计报告段精确英文文本

> 以下四段可直接进论文；数字全部经本评审实读核对。方括号内为作者待办/条件分支。

### 6.1 Statistical protocol（Experimental Setup 节）

> **Statistical protocol.** All benchmarks are evaluated on their full test splits (MMLU: 14,042 items, 5-shot loglikelihood; GSM8K: 1,319 items, 4-shot, greedy decoding) with lm-evaluation-harness v0.4.13 (peft 0.21.2, transformers 4.57.6); no subsampling (`--limit`) is used in any reported table. The evaluation seed is fixed to 42 for all runs. We distinguish two noise sources: (i) *training-run noise*, quantified by replicate trainings under identical configurations (same seed, same rank pattern), and (ii) *evaluation precision*, quantified by item-level paired bootstrap over the full 1,319 GSM8K items (10,000 resamples, 95% percentile intervals); for a fixed adapter the binomial standard error at p≈0.52 is ≈1.4 points on GSM8K and ≈0.4 points on MMLU. Central tendency is reported as mean ± sample standard deviation (ddof = 1) across n training seeds, with per-seed values listed in the appendix. Because every reported cell has n ≤ 3 seeds, no null-hypothesis significance test can attain p < 0.05 (the minimum two-sided p of a Wilcoxon signed-rank test with n = 3 pairs is 0.25; n = 6 pairs would be required to reach 0.03125). We therefore make no claim of statistical significance anywhere in this paper, and instead report descriptive statistics, pre-registered decision criteria, and explicit noise floors.

### 6.2 8B 消融段（阴性结果，n=2）

> **Rank-allocation ablation (Llama-3-8B, BF16 evaluation, n = 2 seeds).** Six allocation strategies were compared under identical training configuration (max length 1024, effective batch 16, 3 epochs over 2,000 OpenHermes-2.5 samples, 126 optimizer steps) and a nominal total-rank budget of 512 over 64 q/v projection sites; after type-level aggregation the realized effective totals are 512 for product / importance-only / error-only / uniform, 448 for equal-weight, and 480 for random (see Limitations). On GSM8K (flexible-extract, full 1,319 items), the proposed product score attained 0.5129 ± 0.0167 (mean ± SD over seeds 3407/2025; per-seed 0.5246 / 0.5011), versus uniform 0.5217 ± 0.0064, random 0.5216 ± 0.0075, equal-weight 0.5069 ± 0.0220, error-only 0.5254 ± 0.0000, and importance-only 0.4989 ± 0.0279. The six strategy means span 2.7 points — *less than the run-to-run noise floor of this setup*: two independent trainings of the **identical** rank configuration (the product and importance-only arms induce the same type-level pattern {q_proj: 12, v_proj: 4}, since the relative NF4 reconstruction error is nearly constant across layers; all 128 adapter tensors nonetheless differ) scored 0.5246 vs 0.4792 on seed 3407 — a 4.55-point spread attributable solely to training non-determinism; the second replicate pair (seed 2025) gave 1.75 points. **We therefore observed no difference beyond the noise floor between the proposed allocation and the random/uniform baselines. With n = 2 seeds this design is underpowered to establish statistical equivalence** (95% CI for the product−random paired mean difference across seeds: [−9.1, +7.3] points); we report a negative, fully reproducible observation rather than evidence of equivalence. [若采纳 P1-8：Item-level paired bootstrap CIs, which isolate evaluation precision from training noise, are reported in Appendix X.]

### 6.3 8B 主表段（仅当按 P0-1 重训后使用；括号内为保留 21 步数据时的降级写法）

> **Main comparison (Llama-3-8B, BF16 + PEFT adapters, n = 3 seeds).** MMLU (5-shot, full split) and GSM8K (4-shot, strict-match) are reported as mean ± SD over training seeds 42/2025/3407. [重训后：Across QLoRA, DoRA, AdaLoRA and AdaQ, MMLU ranges over 0.6533–0.6543 and GSM8K over 0.47xx–0.47xx; pairwise differences are within one pooled seed SD and no method ranking is claimed. Adapter parameter counts differ by method (QLoRA 3.41M, DoRA 3.57M, AdaQ 3.80M, AdaLoRA 6.82M) and are reported alongside every accuracy figure.]
> [若保留 21 步数据（仅可入附录）：The main-table runs completed only 21 optimizer steps (≈0.59M tokens; loss still decreasing at termination, 2.01 → 1.50 over the last ten logged steps, with the cosine schedule fully annealed). In this under-trained regime all four methods remain on a near-base-model plateau (MMLU 0.6533–0.6543; GSM8K-strict 0.4711–0.4799), the AdaLoRA pruning schedule (t_init = 20, t_final = 60 of a hard-coded 200-step budget) never executed, and the table lacks power to differentiate allocation strategies; it is reported for completeness only.]

### 6.4 14B 段（预注册规则；三种子齐备前冻结方向性结论）

> **Scale-up study (Qwen2.5-14B-Instruct, NF4 evaluation).** Because the instruct-tuned base does not emit the `#### N` answer format, strict-match scores (AdaQ 0.1827, QLoRA 0.1304) measure format compliance rather than mathematical ability and are reported for completeness only; all conclusions use flexible-extract [pending the extractor audit of §P0-4: a stratified manual audit of N = 400 generations yielded extractor precision X.XX / recall X.XX, Cohen's κ = X.XX]. A decision rule was pre-registered in the aggregation script before any 14B evaluation completed: a positive effect of AdaQ over QLoRA may be claimed only if the paired flexible-extract differences are (i) of the same sign in all three training seeds and (ii) exceed +4.5 points on average — the latter being the run-to-run noise floor measured on the 8B setup from two same-configuration replicate trainings (observed spreads 4.55 and 1.75 points; the conservative upper value is used). This rule is a pre-registered descriptive criterion, not a calibrated significance test: a same-sign rule across three seeds corresponds to a sign-test family-wise error rate of 0.25, and the threshold is extrapolated from the 8B setting without re-estimation at 14B. Seed 42 yielded +13.3 points (AdaQ 0.6588 vs QLoRA 0.5254 on 1,319 items) with 26.5% fewer adapter parameters (4.62M vs 6.29M); [seeds 2025/3407 pending → 冻结；齐备后：per-seed differences were +A.B, +C.D, +E.F (mean +G.H), satisfying / not satisfying the pre-registered criterion; we therefore report this as consistent with a positive effect under the pre-registered criterion, bounded by the single-configuration noise-floor estimate and by cross-table protocol differences (NF4 vs BF16 evaluation).]

### 6.5 措辞强度对照表（作者自查）

| 证据状态 | 允许的最强措辞 | 禁止 |
|---|---|---|
| n=1 超噪声底（14B seed42） | "a single-seed observation of +13.3 points, pending replication" | "AdaQ outperforms / improves" |
| 3 种子同号且均值>4.5pt | "consistent with a positive effect under a pre-registered criterion" | "significantly better", "p<0.05", "SOTA" |
| n=2、差<噪声底（8B 消融） | "no difference beyond the run-to-run noise floor was observed; underpowered to establish equivalence" | "equivalent", "indistinguishable", "as good as" |
| 主表 21 步（若不重训） | "all methods remain on a near-base-model plateau in this under-trained regime" | 任何方法排序、任何「不输/更优」 |
| strict 与 flex 并存 | 每表单一主判据 + 口径四元组表头 | 跨表数值比较、跨 metric 比较 |

---

## 7. 编号修复清单

**P0（投稿前必须完成）**

- **P0-1 主表重训或降级重定位** —— `scripts/batch_train.py` + `outputs/adapters/*` + `results_v2/`：以对齐消融的步数规模重训主表 4 方法×3 种子（如 `--max-length 1024` → 126 步，或统一 token 预算 ≈2.06M），重跑 `run_eval_missing.py` 评测与聚合；若选择不重训，主表整体移入附录并加 §6.3 降级写法，主比较改由消融+14B 承担。**同时**修 `scripts/train.py` AdaLoraConfig：`total_step=200` 硬编码改为从实际 `max_steps` 动态传入（否则任何短调度下 adalora 基线都被削弱）。
- **P0-2 评测种子协议统一** —— `scripts/run_eval_missing.py:130`：`"--seed", job["seed"]` → `"--seed", "42"`；重跑受此影响的主表 gsm8k/mmlu 评测（或在论文附录报告「few-shot 抽样对 seed 不变」的验证输出）。
- **P0-3 阴性结论措辞降级** —— `docs/claim-revision-norm-aware.md` 摘要草稿：「与随机/均匀分配在均值上不可区分」→「未观察到超出 run 噪声底的差异；n=2 设计不足以确立统计等价」（英文直接采用 §6.2）。
- **P0-4 flex 抽取器人工审计** —— 新建 `scripts/audit_flex_extract.py`（仅此新文件+产物）：从 `results_v2_qwen14b/*/` 与 `results_v2_ablation_multiseed/*/` 的 gsm8k 原始输出分层抽样（「flex对∧strict错」100 条 +「flex错」100 条，×2 模型），双人独立标注，输出 precision/recall/Cohen's κ 至 `docs/review/flex-audit.md`；precision<0.95 则 flex 不能作主判据，须修抽取正则后重评。
- **P0-5 噪声底改区间报告 + 外推声明** —— `results_v2_ablation_multiseed/REPORT.md` 噪声底段与论文：「噪声底 4.5 点」→「两对同配置复训 spread = 4.55 / 1.75 pt；取保守上界 4.5 pt 为判定门槛；σ̂_d≈3.45 pt（df=2，95% CI [1.8, 21.7]）；该阈值未在 14B 设定重估」。
- **P0-6 超参按表报告 + 主表 token 对账** —— 论文实验设置：三张表分别报（max_len, batch×accum, 优化步数, num_tokens, 评测精度, 评测种子, n seeds）七元组；纠正任何把「512/2×8」当全局配置的表述；对主表 593K tokens vs 消融 ≈2.06M 的差异给出数据管道解释（git 版本/预处理），解释不了就按 P0-1 重训。
- **P0-7 14B 三种子补完** —— 等 seed2025 GSM8K 跑完（GPU 正在运行，勿动），启动 seed3407（`run_b4_qwen14b_seeds.py`），完成后跑 `scripts/aggregate_b4_3seeds.py` 产出 `results_v2_qwen14b/summary_3seeds.json` + `REPORT_14B.md`；在此之前论文 14B 结论冻结（§6.4 pending 分支）。

**P1（强烈建议）**

- **P1-8 item-level 配对 bootstrap CI** —— `scripts/aggregate_ablation_multiseed.py`：为六策略 flex 差增加配对 item-level bootstrap（10k 重采样，95% percentile），输出进 summary.json；论文注明该 CI 只含评测精度、不含训练噪声。
- **P1-9 14B 本地噪声底** —— 追加一对 14B 同配置 qlora×2（seed42）复训+GSM8K 评测（≈2×训练 + 2×70min），噪声底升级为 14B 本地 n=3 对。
- **P1-10 random 臂单抽取说明/扩充** —— `scripts/generate_ablation_configs.py:55-57`（无 `_rnd.seed`）：论文说明 random 臂为单次秩抽取（config 已固化发布，可复现）；理想做法是 2-3 个抽取×种子并入 random 臂均值，否则「product≈random」只对照了一个随机实现。
- **P1-11 std 口径注明 + PPL 表处置** —— 所有 mean±std 处注明 *sample SD (ddof=1), n seeds*；`results_v2_ppl/`（sample_len=62）从论文删除，或扩至 ≥1000 条 + bootstrap CI。
- **P1-12 表级口径四元组** —— 论文每张结果表表头标注（基座精度, 主判据 metric, 评测种子, n seeds），正文加「禁止跨表比较」声明（继承 `docs/eval-protocol-v2.md` §1）。

**P2（打磨）**

- **P2-13** `results_v2_ablation_multiseed/REPORT.md` 加脚注：error_only flex 两种子同为 693/1319（.5254），巧合概率 ≈4.5%，非数据复用。
- **P2-14** 论文方法节如实说明：r_min=4 使 v_proj 全部 32 层压下界，逐层分配实际只在 q_proj 分化；且所有已训 adaq adapter 用的是类型级（per_module_type）秩，逐层 per_layer_ranks 未被使用——方法叙事必须与实现对齐（与 rank-budget 评审、jcr-gate 的「Quantization-Aware 名不副实」结论一致）。
- **P2-15** `docs/paper-outline.md` / `docs/paper-framework.md` 的标题与贡献句（"Quantization-Aware"「参数效率提升」）同步到 claim-revision 方向 B；注意 8B 上 adaq 参数 3.80M **多于** qlora 3.41M（+11.5%），「参数效率」仅在 14B（−26.5%）成立，全局叙事必须限定尺度。

---

## 8. 三句话总评

1. **统计诚实性基建是真实强项**：不伪造 p 值、Wilcoxon n=3 最小 p=0.25 的声明数学精确、预注册规则写死在聚合脚本、噪声底前提经 safetensors 权重级验证成立——这套自律在 Q2 阴性结果论文中属于达标水位。
2. **但证据包当前撑不起任何方法差异结论**：主表只有 21 个优化步且 AdaLoRA 剪枝未执行（对方法比较无功效）、三张表三套序列长度/两种评测精度/两种评测种子协议（跨表不可比）、阴性结论「不可区分」在 n=2 下 95% CI 宽达 [−9.1,+7.3]pt（统计上连等价都未确立）、4.5pt 噪声底仅 2 对观测且从 8B 外推 14B（σ 的 95% CI [1.8,21.7]pt）。
3. **裁决 Major Revision**：完成 P0-1～P0-7（主表重训或降级、评测种子统一、措辞降级、flex 人工审计、噪声底区间化、超参分表报告、14B 三种子补完）后，「范数感知+阴性结果」的叙事可达 Q2 投稿标准；14B 方向性结论在预注册规则判定前保持冻结。
