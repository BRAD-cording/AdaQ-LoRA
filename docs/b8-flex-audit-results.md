# B8 flex 抽取器审计结果（F8/P0-4，执行于 2026-11-17）

## 状态：已执行（自动规则双标注版；人工复核可选补充）

## 数据
- 样本：14B seed42 双臂（adaq/qlora）GSM8K 各 400 docs ×2（--limit 400，samples_gsm8k_*.jsonl 各 800 行）
- 工具：scripts/audit_flex_extract.py（inspect/worksheet/kappa）+ scripts/b8_auto_annotate.py（规则A保守/规则B宽松）
- 工作表：docs/b8-worksheet/{adaq,qlora}_annotated.tsv（800 行/臂）

## 核心数字
| 指标 | adaq_seed42 | qlora_seed42 |
|---|---|---|
| flex==gold（答对率口径） | .6700 | .5175 |
| strict==gold | .1775 | .1200 |
| Cohen's κ (规则A vs 规则B) | **0.7015 [FAIL <0.95]** | **0.7535 [FAIL <0.95]** |
| 原始一致率 | .8625 | .8750 |
| flex 漏抽率（输出含 gold 但抓错） | **144/800 = 18.0%** | **212/800 = 26.5%** |

## 发现（三条，全部有样本级证据）
1. **可标注性不达标**：双臂 κ 均 <0.95 → flex 抽取器与「模型意图答案」判断系统性不一致，flex 指标本身不可靠。
2. **双臂差异抽取偏差**：qlora 漏抽率 26.5% > adaq 18.0% → flex 对 qlora 的答对率压得更低 → **实测 flex 差值（14B +8.87pt）方向上被抽取偏差放大**，不能作为 AdaQ 优势证据。
3. **失败机制实证**（doc_id 0）：instruct 模型无 chat-template 时生成后泄漏 fewshot 续写（"### Human: A train travels..."），flex 正则（取最后一个数字串）抓到污染区 '2.' 而非意图答案 $18。strict 极低（.18/.12）同因：instruct 模型不输出 #### 格式。

## 对既有结论的影响
- 14B「flex +8.87pt 正向」在 B8 之后置信进一步下降：除 B11 种子噪声（14.71pt）与 F11 格式错配外，新增**差异抽取偏差**直接威胁差值方向。
- B5（chat-template 复评）成为唯一能澄清 14B 信号的实验：chat-template 消除 fewshot 泄漏后 flex/strict 才可能对齐意图。
- 论文统计段措辞：所有 14B GSM8K flex 差值在 B5 出数前禁入 abstract/intro/conclusion；B8 结果写入评测口径节（flex 抽取器审计不通过，κ=0.70/0.75，双臂漏抽率 18%/26.5%）。

## 诚实边界
- 双标注为**机械规则**（A=flex==gold且gold以数字边界现于输出；B=A或末300字符含答案表述上下文），κ 度量的是规则分歧=可标注性代理，非人工双盲；人工复核可改写 annotA/annotB 列后重跑 kappa。
- flex==gold 与 κ 是两个不同量（答对率 vs 可标注性），本表并列报告，勿混。
