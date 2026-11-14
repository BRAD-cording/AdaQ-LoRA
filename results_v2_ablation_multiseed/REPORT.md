# P1-D4 方案A多种子消融汇总

- 种子: [3407, 2025]  (n=2，仅描述性 trend，不做显著性检验)

## 主表 (mean across seeds)

| strategy | MMLU | GSM8K |
|---|---|---|
| error_only | 0.6457±0.0013 | 0.4189±0.0359 |
| importance_only | 0.6456±0.0012 | 0.3700±0.0847 |
| product | 0.6464±0.0032 | 0.4230±0.0032 |
| uniform | 0.6454±0.0003 | 0.4147±0.0450 |
| equal_weight | 0.6456±0.0004 | 0.4124±0.0032 |
| random | 0.6464±0.0020 | 0.4321±0.0375 |

## 逐种子明细

- error_only/mmlu: 0.6466 / 0.6449
- error_only/gsm8k: 0.3935 / 0.4443
- importance_only/mmlu: 0.6448 / 0.6465
- importance_only/gsm8k: 0.3101 / 0.4299
- product/mmlu: 0.6486 / 0.6441
- product/gsm8k: 0.4253 / 0.4208
- uniform/mmlu: 0.6452 / 0.6456
- uniform/gsm8k: 0.3829 / 0.4466
- equal_weight/mmlu: 0.6453 / 0.6458
- equal_weight/gsm8k: 0.4147 / 0.4102
- random/mmlu: 0.6450 / 0.6478
- random/gsm8k: 0.4056 / 0.4587
## 结论与边界（诚实读数，n=2）

1. **product 不占优**：GSM8K 两种子均值 random .4321 > product .4230 > error_only .4189 > uniform .4147 > equal_weight .4124 > importance_only .3700。MMLU 上 product 与 random 并列 .6464，其余 .6454–.6457，全部落在噪声内。
2. **种子间反转**：seed3407 product 双榜第一（.6486/.4253）；seed2025 random 第一（.6478/.4587），product gsm8k .4208 低于 random/uniform/error_only。
3. **唯一稳定特征是方差**：product 的 gsm8k std 最小（±.0032），random ±.0375、uniform ±.0450、importance_only ±.0847——说明 product 的行为可重复，但可重复地不赢。
4. **不可宣称**：本实验**不支持**「product（误差×范数）稳定压过单一信号或随机/均匀基线」。n=2 且种子方差主导，任何"优于"措辞都无统计依据。
5. **与 seed42 单种子结论冲突**：seed42 曾显示 product .4829 显著高于 random .4435，但那是单种子；扩到 2 种子后该差距消失。**原消融结论应视为未复现**。
6. 绝对水位跨种子漂移大（product gsm8k .4829 → .4253 → .4208），提示 LoRA 秩分配在这套设置下的效应量小于种子噪声。
## 补充：GSM8K 严格/宽松两种抽取的对比（关键方法学发现）

| strategy | s3407 strict | s3407 flex | s2025 strict | s2025 flex | flex mean |
| --- | --- | --- | --- | --- | --- |
| error_only | 0.3935 | 0.5254 | 0.4443 | 0.5254 | 0.5254 |
| uniform | 0.3829 | 0.5262 | 0.4466 | 0.5171 | 0.5217 |
| random | 0.4056 | 0.5269 | 0.4587 | 0.5163 | 0.5216 |
| product | 0.4253 | 0.5246 | 0.4208 | 0.5011 | 0.5129 |
| equal_weight | 0.4147 | 0.5224 | 0.4102 | 0.4913 | 0.5069 |
| importance_only | 0.3101 | 0.4792 | 0.4299 | 0.5186 | 0.4989 |

**读法（三条，都影响论文写法）**

1. **strict-match 的 11–15 个百分点差距主要是格式 artifact**：换成 flexible-extract，六策略收敛到 .499–.525，策略间最大差 ≤2.7 个百分点。
2. **格式错误 ≠ 能力差**：`importance_only` 在 s3407 上 strict .3101 / flex .4792。它的数字经常算对，但 `#### N` 输出格式不合规，strict 罚的是格式。
3. **噪声底（决定性）**：`product` 与 `importance_only` 在两种子下 `rank_pattern` **完全相同**（都是 r∈{4,12} 两组分配），即「同一配置的两次独立训练」。二者 flex 差 = |.5246 − .4792| = **4.5 个百分点**（s3407）。**同配置跑间噪声 4.5 点 > 任何策略间真实差 ≤2.7 点** → 六种秩分配在当前评测信噪比下不可区分。

**因此**：strict-match 不应作为消融主判据；任何"策略 A 优于策略 B"的表述都落在噪声内。
