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