# Phase-3 D 聚合报告

结果目录：`results_v2_phase3_noise`　run 数：9　组数：1

| 组 | MMLU acc | GSM8K strict | GSM8K flex |
|---|---|---|---|
| product | — | 0.4760 ±0.0129 (n=9, range 0.0417) | 0.4939 ±0.0129 (n=9, range 0.0409) |

## 噪声底（同 product 配置 9 次独立训练）
- GSM8K flex：mean 0.4939，std 0.0129，min 0.4754，max 0.5163，**极差 0.0409**（n=9）
- GSM8K strict：mean 0.4760，极差 0.0417（n=9）
- 判定：任何方法间 GSM8K 差异 ≤ 此极差者，不可主张方向性优劣（F8 噪声底）。

### 方差分解（flex）
- seed2025 内 3 repeat 极差 0.0023（非确定性分量）
- seed3407 内 3 repeat 极差 0.0303（非确定性分量）
- seed42 内 3 repeat 极差 0.0409（非确定性分量）
- 跨 seed 均值极差 0.0030（种子分量）

