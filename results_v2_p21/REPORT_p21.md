# P2-1 成本归一化秩分配 — 预注册判定报告

生成：2026-11-18 23:09:08 UTC  
预注册：docs/preregistration-p21-costnorm.md（4107552 + 修正案 448330a）

## 噪声底

N = **4.094 pt**（来源：Phase3-D 实测；Phase3-D 9run gsm8k-flex 极差，n=9）

## 各臂 GSM8K-flex（%，3 seed）

| 臂 | 参数 | mean | std | min | max | 各种子值 |
|---|---|---|---|---|---|---|
| cost_norm | 3,693,568 | 50.1643 | 0.845 | 48.9765 | 50.8719 | [48.9765, 50.6444, 50.8719] |
| raw_score | 3,782,656 | 50.4675 | 0.8849 | 49.2798 | 51.4026 | [50.7202, 51.4026, 49.2798] |
| uniform | 3,407,872 | 49.5072 | 0.1637 | 49.3556 | 49.7346 | [49.3556, 49.4314, 49.7346] |
| uniform_pm | 3,833,856 | 49.3556 | 0.9286 | 48.2183 | 50.4928 | [50.4928, 48.2183, 49.3556] |

## 配对差（gsm8k-flex，pt，逐 seed）

- **dPM = cost_norm − uniform_pm（R1 主判据）**：[-1.5163, 2.4261, 1.5163]  同号 2/3  均值 0.8087
- dU = cost_norm − uniform（R1b 等秩和）：[-0.3791, 1.213, 1.1373]  同号 2/3  均值 0.6571
- dR = cost_norm − raw_score（R3 成本归一化 vs 原始分）：[-1.7437, -0.7582, 1.5921]  同号 1/3  均值 -0.3033

参数对照：cost_norm 3,693,568 vs uniform_pm 3,833,856 → cost_norm 更省参数 = **True**

## 判定（机械套用预注册，只填数不改规则）

- R1（3/3 dPM>0 且 mean>4.094 且 cost_norm 更省参数）= **False**（sign False / mag False / cheaper True）
- R1b（等秩和 3/3 dU>0 且 mean>4.094）= **False**

### 结论

**R2（阴性）：cost_norm 未稳定超过 uniform_pm（参数匹配）→ 维持「机制+阴性+功效边界」四件套，退 TMLR/Neural Networks/Neurocomputing**


## 强制披露（预注册 R5 + 评审）

- n=3 种子欠功效；strict 永不作判据（R4，Qwen/Llama 格式）；MMLU 仅报告不判定。
- 主判据为 BF16 基座 gsm8k-flex；与 NF4 表不可横比。
- cost_norm 与 uniform_pm 参数差 3.7%（cn 更省）→ 对照排除参数量混杂；与 uniform 等秩和（512）对照隔离成本归一化本身。
- 若 R1 命中：先扩 6 seed（R1-confirm）再写正向，防 14B n=3 单种子扛均值覆辙。
