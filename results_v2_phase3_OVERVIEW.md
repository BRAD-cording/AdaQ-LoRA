# Phase-3 总览（自动拼接）  Tue Nov 17 03:11:59 PM UTC 2026

-----
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


-----
# Phase-3 B 聚合报告

结果目录：`results_v2_phase3_main`　run 数：3　组数：1

| 组 | MMLU acc | GSM8K strict | GSM8K flex |
|---|---|---|---|
| adalora | 0.6529 ±0.0005 (n=3, range 0.0009) | 0.5087 ±0.0066 (n=3, range 0.0114) | 0.5123 ±0.0097 (n=3, range 0.0182) |



-----
# Phase-3 E 聚合报告

结果目录：`results_v2_phase3_baselines`　run 数：6　组数：2

| 组 | MMLU acc | GSM8K strict | GSM8K flex |
|---|---|---|---|
| loftq | 0.6409 ±0.0019 (n=3, range 0.0039) | 0.4665 ±0.0205 (n=3, range 0.0387) | 0.4890 ±0.0133 (n=3, range 0.0265) |
| qlorapm | 0.6401 ±0.0012 (n=3, range 0.0022) | 0.4741 ±0.0016 (n=3, range 0.0030) | 0.4958 ±0.0020 (n=3, range 0.0038) |



-----
# Phase-3 C 聚合报告

结果目录：`results_v2_phase3_rcurve`　run 数：12　组数：6

| 组 | MMLU acc | GSM8K strict | GSM8K flex |
|---|---|---|---|
| adaq_R1024 | 0.6419 ±0.0003 (n=2, range 0.0004) | 0.4189 ±0.0306 (n=2, range 0.0432) | 0.4537 ±0.0166 (n=2, range 0.0235) |
| adaq_R128 | 0.6469 ±0.0014 (n=2, range 0.0020) | 0.5159 ±0.0048 (n=2, range 0.0068) | 0.5205 ±0.0048 (n=2, range 0.0068) |
| adaq_R512 | 0.6437 ±0.0004 (n=2, range 0.0006) | 0.4754 ±0.0075 (n=2, range 0.0106) | 0.4924 ±0.0027 (n=2, range 0.0038) |
| qlora_R1024 | 0.6369 ±0.0003 (n=2, range 0.0004) | 0.4621 ±0.0059 (n=2, range 0.0083) | 0.4901 ±0.0080 (n=2, range 0.0114) |
| qlora_R128 | 0.6446 ±0.0020 (n=2, range 0.0029) | 0.5144 ±0.0080 (n=2, range 0.0114) | 0.5201 ±0.0118 (n=2, range 0.0167) |
| qlora_R512 | 0.6431 ±0.0019 (n=2, range 0.0027) | 0.4674 ±0.0134 (n=2, range 0.0190) | 0.4928 ±0.0021 (n=2, range 0.0030) |



-----
# Phase-3 A 聚合报告

结果目录：`results_v2_phase3_main`　run 数：12　组数：4

| 组 | MMLU acc | GSM8K strict | GSM8K flex |
|---|---|---|---|
| adalora | 0.6529 ±0.0005 (n=3, range 0.0009) | 0.5087 ±0.0066 (n=3, range 0.0114) | 0.5123 ±0.0097 (n=3, range 0.0182) |
| adaq | 0.6429 ±0.0026 (n=3, range 0.0051) | 0.4748 ±0.0138 (n=3, range 0.0273) | 0.4928 ±0.0166 (n=3, range 0.0311) |
| dora | 0.6412 ±0.0012 (n=3, range 0.0021) | 0.4819 ±0.0070 (n=3, range 0.0137) | 0.4963 ±0.0039 (n=3, range 0.0076) |
| qlora | 0.6400 ±0.0027 (n=3, range 0.0053) | 0.4845 ±0.0194 (n=3, range 0.0387) | 0.4994 ±0.0131 (n=3, range 0.0250) |


