# 14B 三种子 GSM8K-flex 判定 · 阈值校准修正案（Amendment）

状态：**已冻结（FROZEN）** — 必须在 Phase3-D 评测揭盲之前生效
冻结时间（UTC）：__见 git commit 时间戳__
关联预注册：docs/preregistration-14b-3seeds.md（commit 220fd60）
触发来源：docs/review/jcr-14b-verification.md §3 裁定(a)（JCR_Q1 评审）

---

## 0. 为什么需要本修正案（诚实披露）

原预注册（220fd60）判定规则用 **4.5 pt** 作为 gsm8k_flex 配对差的噪声门槛，来源是 8B 上 product/importance_only「同配置两次训练」的**单对**经验值。后续证据表明该门槛校准失效：

1. `docs/review/fair-baseline.md` F8：8B 同配置（product vs importance_only，同种子、128/128 张量 sha256 全不同）两次独立训练 GSM8K 差达 **11.52 pt**；另一对（error_only vs uniform）差 1.06 pt → 噪声**重尾**，单点 4.5 pt 不可用。
2. Phase-1 止血（commit b0d4dde）已把 `docs/claim-revision-norm-aware.md` §5 噪声底改为**区间 [0.9, 11.5] pt**。
3. 14B 实测 gsm8k_flex 配对差均值 **+8.87 pt 落在 [0.9, 11.5] 区间之内** → 按原 4.5 pt 门槛判「正向」、按项目自己修正后的区间判「不能排除噪声」，**同一数据两个合法口径给出相反结论 = 门槛校准失效**，不能据原门槛下「正向」结论。
4. 14B 配对差跨种子高度不均：seed42 +13.34 / seed2025 +9.02 / seed3407 **+4.25**（自身不过 4.5 线），均值由 seed42 单种子扛；跨种子 std≈4.5 pt。

因此原门槛 4.5 pt 作废，改用 **Phase3-D 实测 pooled 噪声底** 重判。

## 1. 冻结声明（防 HARKing）

- 本修正案在 **Phase3-D 评测结果揭盲之前**冻结（D 当前状态：训练 6/9 rc=0、s2025 进行中、**评测未跑、summary.json 全 null**，见 jcr-14b-verification.md §3）。
- 14B 原始数据（`results_v2_qwen14b_seed{42,2025,3407}/{adaq,qlora}/`）**已固定、不可变**；本修正案**只改校准参数（噪声底）**，不改 14B 数据、不改判定方向、不挑数据。
- 「只改校准参数 + 先于新数据（D 噪声底）冻结」= 合法的**阈值校准（threshold calibration）**，非事后假设（HARKing）。D 揭盲后不得再修改本规则。

## 2. 预指定的噪声底估计量（来自 Phase3-D 的 9 个 gsm8k_flex 值）

Phase3-D = 同一 product 配置（ranks_product.json + per_module_type + α=2r 修复）独立训练 9 次 = 3 seeds {42,3407,2025} × 3 repeats {r0,r1,r2}，评测 gsm8k（flex）。设 9 个 flex 值为 v(seed, rep)。

- 每种子均值 m_s = mean over rep of v(s, rep)，s ∈ {42,3407,2025}
- **N_seed**（种子分量）= max(m_s) − min(m_s)
- **N_rep**（同种子非确定性分量）= max over s of [ max_rep v(s,rep) − min_rep v(s,rep) ]
- **噪声底 N = max(N_seed, N_rep)**（保守取大）

> N 完全由 D 数据决定，本修正案冻结时 D 评测尚未揭盲，故 N 对撰写者是「未来数据」，不构成事后选择。

## 3. 预指定的 14B 重判规则（D 揭盲后机械套用，不得再改）

设 Δ_flex = 14B gsm8k_flex 配对差均值（**已固定 = +8.87 pt**，per-seed +13.34/+9.02/+4.25）。

- **R1（正向）**：若 **Δ_flex > N** 且 **3/3 种子配对差同号（全正）** → 判定「14B 上 AdaQ 相对 QLoRA 在 GSM8K-flex 观察到正向差异（均值 +8.9 pt）」。
  - 强制附带条件：此结论 **pending B5（chat-template GSM8K 复评）排除格式漂移 artifact**；B5 出数前，该差值**不得进入 abstract / intro / conclusion / 图表标题**（仅可入附录 + 明确标注未排除格式混杂）。
  - 措辞红线：禁写「优于 / 等价 / 打平 / 显著」；只能写「观察到正向差异，待 chat-template 口径确认」。
- **R2（不确定）**：若 **Δ_flex ≤ N** 或 三种子符号不一致 → 判定「**未观察到超出实测噪声底的一致差异**」。
- **R3（MMLU，独立于 D）**：Δ_mmlu = +0.05 pt ≈ 14042 题中 6 题 → 判定「**no difference observed; underpowered to establish equivalence**」；禁写「等价 / 不可区分 / 打平」（TOST 未做、n=3 无 CI）。
- **R4（strict，永不用）**：GSM8K strict-match 对 Qwen2.5-14B-Instruct 不可用（输出格式不满足 `#### N`，度量格式而非能力），**一律不作为判据**，仅可附录列示并标注失效原因。
- **R5（参数口径，强制同报）**：任何 14B 比较必须同时报 adapter 参数量（adaq 4,620,288 vs qlora 6,291,456）与秩预算（adaq sum-of-ranks 528 vs qlora 768），并声明**非预算对齐**（F14）；「以少 26.5% 参数取得 MMLU 未劣化」可作描述性参数效率，禁作因果「更优」。

## 4. 本修正案不解决的问题（仍需独立阻断项，见 jcr-14b-verification.md §6）

- **B5（P0）**：chat-template GSM8K 14B 复评——排除 R1 的格式漂移替代解释（最强替代解释，纯评测 6×~70min）。
- **B6/B7（P0）**：α/r 混杂（F15，adaq v_proj α/r=4.0 vs qlora 2.0）+ 14B 非等预算（F14）——污染因果归因；(c) 零成本改述立即执行，重训排 P1。
- **B8（P0）**：flex 抽取器人工审计（400 条双标注 κ，precision≥0.95）——主判定仪器校准。
- **B11（P1）**：TOST + item-level bootstrap CI + 14B 本地噪声对（14B 自己的同配置重复，非 8B 外推）。

## 5. 时间线（揭盲顺序，证明本修正案先于 D 数据）

| 事件 | 时间（UTC） | 来源 |
|---|---|---|
| 原预注册规则冻结 | 2026-11-15 13:37（aggregate mtime）→ 16:14（commit 220fd60） | prereg §时间线 |
| 14B gsm8k seeds 2025/3407 揭盲 | 2026-11-15 18:28–22:30 | chain_b4_seeds.log |
| **本阈值校准修正案冻结** | **2026-11-16 ~00:xx（本 commit）** | git log |
| Phase3-D 评测揭盲（N 可得） | **晚于本修正案**（D 训练进行中、评测未跑） | logs/phase3/chain_phase3.log |

> 关键：本修正案冻结 < Phase3-D 评测揭盲。D 揭盲后填入 N 并机械套用 §3，不得回头改 §2/§3。

---
*本修正案为纯文档，未运行 GPU 任务，未改任何共享代码/结果文件。*
