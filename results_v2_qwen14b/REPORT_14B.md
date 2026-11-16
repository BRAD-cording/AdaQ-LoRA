# P1-B4 规模泛化：Qwen2.5-14B（3 种子）

模型 Qwen2.5-14B-Instruct；SFT `teknium/OpenHermes-2.5[:2000]`；评测 NF4（load_in_4bit + nf4 + double_quant，compute bf16）。

## 逐种子明细

| seed | method | adapter 参数 | MMLU | GSM8K strict | GSM8K flex |
| --- | --- | --- | --- | --- | --- |
| 42 | adaq | 4.62 M | 0.7843 | 0.1827 | 0.6588 |
| 42 | qlora | 6.29 M | 0.7837 | 0.1304 | 0.5254 |
| 2025 | adaq | 4.62 M | 0.7840 | 0.1903 | 0.5118 |
| 2025 | qlora | 6.29 M | 0.7836 | 0.1259 | 0.4215 |
| 3407 | adaq | 4.62 M | 0.7846 | 0.2699 | 0.5603 |
| 3407 | qlora | 6.29 M | 0.7841 | 0.2244 | 0.5178 |

## 均值 ± 标准差（n=3）

| method | MMLU | GSM8K strict | GSM8K flex |
| --- | --- | --- | --- |
| adaq | 0.7843 ± 0.0003 | 0.2143 ± 0.0483 | 0.5770 ± 0.0749 |
| qlora | 0.7838 ± 0.0003 | 0.1602 ± 0.0556 | 0.4882 ± 0.0579 |

## 配对差（同种子 adaq − qlora）

| metric | 42 | 2025 | 3407 | mean | 三种子同号? |
| --- | --- | --- | --- | --- | --- |
| mmlu | +0.0006 | +0.0004 | +0.0004 | +0.0005 | 是(全正) |
| gsm8k_strict | +0.0523 | +0.0644 | +0.0455 | +0.0541 | 是(全正) |
| gsm8k_flex | +0.1334 | +0.0902 | +0.0425 | +0.0887 | 是(全正) |

## 判定规则（先写死，再看数）

1. 若 **gsm8k_flex 的配对差三种子同号且均值 > 单次跑间噪声（4.5 点，来自 8B product/importance_only 同配置两次训练的经验值）**，才能写「14B 上 AdaQ 相对 QLoRA 有正向差异」。
2. 若三种子符号不一致，或均值落在 ±4.5 点内，只能写「未观察到一致差异」。
3. strict-match 一律不作为判据（Qwen-Instruct 格式不合规，strict 度量格式而非能力）。
4. adaq 参数比 qlora 少 26.5%，任何比较都必须同时报参数量。

---

## 重判（修正案 4414d9e · Phase3-D 揭盲后机械套用，2026-11-16）

> 上方「判定规则（先写死）」用的 4.5 pt 门槛已被修正案 4414d9e 作废。下列按修正案 §2/§3 机械套用（D 揭盲后只填 N、不改规则，防 HARKing）。

**Phase3-D 实测噪声底**（同 product 配置 9 次独立训练 GSM8K flex，results_v2_phase3_noise/summary.json）：
- 每种子均值 m_42=0.49583、m_3407=0.49280、m_2025=0.49305
- N_seed（跨种子均值极差）= **0.30 pt**
- 种子内 rep 极差：seed42=4.09 / seed3407=3.03 / seed2025=0.23 → N_rep（最大）= **4.09 pt**
- **噪声底 N = max(N_seed, N_rep) = 4.09 pt**

**R1 机械套用**：Δ_flex=+8.87 pt（per-seed +13.34/+9.02/+4.25，3/3 同号全正）
- Δ_flex (8.87) > N (4.09) ✓ 且 3/3 同号 ✓ → **R1 满足**

**判定（R1 措辞红线）**：
> 14B 上 AdaQ 相对 QLoRA 在 GSM8K-flex 观察到正向差异（均值 +8.9 pt），待 chat-template 口径确认。

**强制附带条件 + 保留（R1 + 修正案 §4）**：
1. **pending B5**：chat-template GSM8K 复评排除格式漂移 artifact 前，该差值**不得进入 abstract/intro/conclusion/图表标题**（仅附录 + 标注未排除格式混杂）。N=4.09<8.87 → 信号未死 → B5 已排队（用户拍板）。
2. **N 是 8B 代理**：噪声底测自 8B（Meta-Llama-3-8B）product 配置；8B→14B 外推未验证；待 B11 14B 本地噪声对核实。
3. **α/r 混杂（F15/B6）**：14B adaq adapter α/r=1.33(q)/4.0(v) vs qlora 2.0，因果归因被污染；待 α=2r 重训才是干净等-α 比较。
4. **seed42 拓扑/显存异常 + 主导信号**：seed42 peak VRAM 6.67GB（vs 新种子 17.6GB，F21 度量 bug 或拓扑差异），且 seed42 单种子扛 +13.34（均值主要由它驱动）；该 run 口径待核。
5. **重尾残余风险（诚实披露）**：pooled N=4.09 pt < 曾见单对极值 11.52 pt（fair-baseline F8，product vs importance_only 同配置）；噪声可能重尾，9 run 未必抓全尾部；B11 14B 本地噪声对进一步核实。
6. **R3（MMLU）**：Δ_mmlu +0.05 pt ≈ 6 题/14042 → "no difference observed; underpowered to establish equivalence"；禁写「等价/不可区分/打平」。
7. **R4（strict）**：永不作判据（Qwen-Instruct 格式不合规，度量格式非能力）。
8. **R5（参数口径）**：强制同报 adaq 4,620,288（sum-of-ranks 528）vs qlora 6,291,456（768）+ 非预算对齐声明；「参数少而 MMLU 未劣化」仅描述性，禁因果「更优」。
