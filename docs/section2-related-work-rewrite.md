# Section 2 — Related Work (revised, with RA-LoRA / RILQ / Variable-Rank)
# Generated: 2026-11-19

## 2.x Error-Aware Rank Allocation for Quantized LLMs

AdaLoRA (Zhang et al., ICLR 2023) pioneered importance-based rank allocation via
gradient sensitivity, but did not consider quantization as the importance signal.
RA-LoRA (Kim et al., ACL 2024) extended rank adaptation to 2-bit quantized LLMs
using rank-subspace analysis, demonstrating that quantization error structure should
inform rank allocation. RILQ (Lee et al., AAAI 2025) introduced rank sensitivity
analysis for 2-bit LQEC, finding that lower bit precision requires higher ranks for
effective error compensation. AQLoRA (Islam, arXiv 2026) independently observed that
NF4 quantization error varies little across layers, using this observation for a
freezing-based speedup strategy rather than rank allocation.

None of these works ask whether the error signal itself carries sufficient cross-layer
variation to drive meaningful rank differentiation. We show that under practical 4-bit
quantization (NF4 and INT4-RTN), the per-layer quantization error eps_l is nearly
constant (CV < 1.5%), causing "error-aware" rank allocation to degenerate to
"norm-aware" rank allocation—a finding that applies to the shared mechanism underlying
AdaLoRA, RA-LoRA, and AQLoRA alike. The recently published Variable-Rank Adaptation
survey (Ye et al., EMNLP 2026) does not identify this degeneracy, highlighting the
gap our work fills.

## Prior work comparison table

| Work | Venue | Mechanism | Quantization | Our relationship |
|---|---|---|---|---|
| AdaLoRA | ICLR 2023 | Gradient+importance→rank | None (FP16) | Shares norm-aware basis; we prove degeneracy under NF4 |
| RA-LoRA | ACL 2024 Findings | Rank-subspace→dynamic rank | 2-bit | Most direct prior; we extend to 4-bit and reveal degeneracy |
| RILQ | AAAI 2025 | Rank-sensitivity→LQEC | 2-bit | Complementary: provides rank sensitivity analysis framework |
| AQLoRA | arXiv 2026.08 | Error-sort→freezing | NF4 | Shares degeneracy observation; different downstream goal |
| Variable-Rank Survey | EMNLP 2026 Findings | Taxonomy of all rank-budget methods | Various | Positions our work as "degeneracy audit" of a shared assumption |