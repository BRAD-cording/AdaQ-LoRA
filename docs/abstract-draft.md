# Revised Abstract — AdaQ-LoRA (Q2 submission draft)
# Generated: 2026-11-19

Error-aware rank allocation—assigning more LoRA adapter capacity to layers with larger
quantization error—is a common motif in quantized fine-tuning (AdaLoRA, RA-LoRA, AQLoRA).
We show that under 4-bit LLM quantization this strategy is degenerate: the per-layer
quantization error eps_l = ||E_l||_F / ||W_l||_F is nearly constant across layers
(NF4: CV = 1.49%, max/min = 1.088; INT4-RTN-g128: CV = 0.49%, max/min = 1.026).
Consequently, the composite score used for rank allocation collapses to a single degree
of freedom: score ≡ ||E_l||_F ∝ ||W_l||_F, making "error-aware" rank allocation
indistinguishable from "norm-aware" rank allocation.

We test this prediction across four independent streams: equal-budget main table,
alpha=2r clean retraining, six-strategy ablation, and cost-normalized correction—all
converge on null results (no consistent accuracy gain over uniform allocation). A
cross-quantizer experiment on INT4-RTN-g128 reveals the degeneracy is NOT specific to
NF4's block-wise absmax scaling but is a broader empirical regularity of 4-bit LLM
quantization (rho(eps_rtn, eps_nf4) = 0.887, p < 10^-21).

Power analysis (TOST, MDE = 12.7pt at n=3) shows the experiment is underpowered to
establish formal equivalence, and we enumerate the seed budget required for definitive
exclusion (n >= 10 at delta = 5pt). We conclude that error-aware rank allocation offers
no reliable benefit over uniform allocation under practical 4-bit quantized LoRA
fine-tuning, and we identify the structural conditions—group-wise symmetric quantization,
near-Gaussian weight distributions, and near-optimal codebooks—under which this
degeneracy is expected to hold.