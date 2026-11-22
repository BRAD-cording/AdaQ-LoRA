# Section 3.2 Replacement — Mechanism (three-layer decomposition)
# Replaces: "Lemma 1 (Degeneration Lemma)" and "error × importance" framing
# Generated: 2026-11-19

## 3.2 When Does Error-Aware Degenerate to Norm-Aware?

### Observation 1 (Scale invariance, exact)
For any per-block or per-tensor quantizer with a shared per-group scaling factor
c = max|w_group|, the relative quantization error eps_l = ||Q(W_l) - W_l||_F / ||W_l||_F
is scale-invariant: multiplying W_l by any scalar leaves eps_l unchanged.
Proof: the scaling factor c absorbs the multiplier, and both numerator and denominator
of eps_l scale identically.

### Empirical regularity (eps_l ≈ constant)
For Meta-Llama-3-8B quantized with NF4 (block-64, asymmetric optimal codebook), eps_l
ranges from 0.0926 to 0.1007 (CV = 1.49%, n = 64 layers across q_proj and v_proj).
For the same model quantized with INT4-RTN (symmetric group-128, round-to-nearest),
eps_l is even flatter: 0.0896 to 0.0919 (CV = 0.49%). Across both quantizers, eps_l
explains < 0.05% of the log-score variance beyond what ||W_l||_F already explains
(R^2 delta_eps <= 0.0005). The cross-quantizer Spearman correlation is rho = 0.887
(p < 10^-21), indicating that the relative ordering of layer-wise errors is highly
preserved across quantization schemes.

### Proposed mechanism
The near-constancy of eps_l is attributed to three jointly sufficient conditions:
(i) per-block or per-group absmax scaling, which guarantees scale invariance;
(ii) approximately Gaussian weight distributions within each group, making the
normalized quantization noise distribution similar across layers;
(iii) near-optimal quantization codebooks (NF4's information-theoretically optimal
levels, or INT4's uniform symmetric grid) that make the per-group quantization error
depend primarily on the group's shape (≈ Gaussian) rather than its scale.

We emphasize that condition (i) is exact while (ii) and (iii) are empirical: we
observe known violations in layers with outlier channels (see Limitations), and
flatness degrades at larger model scale (14B max/min = 1.329 vs 8B 1.088 for NF4).

### Corollary (Rank allocation degeneracy)
Since score_l = eps_l · ||W_l||_F = ||E_l||_F (an algebraic identity, not a product
of two independent signals), and eps_l ≈ constant, it follows that ranking layers by
score_l is equivalent to ranking them by ||W_l||_F: Spearman(score, ||W||) = 0.998
(NF4) and 0.998 (INT4-RTN). "Error-aware" rank allocation is therefore indistinguishable
from "norm-aware" rank allocation under practical 4-bit LLM quantization.