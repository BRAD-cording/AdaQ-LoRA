Dear Editor-in-Chief,

We submit for your consideration the manuscript

  "Error-Scaled Rank Allocation Collapses to Norm-Proportional Allocation in
   4–8-Bit LoRA: A Cross-Quantizer Audit of One Instantiation"

by Yibo Zhang, Muhammad Syafiq bin Mohd Pozi and Osman B Ghazali, for publication
in *Neurocomputing* as a research article.

**What the paper reports.** A family of adaptive-rank methods allocates LoRA
ranks from a per-layer signal. We audit one explicit instantiation — ranks
proportional to ε_l‖W_l‖_F, the product of per-layer quantization error and
weight norm — and report three findings:

1. *The score is an identity.* ε_l‖W_l‖_F equals the Frobenius norm of the
   quantization-error matrix exactly, so the "importance" factor cancels and the
   rule is a monotone transform of error magnitude. This is one line of algebra
   and can be checked before any training run.

2. *At 4 bits the signal is flat.* Per-layer relative error varies by 1.49% (NF4)
   and 0.49% (INT4-RTN) across the 64 adapted modules, so the score tracks the
   weight norm at ρ = 0.998. The realised allocation accordingly differs from a
   norm-proportional one on 2 of 64 modules, and what it actually encodes is
   projection type, not layer.

3. *No measurable difference, with quantified bounds.* Seed-matched, paired
   audits across NF4, INT4-RTN and INT8 on MMLU and GSM8K give no significant
   cell after Benjamini–Hochberg control (q ≥ 0.337), with equivalence bounds of
   ±0.5 points on MMLU and ±2 points on GSM8K for all three quantizers. A
   maximum-contrast arm (q = 14 vs v = 2 ranks) moves nothing either, so the null
   is not a weak-intervention artefact.

**Why we believe it suits Neurocomputing.** The contribution is an empirical
audit with an explicit resolution, not a new method. We state the sample sizes
needed to settle effects below our bounds (about 14 seeds per GSM8K cell at a
one-point threshold), we report a positive control showing the same pipeline
resolves a 4.3-point quantizer main effect, and we release per-seed evaluation
records plus the scripts that regenerate every table. A reader can therefore
either reuse the diagnostic or challenge the bound directly, which we take to be
the useful form of a negative result.

**Declarations.** The manuscript is original, has not been published elsewhere,
and is not under consideration by another journal. All authors have approved the
submission and declare no competing interests. The work received no specific
funding. Code, per-seed records and the scripts that generate every table are
available at https://github.com/BRAD-cording/AdaQ-LoRA (public upon submission)
and from the corresponding author. The use of a large language model for language
editing and analysis-script assistance is disclosed in the manuscript; all
numerical results were checked against the raw per-seed records by the authors,
who take full responsibility for the content.

Corresponding author:
  Yibo Zhang
  School of Computing, Universiti Utara Malaysia
  zhang_yibo@ahsgs.uum.edu.my

Yours sincerely,
Yibo Zhang, on behalf of all authors
