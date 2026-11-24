# Submission checklist — Neurocomputing

## Ready in this repository

| Item | File | State |
|---|---|---|
| Manuscript (PDF, 14 pp, elsarticle `preprint` 12pt) | `paper/neurocomputing_paper.pdf` | built, 0 overfull, 0 undefined refs |
| Manuscript source | `paper/neurocomputing_paper.tex` | 289 lines |
| Bibliography | `paper/references.bib` | 36 entries, all cited, 0 missing, 0 chimeric |
| Highlights | `paper/highlights.txt` | 5 bullets, longest 77 chars (limit 85) |
| Cover letter | `paper/cover-letter.md` | 1 page |
| Figures | `paper/audit_framework.png`, `paper/epsilon_cross_quantizer.png` | both programmatic |
| **Table 1 generator** | `paper/final_table.py` | **single source of truth**; recomputes all six cells from per-seed files |
| Per-seed data | `rtn_b16_gsm8k.csv`, `rtn_b16_mmlu.csv`, `shot5_gsm8k.csv`, `nf4_gsm8k_n10.csv`, `extreme_gsm8k.csv`, `bf16_lora_control.csv` | one file per arm |
| Citation audit | `paper/citation-audit.md` | 36/36 verified |
| Numeric cross-check | `paper/check_numbers.py` | prose intervals vs Table 1, 0 mismatches |

Declarations in the manuscript: CRediT (two supervisors, Supervision only),
competing interests, funding (none), generative-AI use, data availability
(repository URL, verified reachable).

## Revisions made after the nine-expert review

| Issue | Resolution |
|---|---|
| q column was not valid BH (INT8 rows held *p* values) | Recomputed: min q = 0.401; text says `q >= 0.40` |
| NF4/MMLU CI conflicted with its own *d* | Replaced with source values [−0.16, +0.02], p = 0.073 |
| INT8 evidence CSV held pre-retrain scores | CSV updated; `final_table.py` reproduces the table verbatim |
| "Proposition" was a definitional tautology | Demoted to Observation, explicitly "an identity, not a theorem" |
| Missing bf16+LoRA control | Added, 3 seeds; §5.3 rewritten (see below) |
| AQLoRA described opposite to its abstract | Reworded, AQLoRA's negative control now stated |
| DoRA "30% more parameters" | Corrected to 4.8% (computed) |
| "64 query projections", CV 1.48/0.48, "opposite direction" | All corrected |
| Post-hoc pooling absent | −0.45 pp (Q = 0.83) added to Threats, marked exploratory |

## The bf16+LoRA control changed a claim

| Arm | MMLU | Δ vs bf16 base |
|---|---|---|
| bf16 base | 0.6536 | — |
| bf16 + LoRA | 0.6472 | −0.64 pp |
| NF4 + LoRA | 0.6533 | −0.03 pp |
| INT4-RTN + LoRA | 0.6092 | −4.44 pp |

The manuscript previously claimed MMLU "tracks quantisation damage rather than
adapter quality". With the missing arm measured, adaptation alone moves MMLU by
0.64 pp — above the 0.11-point detection floor. The claim was rewritten: the
benchmark is not blind, but allocation differences (0.05–0.09 pp) sit an order
of magnitude below the adaptation effect, which is what its rows cannot resolve.

## Limits stated in Threats

- one model (Llama-3-8B), one data recipe (OpenHermes-2.5, first 2000 samples),
  63 optimiser steps, two benchmarks
- 63 steps is explicitly recorded as *unexamined*, not as a null: settling the
  capacity question at comparable resolution needs about ten seeds per arm
- the dose-response arm (q = 14, v = 2) has n = 3 and CI [−5.7, +5.6], so it
  excludes only effects larger than five points
- INT8 uses one scale per weight matrix, not per channel; a per-channel rerun
  was not performed

## Needs you (submission-system side)

- [ ] ORCID for each of the three authors
- [ ] Affiliation and e-mail confirmed in the system
- [ ] Paste the five Highlights (`paper/highlights.txt`)
- [ ] Upload the cover letter (`paper/cover-letter.md`)
- [ ] Suggested reviewers — none proposed here; inventing names is not acceptable
- [ ] Keep the repository public during review
