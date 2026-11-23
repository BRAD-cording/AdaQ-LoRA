# Submission checklist — Neurocomputing

## Ready in this repository

| Item | File | State |
|---|---|---|
| Manuscript (PDF, 14 pp, elsarticle `preprint` 12pt) | `paper/neurocomputing_paper.pdf` | built, 0 overfull, 0 undefined refs |
| Manuscript source | `paper/neurocomputing_paper.tex` | 288 lines |
| Bibliography | `paper/references.bib` | 36 entries, all cited, 0 missing, 0 chimeric |
| Highlights | `paper/highlights.txt` | 5 bullets, longest 77 chars (limit 85) |
| Cover letter | `paper/cover-letter.md` | 1 page |
| Figures | `paper/audit_framework.png`, `paper/epsilon_cross_quantizer.png` | both programmatic |
| Table 1 data | `paper/rtn_b16_gsm8k.csv`, `paper/rtn_b16_mmlu.csv`, `paper/shot5_gsm8k.csv`, `paper/nf4_gsm8k_n10.csv`, `paper/extreme_gsm8k.csv` | per-seed records |
| Table regeneration | `paper/recompute_stats.py`, `paper/calc_shot5.py`, `paper/analyze_b16.py` | pure stdlib |
| Citation audit | `paper/citation-audit.md` | 36/36 verified |

Declarations present in the manuscript: CRediT (two supervisors, Supervision
only), competing interests, funding (none), generative-AI use, data availability
(repository URL, verified reachable).

## Needs you (submission-system side)

- [ ] ORCID for each of the three authors
- [ ] Affiliation and e-mail confirmed in the system
- [ ] Paste the five Highlights into the system field (`paper/highlights.txt`)
- [ ] Upload the cover letter (`paper/cover-letter.md`)
- [ ] Suggested reviewers — the journal may require them; none are proposed here
      because inventing reviewer names is not acceptable
- [ ] Keep the repository public while the paper is under review

## Limits already stated in Threats

- one model (Llama-3-8B), one data recipe (OpenHermes-2.5, first 2000 samples),
  63 optimiser steps, two benchmarks
- the three MMLU cells use three to five seeds (CI ≤ ±0.35 pp)
- INT8 uses one scale per weight matrix, not per channel; a per-channel rerun
  would be the cleaner comparator and was not performed
