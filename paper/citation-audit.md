# Citation audit — references.bib

Scope: all 36 entries cited in `neurocomputing_paper.tex`; the `.bib` holds no
uncited entries.

## Method

Three sources with different authority, verdicts kept separate:

| Source | Role | Verdict |
|---|---|---|
| Crossref `/works/{doi}` | registration agency | `CONFIRMED` when title/author/year agree |
| `export.arxiv.org/api` | independent field source for preprints | `CONFIRMED` |
| OpenAlex title search | index only, no registration authority | `INDEX_ONLY` |

An entry with no identifier at all (ICLR/ICML/NeurIPS papers have no DOI) is
verified by index match only; those venues never issue DOIs, so absence of a
registration record is not an error.

## Result

| Verdict | n | Meaning |
|---|---|---|
| `CONFIRMED` (Crossref or arXiv) | 16 | registered metadata agrees with the citation |
| `INDEX_ONLY` (OpenAlex) | 11 | conference paper without DOI; title/authors/year match in an index |
| year differs from index | 8 | conference year vs arXiv-preprint year — see below |
| dataset record | 1 | not indexed as a work; verified through the Hugging Face API instead |
| **confirmed missing** | **0** | — |
| **chimeric** | **0** | — |

## Corrections made during this audit

1. **`ploner2026variablerank` DOI restored.** An earlier round "harmonised" the
   DOI to `10.18653/v1/2026.findings-emnlp.2430` for consistency with other ACL
   entries. Crossref returns **404** for that form and **200** for
   `10.18653/2026.findings-emnlp.2430`. The registered record matches the
   citation exactly (Ploner & Akbik; Findings of ACL: EMNLP 2026; pp.
   52150–52172). Formatting consistency is not evidence — the registration is.
2. **`teknium2023openhermes` title.** Replaced a descriptive title ("OpenHermes-2.5
   instruction dataset") with the dataset's own name. `createdAt = 2023-11-12`
   from the Hugging Face API supports the year.
3. **RILQ description in §2.** "rank-insensitive branch" overstated the source;
   the arXiv abstract describes a *rank-insensitive loss* applied across layers.
   Reworded.

## The eight year differences are not errors

These entries cite the **conference** year while the index records the earlier
**arXiv** year:

| Entry | bib | index | basis |
|---|---|---|---|
| hu2022lora | 2022 | 2021 | ICLR 2022 |
| he2022unified | 2022 | 2021 | ICLR 2022 |
| frantar2023gptq | 2023 | 2022 | ICLR 2023 |
| lin2024awq | 2024 | 2023 | MLSys 2024 |
| xiao2023smoothquant | 2023 | 2022 | ICML 2023 |
| hendrycks2021mmlu | 2021 | 2020 | ICLR 2021 |
| simmons2011falsepositive | 2011 | — | Crossref stores the subtitle separately; the full title is correct |
| cohen1988 | 1988 | 1989 | second edition, Lawrence Erlbaum 1988; the 1989 index hit is a book review |

## Claim–citation support

Every `\citep` context was read against the cited work. The three closest prior
works were checked against their abstracts:

- RA-LoRA — "allocates rank for 2-bit models" (matches the title and abstract).
- RILQ — corrected to "rank-insensitive loss applied across layers".
- AQLoRA — the negative control result quoted in §2 ("error-based layer
  selection no better than random on five models") is verbatim in its abstract.

No citation is used as evidence for a proposition the source does not state.

## Residual limits

- `INDEX_ONLY` verdicts rest on an index rather than a registration agency; for
  venues without DOIs that is the strongest available evidence.
- This audit checks metadata identity, not whether a claim is scientifically
  correct; semantic support was assessed by reading, not by a locator-level
  quote table.
- No retraction or correction notice was found for any cited work.
