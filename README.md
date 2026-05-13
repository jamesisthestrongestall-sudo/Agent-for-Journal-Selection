# Legal Journal Agent

Legal Journal Agent is an AI-assisted journal recommendation pipeline for legal manuscripts. It builds supervised ranking models from SSCI-indexed law and law-adjacent social-science journals, crawls recent article metadata from OpenAlex, and recommends venues by matching a manuscript's title, abstract, keywords, methodology, and subfield signals against journal profiles.

The current workflow expands beyond a narrow law-only list into an interdisciplinary SSCI pool covering law, criminology, political science, public administration, international relations, ethics, social issues, and interdisciplinary social sciences. The bundled model now uses a semantic-calibrated reranker so scope, recent article profile, subfield fit, and language compatibility are trusted more than the overfit logistic probability from the small five-article training sample.

## Recommended Workflow

Install first:

```bash
pip install -e .
```

1. Build the focused SSCI law-interdisciplinary dataset.
   This step collects SSCI law and closely law-adjacent social-science journals, then enriches each journal with the five latest OpenAlex article records.

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_law_interdisciplinary_articles5.json ^
  --output data/legal_journals.ssci_law_interdisciplinary_articles5.json
```

2. Train the supervised model with a real `train / validation / test` split.
   For this larger pool, sampled hard negatives keep training practical and improve close-call ranking accuracy.

```bash
python -m journal_agent train-supervised ^
  --dataset data/legal_journals.ssci_law_interdisciplinary_articles5.json ^
  --max-articles-per-journal 5 ^
  --regularization-grid 0.5 ^
  --negative-samples-per-query 80 ^
  --hard-negative-samples-per-query 40 ^
  --model-output output/supervised_journal_model.ssci_law_interdisciplinary_articles5.clean_hard.pkl ^
  --report output/supervised_training_report.ssci_law_interdisciplinary_articles5.clean_hard.json
```

3. Run recommendation with the trained model.

```bash
python -m journal_agent recommend ^
  --manuscript data/sample_manuscript.txt ^
  --dataset data/legal_journals.ssci_law_interdisciplinary_articles5.json ^
  --model output/supervised_journal_model.ssci_law_interdisciplinary_articles5.clean_hard.pkl ^
  --output output/recommendations.csv ^
  --top-k 10
```

For clearly legal manuscripts where you want a stricter law-journal candidate pool, add:

```bash
  --candidate-scope law-only
```

For a Scopus-indexed law-journal pool, download the Scopus Source List XLSX and add:

```bash
  --candidate-scope scopus-law ^
  --scopus-source-list data/scopus_source_list_mar_2026.xlsx
```

4. Or provide title / abstract / keywords directly.

```bash
python -m journal_agent recommend ^
  --title "Platform governance and algorithmic due process" ^
  --abstract "This article studies due process obligations in platform governance..." ^
  --keywords "platform governance; algorithm; due process; digital regulation" ^
  --dataset data/legal_journals.ssci_law_interdisciplinary_articles5.json ^
  --model output/supervised_journal_model.ssci_law_interdisciplinary_articles5.clean_hard.pkl ^
  --output output/recommendations.csv
```

## Current Model

- Candidate pool: `644` SSCI law and law-adjacent interdisciplinary social-science journals
- Journal portrait factors: recent `title / abstract / keywords`, SSCI category metadata, and generated scope text
- Learning setup: supervised ranking with non-research article filtering, sampled negatives, hard negatives, and real `train / validation / test` split
- Recommendation policy: when using the interdisciplinary dataset, the recommender ensures at least one interdisciplinary journal appears in the top 5 if an eligible candidate exists
- Recommendation output labels each journal as `interdisciplinary` or `single-field` and exports `subdisciplines` plus `is_interdisciplinary` in the CSV
- Venue-fit safeguards penalize Asia-Pacific journals when the manuscript lacks regional/comparative focus, and peace/conflict journals when the manuscript lacks peace/conflict/security focus
- Optional candidate scoping: `--candidate-scope law-only` narrows recommendations to journals with explicit law/legal/criminal-justice signals; `--candidate-scope scopus-law` narrows to active Scopus ASJC `3308 Law` journals when a Scopus Source List XLSX is available
- Internal exact-source test accuracy after semantic calibration: Top-1 `19.9%`, Top-3 `29.5%`, Top-5 `34.5%`, MRR `0.269`
- External Scopus/OpenAlex validation on 200 fresh recent articles: Top-1 `11.5%`, Top-3 `19.0%`, Top-5 `24.5%`, Top-10 `28.5%`, MRR `0.177`
- External semantic validity on the same Scopus/OpenAlex sample: Top-5 category overlap `88.0%`, Top-5 focus overlap `78.0%`
- Focused Scopus ASJC Law validation on 200 fresh recent law articles: broad scope Top-5 `20.5%`; law-only candidate scope Top-5 `23.0%`
- Fresh 50-paper Scopus ASJC Law validation: broad scope Top-5 `22.0%`; law-only scope Top-5 `30.0%`; Scopus-law scope Top-5 `36.0%`
- Manuscript input: `docx`, `txt`, `md`, or direct title / abstract / keywords

## Other Commands

Rule-based benchmark:

```bash
python -m journal_agent evaluate ^
  --dataset data/legal_journals.law_list.full.json ^
  --taxonomy data/law_taxonomy.json ^
  --report output/evaluation_report.json ^
  --details-output output/evaluation_details.csv
```

Expanded SSCI crawl:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci.json ^
  --output data/legal_journals.collected.json
```

Quick metadata-only build for the expanded law-adjacent SSCI pool:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_quick.json ^
  --output data/legal_journals.quick.json
```

## Notes

- `data/source_manifest.ssci_law_list.json` is the default manifest for the expanded law recommendation pool.
- Recommendation language follows manuscript language.
- `legal_journals.seed.json` is only for demo/testing.
