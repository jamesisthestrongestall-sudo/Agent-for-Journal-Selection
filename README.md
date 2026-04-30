# Legal Journal Agent

Legal Journal Agent is an AI-assisted journal recommendation pipeline for legal manuscripts. It builds supervised ranking models from SSCI-indexed law and law-adjacent social-science journals, crawls recent article metadata from OpenAlex, and recommends venues by matching a manuscript's title, abstract, keywords, methodology, and subfield signals against journal profiles.

The current workflow expands beyond a narrow law-only list into an interdisciplinary SSCI pool covering law, criminology, political science, public administration, international relations, ethics, social issues, and interdisciplinary social sciences. The best bundled model uses cleaned article samples and hard-negative training to improve close-call ranking accuracy across the expanded candidate pool.

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
- Latest test accuracy: Top-1 `85.7%`, Top-3 `95.0%`, Top-5 `97.5%`, MRR `0.908`
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
