# Legal Journal Agent

## Recommended Workflow

Install first:

```bash
pip install -e .
```

1. Build the SSCI law-journal dataset from the law list.
   This step enriches each journal with `Aims & Scope`, recent articles, and a one-year publication count from the journal site or OpenAlex.

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_law_list.json ^
  --output data/legal_journals.law_list.full.json
```

2. Train the supervised model with a real `train / validation / test` split.

```bash
python -m journal_agent train-supervised ^
  --dataset data/legal_journals.law_list.full.json ^
  --model-output output/supervised_journal_model.pkl ^
  --report output/supervised_training_report.json
```

3. Run recommendation with the trained model.

```bash
python -m journal_agent recommend ^
  --manuscript data/sample_manuscript.txt ^
  --dataset data/legal_journals.law_list.full.json ^
  --model output/supervised_journal_model.pkl ^
  --output output/recommendations.csv ^
  --top-k 10
```

4. Or provide title / abstract / keywords directly.

```bash
python -m journal_agent recommend ^
  --title "Platform governance and algorithmic due process" ^
  --abstract "This article studies due process obligations in platform governance..." ^
  --keywords "platform governance; algorithm; due process; digital regulation" ^
  --dataset data/legal_journals.law_list.full.json ^
  --model output/supervised_journal_model.pkl ^
  --output output/recommendations.csv
```

## Current Model

- Candidate pool: `156` SSCI law journals
- Journal portrait factors: `Aims & Scope`, recent `title / abstract / keywords`, one-year publication count
- Learning setup: supervised ranking with real `train / validation / test` split
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

Generic SSCI crawl:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci.json ^
  --output data/legal_journals.collected.json
```

Quick metadata-only build:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_quick.json ^
  --output data/legal_journals.quick.json
```

## Notes

- `data/source_manifest.ssci_law_list.json` is the default law-list manifest.
- Recommendation language follows manuscript language.
- `legal_journals.seed.json` is only for demo/testing.
