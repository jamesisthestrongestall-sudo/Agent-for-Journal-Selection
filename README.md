# Legal Journal Agent

## Recommended Workflow

Install first:

```bash
pip install -e .
```

1. Build an enriched journal dataset from the SSCI CSV:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci.json ^
  --output data/legal_journals.collected.json
```

2. Run recommendation with the generated dataset:

```bash
python -m journal_agent recommend ^
  --manuscript data/sample_manuscript.txt ^
  --dataset data/legal_journals.collected.json ^
  --taxonomy data/law_taxonomy.json ^
  --output output/recommendations.csv ^
  --top-k 10
```

3. Or provide title / abstract / keywords directly:

```bash
python -m journal_agent recommend ^
  --title "Platform governance and algorithmic due process" ^
  --abstract "This article studies due process obligations in platform governance..." ^
  --keywords "platform governance; algorithm; due process; digital regulation" ^
  --dataset data/legal_journals.collected.json ^
  --taxonomy data/law_taxonomy.json ^
  --output output/recommendations.csv
```

## Quick Mode

If you do not want live crawling, use the fast SSCI metadata-only build:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_quick.json ^
  --output data/legal_journals.quick.json
```

Then:

```bash
python -m journal_agent recommend ^
  --manuscript data/sample_manuscript.txt ^
  --dataset data/legal_journals.quick.json ^
  --taxonomy data/law_taxonomy.json ^
  --output output/recommendations.csv
```

## What It Does

- Reads the SSCI journal list from `data/Social Sciences Citation Index (SSCI).csv`
- Builds journal portraits
- In enriched mode, crawls real `Aims & Scope` and recent article metadata
- Parses manuscript text from `docx`, `txt`, `md`, or structured input
- Recommends journals and exports ranked results to CSV

## Notes

- `data/source_manifest.ssci.json` is the enriched mode.
- `data/source_manifest.ssci_quick.json` is the fast mode.
- Recommendation language follows manuscript language.
- `legal_journals.seed.json` is only for demo/testing, not the default formal recommendation source.
