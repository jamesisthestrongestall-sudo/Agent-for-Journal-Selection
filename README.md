# Legal Journal Agent

`Legal Journal Agent` is a Python journal recommendation agent scaffold focused on law journals first, with room to expand into other disciplines later.

Current capabilities:

- Parse manuscript content from `docx`, `txt`, or direct CLI input.
- Detect manuscript language and keep Chinese and English recommendation pools separate.
- Remove reference sections from the ranking text and isolate Word footnotes from the main body.
- Build journal portraits directly from the local SSCI CSV in `data/`.
- Enrich SSCI journal portraits by crawling real `Aims & Scope` pages and recent article metadata.
- Optionally export a normalized journal dataset from local CSV or JSON source manifests.
- Score journals from three fit dimensions:
  - content fit
  - methodology fit
  - editorial preference fit
- Blend fit with venue factors:
  - aims and scope
  - JCR quartile
  - impact factor
  - review cycle
  - acceptance rate
  - indexing labels
  - recent article examples
- Export ranked recommendations to a UTF-8 CSV file.

Important note:

- This repository ships with a seed law-journal dataset for local testing.
- The current default English-journal workflow no longer depends on crawling legal-journal sites. It reads `data/Social Sciences Citation Index (SSCI).csv`, then generates journal portraits from SSCI metadata and category lookup rules.
- If you run `crawl --manifest data/source_manifest.ssci.json`, the builder will also try to enrich each journal with real `Aims & Scope` text and recent article abstracts/keywords using OpenAlex plus publisher-site crawling.

## Project layout

```text
legal_journal_agent/
├─ data/
│  ├─ legal_journals.seed.json
│  ├─ law_taxonomy.json
│  └─ source_manifest.example.json
├─ output/
├─ src/journal_agent/
│  ├─ cli/
│  ├─ data/
│  ├─ ingestion/
│  ├─ models/
│  ├─ ranking/
│  └─ utils/
└─ README.md
```

## Quick start

Install the package in editable mode first:

```bash
pip install -e .
```

Create recommendations from a manuscript file:

```bash
python -m journal_agent recommend ^
  --manuscript data/sample_manuscript.txt ^
  --dataset "data/Social Sciences Citation Index (SSCI).csv" ^
  --taxonomy data/law_taxonomy.json ^
  --output output/recommendations.csv ^
  --top-k 10
```

Or provide structured text directly:

```bash
python -m journal_agent recommend ^
  --title "Platform governance and algorithmic due process" ^
  --abstract "This article studies due process obligations in platform governance..." ^
  --keywords "platform governance; algorithm; due process; digital regulation" ^
  --dataset "data/Social Sciences Citation Index (SSCI).csv" ^
  --taxonomy data/law_taxonomy.json
```

If you want to export an enriched SSCI dataset JSON with crawled `Aims & Scope` and recent article metadata:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci.json ^
  --output data/legal_journals.collected.json
```

If you only want the fast metadata-derived version without live crawling:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.ssci_quick.json ^
  --output data/legal_journals.quick.json
```

For a local smoke test without external sources:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.demo.json ^
  --output data/legal_journals.demo_output.json
```

## How ranking works

The default ranking engine uses a hybrid heuristic approach that is easy to inspect and extend:

1. Parse manuscript title, abstract, keywords, and body.
2. Detect law-specific methodology and editorial signals from the taxonomy file.
3. Compare the manuscript to each journal profile using TF-IDF character n-gram similarity plus keyword overlap.
4. Compare manuscript sections (title, abstract, keywords, main body) against journal OA article sections when available.
5. For non-OA article samples, compare manuscript title/abstract/keywords against article title/abstract/keywords.
6. Optionally compare manuscript references against references from journal article samples when those fields exist in the dataset.
7. Compute:
   - `content_fit`
   - `methodology_fit`
   - `editorial_fit`
   - `venue_quality`
   - `feasibility`
8. Produce:
   - `overall_score`
   - `match_probability`
   - plain-language rationale

## Data flow

The current default English-journal pipeline is:

1. Read journal names and metadata from `data/Social Sciences Citation Index (SSCI).csv`.
2. Generate a journal portrait for each title from:
   - journal title
   - language
   - publisher
   - Web of Science categories
   - ISSN / eISSN
3. Map SSCI categories into:
   - subdisciplines
   - keywords
   - methodology preferences
   - editorial preferences
4. Use the generated portrait directly in recommendation.

The enriched SSCI build pipeline is:

1. Read the SSCI CSV list.
2. Use ISSN/eISSN and title to locate the journal source in OpenAlex.
3. Pull the journal homepage URL when available.
4. Crawl the publisher site to find and extract real `Aims & Scope` text.
5. Pull recent journal articles from OpenAlex and store their abstracts and keywords in `recent_articles`.
6. Cache enrichment responses under `data/cache/ssci_enrichment/` so reruns do not start from zero.

The source manifest system is still available when you want to export or merge local datasets. It supports:

- `json`: load normalized journal records directly.
- `csv`: import local CSV tables.
- `ssci_csv_lookup`: read the uploaded SSCI CSV and generate journal portraits without crawling.
- `ssci_csv_enriched`: read the uploaded SSCI CSV, then enrich portraits with live crawling and recent article metadata.

Suggested real-world workflow:

1. Keep `data/Social Sciences Citation Index (SSCI).csv` up to date.
2. Run `crawl --manifest data/source_manifest.ssci.json` to build an enriched JSON snapshot.
3. Use that enriched JSON for recommendation when you want higher-quality journal portraits.
4. If you only need a fast baseline, run recommendation directly against the raw CSV.
5. For Chinese journals, use local CSV or JSON imports instead of crawler-dependent flows.

Candidate policy in the current implementation:

- Chinese manuscript: recommend Chinese law journals only.
- English manuscript: recommend English journals only; prioritize law and adjacent humanities/social-science journals (for example SSCI/AHCI/ESCI indexed titles).

Language rule used by the agent:

- The recommendation language is always locked to the manuscript language.
- English manuscript -> English journals only.
- Chinese manuscript -> Chinese journals only.
- Manuscript discipline can be legal, but candidate journals are not hard-limited to law; adjacent humanities/social-science venues can be considered when the topic matches.

## Extending to other disciplines

To support a new discipline later:

1. Add a taxonomy JSON file like `economics_taxonomy.json`.
2. Prepare a normalized journal dataset with `discipline: "economics"`.
3. Tune the keyword, methodology, and editorial signals.
4. Optionally add a discipline-specific scoring engine.

## Limitations

- The included journal seed data is for demo and pipeline verification, not a final production knowledge base.
- `docx` parsing uses the Word XML archive directly and is designed for typical text-heavy academic documents.
- Some source sites may require authentication, JavaScript rendering, or anti-bot handling. In those cases, CSV import is usually the safer first step.
