# Legal Journal Agent

`Legal Journal Agent` is a Python journal recommendation agent scaffold focused on law journals first, with room to expand into other disciplines later.

Current capabilities:

- Parse manuscript content from `docx`, `txt`, or direct CLI input.
- Detect manuscript language and keep Chinese and English recommendation pools separate.
- Remove reference sections from the ranking text and isolate Word footnotes from the main body.
- Build a normalized journal knowledge base from JSON, CSV, or HTML source manifests.
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
- Real JCR, CSSCI, and PKU core data should be refreshed through official exports or source-specific crawlers because those lists change over time and some sources are access-controlled.

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
  --dataset data/legal_journals.seed.json ^
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
  --dataset data/legal_journals.seed.json ^
  --taxonomy data/law_taxonomy.json
```

Build a merged journal dataset from a manifest:

```bash
python -m journal_agent crawl ^
  --manifest data/source_manifest.example.json ^
  --output data/legal_journals.collected.json
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

## Data collection strategy

The source manifest system lets you mix multiple source types:

- `json`: load normalized journal records directly.
- `csv`: import exported tables, useful for JCR or manual PKU core lists.
- `html_list`: crawl public listing pages with CSS selectors.

Suggested real-world workflow:

1. Use official or licensed exports for all eligible `SSCI` journals if you want English manuscripts to consider the full SSCI candidate pool.
2. Use official or licensed exports for JCR law journals.
3. Use official or institution-maintained pages for CSSCI law journals where crawling is permitted.
4. Use manually curated CSV imports for PKU core law journals if no stable public machine-readable source exists.
5. Merge everything into one normalized dataset, then run recommendation.

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
