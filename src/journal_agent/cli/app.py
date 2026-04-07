from __future__ import annotations

import argparse
from pathlib import Path

from journal_agent.data.sources import DatasetBuilder
from journal_agent.ranking.recommender import JournalRecommendationAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="journal-agent",
        description="AI-assisted legal journal recommendation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend = subparsers.add_parser("recommend", help="Recommend journals for a manuscript.")
    recommend.add_argument("--manuscript", help="Path to .docx, .txt, .md, or .json manuscript file.")
    recommend.add_argument("--title", help="Fallback title when no manuscript path is provided.")
    recommend.add_argument("--abstract", help="Fallback abstract when no manuscript path is provided.")
    recommend.add_argument("--keywords", help="Semicolon or comma separated keyword list.")
    recommend.add_argument("--dataset", default="data/legal_journals.seed.json", help="Path to normalized journal dataset JSON.")
    recommend.add_argument("--taxonomy", default="data/law_taxonomy.json", help="Path to taxonomy JSON.")
    recommend.add_argument("--discipline", default="law", help="Discipline key. Default: law.")
    recommend.add_argument("--output", default="output/recommendations.csv", help="CSV output path.")
    recommend.add_argument("--top-k", type=int, default=15, help="Number of journals to keep.")

    crawl = subparsers.add_parser("crawl", help="Build a normalized journal dataset from a manifest.")
    crawl.add_argument("--manifest", required=True, help="Path to source manifest JSON.")
    crawl.add_argument("--output", required=True, help="Path to merged output JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "recommend":
        run_recommend(args)
        return
    if args.command == "crawl":
        run_crawl(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


def run_recommend(args: argparse.Namespace) -> None:
    if not args.manuscript and not args.title:
        raise ValueError("Provide either --manuscript or at least --title.")
    agent = JournalRecommendationAgent()
    manuscript, recommendations = agent.recommend(
        dataset_path=args.dataset,
        taxonomy_path=args.taxonomy,
        manuscript_path=args.manuscript,
        title=args.title,
        abstract=args.abstract,
        keywords=args.keywords,
        discipline=args.discipline,
        top_k=args.top_k,
    )
    agent.export_csv(recommendations, args.output)
    print(f"Manuscript: {manuscript.title}")
    print(f"Language: {manuscript.language}")
    print(f"Output CSV: {Path(args.output).resolve()}")
    print("Top recommendations:")
    for index, item in enumerate(recommendations[:5], start=1):
        print(
            f"{index}. {item.journal.title} | "
            f"overall={item.overall_score:.3f} | "
            f"probability={item.match_probability:.3f} | "
            f"level={item.match_level}"
        )


def run_crawl(args: argparse.Namespace) -> None:
    builder = DatasetBuilder()
    profiles = builder.build_from_manifest(args.manifest)
    builder.save(profiles, args.output)
    print(f"Collected {len(profiles)} journals.")
    print(f"Saved dataset to: {Path(args.output).resolve()}")
