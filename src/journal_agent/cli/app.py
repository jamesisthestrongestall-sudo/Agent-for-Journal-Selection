from __future__ import annotations

import argparse
import sys
from pathlib import Path

from journal_agent.data.sources import DatasetBuilder
from journal_agent.data.repository import JournalRepository
from journal_agent.models.schemas import JournalProfile, is_interdisciplinary_journal_profile
from journal_agent.ranking.recommender import JournalRecommendationAgent
from journal_agent.ranking.evaluation import RecommendationValidator
from journal_agent.ranking.scoring import CorpusWeightProfile
from journal_agent.ranking.supervised import SupervisedJournalDatasetBuilder, SupervisedJournalRanker


def _console_safe(text: object) -> str:
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        value.encode(encoding)
        return value
    except UnicodeEncodeError:
        return value.encode(encoding, errors="replace").decode(encoding)


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
    recommend.add_argument(
        "--dataset",
        default="data/Social Sciences Citation Index (SSCI).csv",
        help="Path to a normalized journal dataset JSON or the uploaded SSCI CSV file.",
    )
    recommend.add_argument("--model", help="Optional supervised model .pkl path. If provided, recommendation uses the trained model.")
    recommend.add_argument("--taxonomy", default="data/law_taxonomy.json", help="Path to taxonomy JSON.")
    recommend.add_argument("--discipline", default="law", help="Discipline key. Default: law.")
    recommend.add_argument(
        "--candidate-scope",
        default="law-related",
        choices=["law-related", "law-only", "scopus-law", "wos-law"],
        help=(
            "Use law-only for explicit law/legal/criminal-justice journals, scopus-law "
            "for active Scopus ASJC Law journals, or wos-law for Web of Science SSCI Law journals."
        ),
    )
    recommend.add_argument(
        "--scopus-source-list",
        default="data/scopus_source_list_mar_2026.xlsx",
        help="Path to the Scopus Source List XLSX. Used only with --candidate-scope scopus-law.",
    )
    recommend.add_argument(
        "--wos-ssci-list",
        default="data/Social Sciences Citation Index (SSCI).csv",
        help="Path to the Web of Science SSCI CSV. Used only with --candidate-scope wos-law.",
    )
    recommend.add_argument("--output", default="output/recommendations.csv", help="CSV output path.")
    recommend.add_argument("--top-k", type=int, default=15, help="Number of journals to keep.")

    crawl = subparsers.add_parser("crawl", help="Build a normalized journal dataset from a local manifest or CSV source.")
    crawl.add_argument("--manifest", required=True, help="Path to source manifest JSON.")
    crawl.add_argument("--output", required=True, help="Path to merged output JSON.")

    evaluate = subparsers.add_parser("evaluate", help="Validate recommendation accuracy with held-out recent articles.")
    evaluate.add_argument(
        "--dataset",
        default="data/legal_journals.ssci.generated.json",
        help="Path to an enriched journal dataset JSON with aims & scope and recent articles.",
    )
    evaluate.add_argument("--taxonomy", default="data/law_taxonomy.json", help="Path to taxonomy JSON.")
    evaluate.add_argument("--discipline", default="law", help="Discipline key. Default: law.")
    evaluate.add_argument("--sample-size", type=int, default=60, help="Maximum number of held-out article samples.")
    evaluate.add_argument(
        "--max-samples-per-journal",
        type=int,
        default=2,
        help="Maximum held-out article samples per journal.",
    )
    evaluate.add_argument(
        "--min-articles-per-journal",
        type=int,
        default=3,
        help="Minimum recent article count required before a journal can enter validation.",
    )
    evaluate.add_argument(
        "--target-metric",
        default="top3_accuracy",
        choices=["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"],
        help="Validation metric used to pick the best core-weight profile.",
    )
    evaluate.add_argument(
        "--target-threshold",
        type=float,
        default=0.90,
        help="Threshold to compare against the target metric.",
    )
    evaluate.add_argument("--report", default="output/evaluation_report.json", help="JSON summary output path.")
    evaluate.add_argument(
        "--details-output",
        default="output/evaluation_details.csv",
        help="CSV detail output path for each validation sample.",
    )
    evaluate.add_argument("--scope-weight", type=float, help="Optional fixed scope weight for one-pass evaluation.")
    evaluate.add_argument(
        "--article-corpus-weight",
        type=float,
        help="Optional fixed article corpus weight for one-pass evaluation.",
    )
    evaluate.add_argument(
        "--best-article-weight",
        type=float,
        help="Optional fixed best-article weight for one-pass evaluation.",
    )
    evaluate.add_argument("--keyword-weight", type=float, help="Optional fixed keyword weight for one-pass evaluation.")

    supervised = subparsers.add_parser("train-supervised", help="Train a supervised journal ranking model with train/validation/test splits.")
    supervised.add_argument(
        "--dataset",
        default="data/legal_journals.law_list.full.json",
        help="Path to an enriched journal dataset JSON with aims & scope and recent articles.",
    )
    supervised.add_argument("--discipline", default="law", help="Discipline key. Default: law.")
    supervised.add_argument("--max-articles-per-journal", type=int, default=8, help="Maximum representative recent articles per journal to use as labeled samples.")
    supervised.add_argument("--article-count-grid", help="Optional comma-separated sweep, for example: 5,6,7,8")
    supervised.add_argument(
        "--regularization-grid",
        help="Optional comma-separated logistic-regression C sweep, for example: 1,2,4. Default: 0.5,1,2,4,8",
    )
    supervised.add_argument(
        "--negative-samples-per-query",
        type=int,
        help="Optional number of negative journals to sample per training article. Use this for large candidate pools.",
    )
    supervised.add_argument(
        "--hard-negative-samples-per-query",
        type=int,
        default=0,
        help="Optional number of sampled negatives chosen from the most text-similar wrong journals.",
    )
    supervised.add_argument(
        "--target-metric",
        default="top3_accuracy",
        choices=["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"],
        help="Validation metric used to select the best supervised configuration.",
    )
    supervised.add_argument("--random-seed", type=int, default=42, help="Random seed for deterministic train/validation/test splits.")
    supervised.add_argument("--model-output", default="output/supervised_journal_model.pkl", help="Path to save the trained supervised model.")
    supervised.add_argument("--report", default="output/supervised_training_report.json", help="JSON summary output path.")
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
    if args.command == "evaluate":
        run_evaluate(args)
        return
    if args.command == "train-supervised":
        run_train_supervised(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


def run_recommend(args: argparse.Namespace) -> None:
    if not args.manuscript and not args.title:
        raise ValueError("Provide either --manuscript or at least --title.")
    agent = JournalRecommendationAgent()
    manuscript, recommendations = agent.recommend(
        dataset_path=args.dataset,
        taxonomy_path=args.taxonomy,
        model_path=args.model,
        manuscript_path=args.manuscript,
        title=args.title,
        abstract=args.abstract,
        keywords=args.keywords,
        discipline=args.discipline,
        top_k=args.top_k,
        candidate_scope=args.candidate_scope,
        scopus_source_list=args.scopus_source_list,
        wos_source_list=args.wos_ssci_list,
    )
    agent.export_results(recommendations, args.output)
    print(_console_safe(f"Manuscript: {manuscript.title}"))
    print(_console_safe(f"Language: {manuscript.language}"))
    print(_console_safe(f"Output file: {Path(args.output).resolve()}"))
    print("Top recommendations:")
    for index, item in enumerate(recommendations[:5], start=1):
        interdisciplinary_label = "interdisciplinary" if is_interdisciplinary_journal_profile(item.journal) else "single-field"
        category_text = _journal_category_text(item.journal)
        print(_console_safe(
            f"{index}. {item.journal.title} | "
            f"{interdisciplinary_label} | "
            f"categories={category_text} | "
            f"overall={item.overall_score:.3f} | "
            f"probability={item.match_probability:.3f} | "
            f"level={item.match_level}"
        ))


def run_crawl(args: argparse.Namespace) -> None:
    builder = DatasetBuilder()
    profiles = builder.build_from_manifest(args.manifest)
    builder.save(profiles, args.output)
    print(f"Collected {len(profiles)} journals.")
    print(f"Saved dataset to: {Path(args.output).resolve()}")
    print("Use this JSON file as --dataset for recommendation if you want the enriched journal portraits.")


def run_evaluate(args: argparse.Namespace) -> None:
    repository = JournalRepository()
    journals = repository.load_journals(args.dataset, discipline=args.discipline)
    taxonomy = repository.load_taxonomy(args.taxonomy)
    validator = RecommendationValidator(
        taxonomy,
        target_metric=args.target_metric,
        target_threshold=args.target_threshold,
    )
    samples = validator.build_samples(
        journals,
        max_samples_per_journal=args.max_samples_per_journal,
        sample_size=args.sample_size,
        min_articles_per_journal=args.min_articles_per_journal,
    )
    if not samples:
        raise ValueError(
            "No validation samples were available. Make sure the dataset is the enriched JSON output from crawl "
            "and that journals contain recent articles."
        )
    fixed_weight_values = [
        args.scope_weight,
        args.article_corpus_weight,
        args.best_article_weight,
        args.keyword_weight,
    ]
    if any(value is not None for value in fixed_weight_values):
        if not all(value is not None for value in fixed_weight_values):
            raise ValueError(
                "If you pass fixed evaluation weights, provide --scope-weight, --article-corpus-weight, "
                "--best-article-weight, and --keyword-weight together."
            )
        core_weights = CorpusWeightProfile(
            scope_weight=args.scope_weight,
            article_corpus_weight=args.article_corpus_weight,
            best_article_weight=args.best_article_weight,
            keyword_weight=args.keyword_weight,
        ).normalized()
        metrics, detail_rows = validator.evaluate(journals, samples=samples, core_weights=core_weights)
    else:
        core_weights, metrics, detail_rows = validator.optimize_core_weights(journals, samples=samples)
    validator.save_report(
        report_path=args.report,
        detail_output_path=args.details_output,
        core_weights=core_weights,
        metrics=metrics,
        detail_rows=detail_rows,
    )
    threshold_met = getattr(metrics, args.target_metric) >= args.target_threshold
    print(f"Validation samples: {metrics.sample_count}")
    print(f"Best core weights: {core_weights.as_dict()}")
    print(f"Top-1 accuracy: {metrics.top1_accuracy:.3f}")
    print(f"Top-3 accuracy: {metrics.top3_accuracy:.3f}")
    print(f"Top-5 accuracy: {metrics.top5_accuracy:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")
    print(f"Threshold met ({args.target_metric} >= {args.target_threshold:.2f}): {threshold_met}")
    print(f"Report file: {Path(args.report).resolve()}")
    print(f"Detail file: {Path(args.details_output).resolve()}")


def run_train_supervised(args: argparse.Namespace) -> None:
    repository = JournalRepository()
    journals = repository.load_journals(args.dataset, discipline=args.discipline)
    article_count_grid = _parse_article_count_grid(args.article_count_grid)
    regularization_grid = _parse_float_grid(args.regularization_grid)
    candidate_counts = article_count_grid or [args.max_articles_per_journal]
    best_run: dict[str, object] | None = None
    sweep_rows: list[dict[str, object]] = []

    for article_count in candidate_counts:
        builder = SupervisedJournalDatasetBuilder(
            max_articles_per_journal=article_count,
            random_seed=args.random_seed,
        )
        dataset = builder.build(journals)
        ranker = SupervisedJournalRanker(
            regularization_values=regularization_grid or None,
            negative_samples_per_query=args.negative_samples_per_query,
            hard_negative_samples_per_query=args.hard_negative_samples_per_query,
            random_seed=args.random_seed,
        )
        report_payload, validation_rows, test_rows = ranker.fit(dataset, target_metric=args.target_metric)
        validation_metric = float(report_payload["validation_metrics"][args.target_metric])
        sweep_rows.append(
            {
                "article_count": article_count,
                "validation_metric": validation_metric,
                "validation_top1": report_payload["validation_metrics"]["top1_accuracy"],
                "validation_top3": report_payload["validation_metrics"]["top3_accuracy"],
                "test_top1": report_payload["test_metrics"]["top1_accuracy"],
                "test_top3": report_payload["test_metrics"]["top3_accuracy"],
            }
        )
        if best_run is None or validation_metric > float(best_run["validation_metric"]):
            best_run = {
                "article_count": article_count,
                "validation_metric": validation_metric,
                "dataset": dataset,
                "ranker": ranker,
                "report_payload": report_payload,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
            }

    if best_run is None:
        raise ValueError("No supervised training run was produced.")

    report_payload = dict(best_run["report_payload"])
    report_payload["article_count_sweep"] = sweep_rows
    ranker = best_run["ranker"]
    ranker.save(args.model_output)
    ranker.save_report(
        report_path=args.report,
        report_payload=report_payload,
        validation_rows=best_run["validation_rows"],
        test_rows=best_run["test_rows"],
    )
    print("Supervised model training completed.")
    print(f"Selected article-count cap: {best_run['article_count']}")
    print(f"Train samples: {best_run['dataset'].split_summary['train_samples']}")
    print(f"Validation samples: {best_run['dataset'].split_summary['validation_samples']}")
    print(f"Test samples: {best_run['dataset'].split_summary['test_samples']}")
    print(f"Validation Top-1 accuracy: {report_payload['validation_metrics']['top1_accuracy']:.3f}")
    print(f"Validation Top-3 accuracy: {report_payload['validation_metrics']['top3_accuracy']:.3f}")
    print(f"Test Top-1 accuracy: {report_payload['test_metrics']['top1_accuracy']:.3f}")
    print(f"Test Top-3 accuracy: {report_payload['test_metrics']['top3_accuracy']:.3f}")
    print(f"Saved model to: {Path(args.model_output).resolve()}")
    print(f"Report file: {Path(args.report).resolve()}")


def _parse_article_count_grid(raw_value: str | None) -> list[int]:
    if not raw_value:
        return []
    values: list[int] = []
    for item in raw_value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        values.append(int(normalized))
    return sorted(set(value for value in values if value > 0))


def _parse_float_grid(raw_value: str | None) -> list[float]:
    if not raw_value:
        return []
    values: list[float] = []
    for item in raw_value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        values.append(float(normalized))
    return sorted(set(value for value in values if value > 0))


def _journal_category_text(journal: JournalProfile) -> str:
    if not journal.subdisciplines:
        return journal.discipline
    return "; ".join(journal.subdisciplines[:4])
