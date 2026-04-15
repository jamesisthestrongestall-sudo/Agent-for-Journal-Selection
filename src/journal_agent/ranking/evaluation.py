from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from journal_agent.models.schemas import JournalProfile, ManuscriptProfile, TaxonomyProfile
from journal_agent.ranking.scoring import CorpusScoringEngine, CorpusWeightProfile, build_core_weight_candidates
from journal_agent.utils.text_processing import normalize_space, normalize_title_key


VALIDATION_METRICS = {"top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"}


@dataclass(frozen=True)
class ValidationSample:
    sample_id: str
    journal_id: str
    journal_title: str
    article_title: str
    abstract: str
    keywords: list[str]
    language: str = "en"
    discipline: str = "law"


@dataclass
class EvaluationMetrics:
    sample_count: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float


class RecommendationValidator:
    def __init__(
        self,
        taxonomy: TaxonomyProfile,
        *,
        target_metric: str = "top3_accuracy",
        target_threshold: float = 0.90,
        random_seed: int = 42,
    ) -> None:
        if target_metric not in VALIDATION_METRICS:
            raise ValueError(f"Unsupported target metric: {target_metric}")
        self.taxonomy = taxonomy
        self.target_metric = target_metric
        self.target_threshold = target_threshold
        self.random_seed = random_seed

    def build_samples(
        self,
        journals: list[JournalProfile],
        *,
        max_samples_per_journal: int = 2,
        sample_size: int | None = 60,
        min_articles_per_journal: int = 3,
    ) -> list[ValidationSample]:
        rng = random.Random(self.random_seed)
        samples: list[ValidationSample] = []
        for journal in journals:
            eligible_articles = [
                article
                for article in journal.recent_articles
                if normalize_space(article.abstract_snippet) and len(journal.recent_articles) >= min_articles_per_journal
            ]
            if len(eligible_articles) < min_articles_per_journal:
                continue
            if max_samples_per_journal and len(eligible_articles) > max_samples_per_journal:
                eligible_articles = rng.sample(eligible_articles, max_samples_per_journal)
            for index, article in enumerate(eligible_articles, start=1):
                samples.append(
                    ValidationSample(
                        sample_id=f"{journal.journal_id or normalize_title_key(journal.title)}-{index}",
                        journal_id=journal.journal_id or normalize_title_key(journal.title),
                        journal_title=journal.title,
                        article_title=article.title,
                        abstract=article.abstract_snippet,
                        keywords=article.keywords,
                        language=journal.language or "en",
                        discipline="law",
                    )
                )
        rng.shuffle(samples)
        if sample_size is not None:
            samples = samples[:sample_size]
        return samples

    def evaluate(
        self,
        journals: list[JournalProfile],
        *,
        samples: list[ValidationSample],
        core_weights: CorpusWeightProfile,
    ) -> tuple[EvaluationMetrics, list[dict[str, object]]]:
        engine = CorpusScoringEngine(self.taxonomy, core_weights=core_weights)
        detail_rows: list[dict[str, object]] = []
        ranks: list[int] = []
        for sample in samples:
            evaluation_journals = self._hold_out_article(journals, sample)
            manuscript = ManuscriptProfile(
                title=sample.article_title,
                abstract=sample.abstract,
                keywords=sample.keywords,
                language=sample.language,
                discipline=sample.discipline,
            )
            recommendations = engine.score(manuscript, evaluation_journals)
            rank = next(
                (
                    index
                    for index, item in enumerate(recommendations, start=1)
                    if (item.journal.journal_id or normalize_title_key(item.journal.title)) == sample.journal_id
                ),
                len(evaluation_journals) + 1,
            )
            ranks.append(rank)
            top_predictions = recommendations[:5]
            true_recommendation = next(
                (
                    item
                    for item in recommendations
                    if (item.journal.journal_id or normalize_title_key(item.journal.title)) == sample.journal_id
                ),
                None,
            )
            detail_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "true_journal": sample.journal_title,
                    "article_title": sample.article_title,
                    "manuscript_primary_subfield": top_predictions[0].manuscript_primary_subfield if top_predictions else "",
                    "true_journal_primary_subfield": true_recommendation.journal_primary_subfield if true_recommendation else "",
                    "rank": rank,
                    "top1_prediction": top_predictions[0].journal.title if len(top_predictions) >= 1 else "",
                    "top1_prediction_subfield": top_predictions[0].journal_primary_subfield if len(top_predictions) >= 1 else "",
                    "top2_prediction": top_predictions[1].journal.title if len(top_predictions) >= 2 else "",
                    "top3_prediction": top_predictions[2].journal.title if len(top_predictions) >= 3 else "",
                    "top5_predictions": " | ".join(item.journal.title for item in top_predictions),
                    "top1_hit": int(rank == 1),
                    "top3_hit": int(rank <= 3),
                    "top5_hit": int(rank <= 5),
                }
            )
        metrics = self._calculate_metrics(ranks)
        return metrics, detail_rows

    def optimize_core_weights(
        self,
        journals: list[JournalProfile],
        *,
        samples: list[ValidationSample],
    ) -> tuple[CorpusWeightProfile, EvaluationMetrics, list[dict[str, object]]]:
        best_profile: CorpusWeightProfile | None = None
        best_metrics: EvaluationMetrics | None = None
        best_rows: list[dict[str, object]] = []
        best_key: tuple[float, float, float, float] | None = None

        for profile in build_core_weight_candidates():
            metrics, detail_rows = self.evaluate(journals, samples=samples, core_weights=profile)
            candidate_key = (
                getattr(metrics, self.target_metric),
                metrics.top1_accuracy,
                metrics.top3_accuracy,
                metrics.mrr,
            )
            if best_key is None or candidate_key > best_key:
                best_profile = profile
                best_metrics = metrics
                best_rows = detail_rows
                best_key = candidate_key

        if best_profile is None or best_metrics is None:
            raise ValueError("No validation results were generated.")
        return best_profile, best_metrics, best_rows

    def save_report(
        self,
        *,
        report_path: str | Path,
        detail_output_path: str | Path,
        core_weights: CorpusWeightProfile,
        metrics: EvaluationMetrics,
        detail_rows: list[dict[str, object]],
    ) -> None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "target_metric": self.target_metric,
            "target_threshold": self.target_threshold,
            "threshold_met": getattr(metrics, self.target_metric) >= self.target_threshold,
            "core_weights": core_weights.as_dict(),
            "metrics": asdict(metrics),
            "misclassification_summary": self._summarize_misclassifications(detail_rows),
        }
        report_file.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        detail_file = Path(detail_output_path)
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        with detail_file.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()) if detail_rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(detail_rows)

    def _hold_out_article(self, journals: list[JournalProfile], sample: ValidationSample) -> list[JournalProfile]:
        held_out_title_key = normalize_title_key(sample.article_title)
        evaluation_journals: list[JournalProfile] = []
        for journal in journals:
            journal_id = journal.journal_id or normalize_title_key(journal.title)
            if journal_id != sample.journal_id:
                evaluation_journals.append(journal)
                continue
            filtered_articles = [
                article
                for article in journal.recent_articles
                if normalize_title_key(article.title) != held_out_title_key
            ]
            evaluation_journals.append(journal.model_copy(update={"recent_articles": filtered_articles}))
        return evaluation_journals

    def _calculate_metrics(self, ranks: list[int]) -> EvaluationMetrics:
        total = len(ranks)
        if total == 0:
            raise ValueError("No validation samples were available. Make sure journals contain recent articles.")
        return EvaluationMetrics(
            sample_count=total,
            top1_accuracy=sum(1 for rank in ranks if rank == 1) / total,
            top3_accuracy=sum(1 for rank in ranks if rank <= 3) / total,
            top5_accuracy=sum(1 for rank in ranks if rank <= 5) / total,
            mrr=sum(1 / rank for rank in ranks) / total,
        )

    def _summarize_misclassifications(self, detail_rows: list[dict[str, object]]) -> dict[str, object]:
        top1_confusions: Counter[tuple[str, str]] = Counter()
        bucket_confusions: Counter[tuple[str, str]] = Counter()
        journal_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"samples": 0, "rank_total": 0.0, "top1_hits": 0.0, "top3_hits": 0.0})
        error_examples: list[dict[str, object]] = []

        for row in detail_rows:
            true_journal = str(row.get("true_journal", ""))
            top1_prediction = str(row.get("top1_prediction", ""))
            rank = int(row.get("rank", 0) or 0)
            journal_stats[true_journal]["samples"] += 1
            journal_stats[true_journal]["rank_total"] += rank
            journal_stats[true_journal]["top1_hits"] += int(row.get("top1_hit", 0) or 0)
            journal_stats[true_journal]["top3_hits"] += int(row.get("top3_hit", 0) or 0)
            if top1_prediction and top1_prediction != true_journal:
                top1_confusions[(true_journal, top1_prediction)] += 1
                true_bucket = str(row.get("true_journal_primary_subfield", ""))
                predicted_bucket = str(row.get("top1_prediction_subfield", ""))
                if true_bucket and predicted_bucket:
                    bucket_confusions[(true_bucket, predicted_bucket)] += 1
                if len(error_examples) < 15:
                    error_examples.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "true_journal": true_journal,
                            "article_title": row.get("article_title"),
                            "manuscript_primary_subfield": row.get("manuscript_primary_subfield"),
                            "true_journal_primary_subfield": row.get("true_journal_primary_subfield"),
                            "rank": rank,
                            "top1_prediction": top1_prediction,
                            "top1_prediction_subfield": row.get("top1_prediction_subfield"),
                            "top5_predictions": row.get("top5_predictions"),
                        }
                    )

        hardest_journals = []
        for journal, stats in journal_stats.items():
            sample_count = int(stats["samples"])
            if sample_count <= 0:
                continue
            hardest_journals.append(
                {
                    "journal": journal,
                    "sample_count": sample_count,
                    "avg_rank": stats["rank_total"] / sample_count,
                    "top1_accuracy": stats["top1_hits"] / sample_count,
                    "top3_accuracy": stats["top3_hits"] / sample_count,
                }
            )
        hardest_journals.sort(key=lambda item: (-item["avg_rank"], item["top3_accuracy"], item["journal"]))

        top_confusion_list = [
            {
                "true_journal": true_journal,
                "predicted_journal": predicted_journal,
                "count": count,
            }
            for (true_journal, predicted_journal), count in top1_confusions.most_common(15)
        ]
        bucket_confusion_list = [
            {
                "true_bucket": true_bucket,
                "predicted_bucket": predicted_bucket,
                "count": count,
            }
            for (true_bucket, predicted_bucket), count in bucket_confusions.most_common(15)
        ]

        return {
            "error_count": sum(1 for row in detail_rows if int(row.get("top1_hit", 0) or 0) == 0),
            "top1_confusions": top_confusion_list,
            "top1_bucket_confusions": bucket_confusion_list,
            "hardest_journals": hardest_journals[:15],
            "error_examples": error_examples,
        }
