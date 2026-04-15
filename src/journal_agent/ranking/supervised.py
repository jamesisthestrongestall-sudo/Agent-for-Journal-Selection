from __future__ import annotations

import csv
import json
import math
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from journal_agent.models.schemas import JournalProfile, ManuscriptProfile, RecommendationResult
from journal_agent.utils.text_processing import (
    TOKEN_PATTERN,
    clamp,
    extract_candidate_terms,
    keyword_overlap,
    normalize_space,
    normalize_title_key,
)


VALIDATION_METRICS = {"top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"}
FEATURE_NAMES = (
    "scope_similarity",
    "recent_article_similarity",
    "title_scope_similarity",
    "title_recent_similarity",
    "journal_title_similarity",
    "keyword_overlap",
    "term_overlap",
    "publication_count_ratio",
    "publication_count_log",
)
PUBLICATION_COUNT_CAP = 120


def tfidf_tokenizer(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_space(text).lower())


@dataclass(frozen=True)
class EncodedTextBlock:
    word_matrix: Any
    char_matrix: Any


@dataclass(frozen=True)
class SupervisedArticleSample:
    sample_id: str
    article_id: str
    journal_id: str
    journal_title: str
    title: str
    abstract: str
    keywords: tuple[str, ...] = ()
    language: str = "en"

    def query_text(self) -> str:
        keyword_text = " ".join(self.keywords)
        return normalize_space(
            "\n".join(
                [
                    self.title,
                    self.title,
                    self.abstract,
                    keyword_text,
                    keyword_text,
                ]
            )
        )

    def title_text(self) -> str:
        return normalize_space(self.title or self.abstract[:160])

    def keyword_pool(self) -> list[str]:
        query = self.query_text()
        return list(
            dict.fromkeys(
                [
                    *[keyword for keyword in self.keywords if normalize_space(keyword)],
                    *extract_candidate_terms(query, top_k=14),
                ]
            )
        )


@dataclass
class CandidateJournalProfile:
    journal_id: str
    title: str
    language: str | None
    scope_text: str
    recent_articles_text: str
    journal_title_text: str
    keyword_pool: tuple[str, ...]
    publication_count: int
    excluded_recent_articles_text: dict[str, str] = field(default_factory=dict)
    journal_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingMetrics:
    sample_count: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float


@dataclass
class SupervisedDataset:
    candidate_profiles: list[CandidateJournalProfile]
    train_samples: list[SupervisedArticleSample]
    validation_samples: list[SupervisedArticleSample]
    test_samples: list[SupervisedArticleSample]
    split_summary: dict[str, Any]


class HybridTextEncoder:
    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=tfidf_tokenizer,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
        )
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    def fit(self, corpus: list[str]) -> "HybridTextEncoder":
        cleaned = [normalize_space(text) or "empty" for text in corpus]
        self.word_vectorizer.fit(cleaned)
        self.char_vectorizer.fit(cleaned)
        return self

    def encode(self, texts: list[str]) -> EncodedTextBlock:
        cleaned = [normalize_space(text) or "empty" for text in texts]
        return EncodedTextBlock(
            word_matrix=self.word_vectorizer.transform(cleaned),
            char_matrix=self.char_vectorizer.transform(cleaned),
        )

    def similarity(self, left: EncodedTextBlock, right: EncodedTextBlock) -> np.ndarray:
        word_scores = left.word_matrix @ right.word_matrix.T
        char_scores = left.char_matrix @ right.char_matrix.T
        return (0.55 * word_scores.toarray()) + (0.45 * char_scores.toarray())


class SupervisedJournalDatasetBuilder:
    def __init__(self, *, max_articles_per_journal: int = 5, random_seed: int = 42) -> None:
        self.max_articles_per_journal = max_articles_per_journal
        self.random_seed = random_seed

    def build(self, journals: list[JournalProfile]) -> SupervisedDataset:
        rng = random.Random(self.random_seed)
        candidate_profiles: list[CandidateJournalProfile] = []
        train_samples: list[SupervisedArticleSample] = []
        validation_samples: list[SupervisedArticleSample] = []
        test_samples: list[SupervisedArticleSample] = []
        journals_with_samples = 0
        journals_without_samples = 0

        for journal in journals:
            samples = self._journal_samples(journal)
            candidate_profiles.append(self._candidate_profile(journal, samples))
            if not samples:
                journals_without_samples += 1
                continue
            journals_with_samples += 1
            split_samples = list(samples)
            rng.shuffle(split_samples)
            train_cut, validation_cut, test_cut = self._split_counts(len(split_samples))
            train_samples.extend(split_samples[:train_cut])
            validation_samples.extend(split_samples[train_cut : train_cut + validation_cut])
            test_samples.extend(split_samples[train_cut + validation_cut : train_cut + validation_cut + test_cut])

        split_summary = {
            "candidate_journal_count": len(candidate_profiles),
            "journals_with_samples": journals_with_samples,
            "journals_without_samples": journals_without_samples,
            "max_articles_per_journal": self.max_articles_per_journal,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            "validation_journal_count": len({sample.journal_id for sample in validation_samples}),
            "test_journal_count": len({sample.journal_id for sample in test_samples}),
        }
        return SupervisedDataset(
            candidate_profiles=candidate_profiles,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            split_summary=split_summary,
        )

    def _split_counts(self, sample_count: int) -> tuple[int, int, int]:
        if sample_count >= 5:
            return sample_count - 2, 1, 1
        if sample_count == 4:
            return 2, 1, 1
        if sample_count == 3:
            return 1, 1, 1
        if sample_count == 2:
            return 1, 0, 1
        if sample_count == 1:
            return 1, 0, 0
        return 0, 0, 0

    def _journal_samples(self, journal: JournalProfile) -> list[SupervisedArticleSample]:
        usable = []
        for index, article in enumerate(journal.recent_articles, start=1):
            title = normalize_space(article.title)
            abstract = normalize_space(article.abstract_snippet)
            keywords = tuple(keyword for keyword in article.keywords if normalize_space(keyword))
            if not title and not abstract and not keywords:
                continue
            article_key = normalize_title_key(title) or f"article-{index}"
            usable.append(
                SupervisedArticleSample(
                    sample_id=f"{journal.journal_id or normalize_title_key(journal.title)}-{index}",
                    article_id=article_key,
                    journal_id=journal.journal_id or normalize_title_key(journal.title),
                    journal_title=journal.title,
                    title=title,
                    abstract=abstract,
                    keywords=keywords,
                    language=journal.language or "en",
                )
            )
            if len(usable) >= self.max_articles_per_journal:
                break
        return usable

    def _candidate_profile(self, journal: JournalProfile, samples: list[SupervisedArticleSample]) -> CandidateJournalProfile:
        scope_text = normalize_space(
            "\n".join(
                [
                    journal.title,
                    journal.aims_and_scope,
                    " ".join(journal.subdisciplines),
                    " ".join(journal.keywords),
                ]
            )
        )
        article_texts = [self._article_profile_text(sample) for sample in samples]
        recent_articles_text = normalize_space("\n".join(article_texts)) or scope_text
        excluded_recent_articles_text: dict[str, str] = {}
        for sample in samples:
            remaining = [self._article_profile_text(other) for other in samples if other.article_id != sample.article_id]
            excluded_recent_articles_text[sample.article_id] = normalize_space("\n".join(remaining)) or scope_text

        keyword_pool = list(dict.fromkeys([*journal.keywords, *journal.subdisciplines]))
        for sample in samples:
            keyword_pool.extend(sample.keywords)
            keyword_pool.extend(extract_candidate_terms(sample.title, top_k=6))

        publication_count = journal.annual_publication_count
        if publication_count is None:
            publication_count = sum(
                1
                for article in journal.recent_articles
                if normalize_space(article.title) or normalize_space(article.abstract_snippet) or article.keywords
            )
        return CandidateJournalProfile(
            journal_id=journal.journal_id or normalize_title_key(journal.title),
            title=journal.title,
            language=journal.language,
            scope_text=scope_text,
            recent_articles_text=recent_articles_text,
            journal_title_text=normalize_space(journal.title),
            keyword_pool=tuple(item for item in keyword_pool if normalize_space(item)),
            publication_count=publication_count,
            excluded_recent_articles_text=excluded_recent_articles_text,
            journal_payload=journal.model_dump(mode="json"),
        )

    def _article_profile_text(self, sample: SupervisedArticleSample) -> str:
        return normalize_space("\n".join([sample.title, sample.abstract, " ".join(sample.keywords)]))


class SupervisedJournalRanker:
    def __init__(self) -> None:
        self.encoder: HybridTextEncoder | None = None
        self.model: Pipeline | None = None
        self.candidate_profiles: list[CandidateJournalProfile] = []
        self.candidate_indices: dict[str, int] = {}
        self.scope_block: EncodedTextBlock | None = None
        self.recent_block: EncodedTextBlock | None = None
        self.title_block: EncodedTextBlock | None = None
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        self.best_config: dict[str, Any] = {}
        self.split_summary: dict[str, Any] = {}
        self.validation_metrics: RankingMetrics | None = None
        self.test_metrics: RankingMetrics | None = None

    def fit(
        self,
        dataset: SupervisedDataset,
        *,
        target_metric: str = "top3_accuracy",
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        if target_metric not in VALIDATION_METRICS:
            raise ValueError(f"Unsupported target metric: {target_metric}")
        self.candidate_profiles = list(dataset.candidate_profiles)
        self.candidate_indices = {profile.journal_id: index for index, profile in enumerate(self.candidate_profiles)}
        self.split_summary = dict(dataset.split_summary)

        best_state: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float] | None = None

        for config in self._candidate_configs():
            encoder = self._fit_encoder(dataset.train_samples, dataset.candidate_profiles)
            scope_block, recent_block, title_block = self._candidate_blocks(encoder, dataset.candidate_profiles)
            train_x, train_y = self._pairwise_dataset(
                samples=dataset.train_samples,
                candidate_profiles=dataset.candidate_profiles,
                encoder=encoder,
                scope_block=scope_block,
                recent_block=recent_block,
                title_block=title_block,
            )
            model = self._fit_model(train_x, train_y, config)
            validation_metrics, validation_rows = self._evaluate(
                samples=dataset.validation_samples,
                candidate_profiles=dataset.candidate_profiles,
                encoder=encoder,
                model=model,
                scope_block=scope_block,
                recent_block=recent_block,
                title_block=title_block,
            )
            candidate_key = (
                getattr(validation_metrics, target_metric),
                validation_metrics.top1_accuracy,
                validation_metrics.top3_accuracy,
                validation_metrics.mrr,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_state = {
                    "config": config,
                    "validation_metrics": validation_metrics,
                    "validation_rows": validation_rows,
                }

        if best_state is None:
            raise ValueError("No supervised model could be trained.")

        combined_train_samples = [*dataset.train_samples, *dataset.validation_samples]
        self.best_config = dict(best_state["config"])
        self.validation_metrics = best_state["validation_metrics"]
        self.encoder = self._fit_encoder(combined_train_samples, dataset.candidate_profiles)
        self.scope_block, self.recent_block, self.title_block = self._candidate_blocks(self.encoder, dataset.candidate_profiles)
        train_x, train_y = self._pairwise_dataset(
            samples=combined_train_samples,
            candidate_profiles=dataset.candidate_profiles,
            encoder=self.encoder,
            scope_block=self.scope_block,
            recent_block=self.recent_block,
            title_block=self.title_block,
        )
        self.model = self._fit_model(train_x, train_y, self.best_config)
        self.test_metrics, test_rows = self._evaluate(
            samples=dataset.test_samples,
            candidate_profiles=dataset.candidate_profiles,
            encoder=self.encoder,
            model=self.model,
            scope_block=self.scope_block,
            recent_block=self.recent_block,
            title_block=self.title_block,
        )
        report = {
            "target_metric": target_metric,
            "best_config": self.best_config,
            "feature_names": list(self.feature_names),
            "split_summary": self.split_summary,
            "validation_metrics": asdict(self.validation_metrics),
            "test_metrics": asdict(self.test_metrics),
        }
        return report, list(best_state["validation_rows"]), test_rows

    def save(self, path: str | Path) -> None:
        if self.encoder is None or self.model is None:
            raise ValueError("Train the model before saving it.")
        payload = {
            "candidate_profiles": self.candidate_profiles,
            "encoder": self.encoder,
            "model": self.model,
            "feature_names": self.feature_names,
            "best_config": self.best_config,
            "split_summary": self.split_summary,
            "validation_metrics": asdict(self.validation_metrics) if self.validation_metrics else None,
            "test_metrics": asdict(self.test_metrics) if self.test_metrics else None,
        }
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedJournalRanker":
        payload = pickle.loads(Path(path).read_bytes())
        ranker = cls()
        ranker.candidate_profiles = payload["candidate_profiles"]
        ranker.candidate_indices = {profile.journal_id: index for index, profile in enumerate(ranker.candidate_profiles)}
        ranker.encoder = payload["encoder"]
        ranker.model = payload["model"]
        ranker.feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        ranker.best_config = payload.get("best_config", {})
        ranker.split_summary = payload.get("split_summary", {})
        if payload.get("validation_metrics"):
            ranker.validation_metrics = RankingMetrics(**payload["validation_metrics"])
        if payload.get("test_metrics"):
            ranker.test_metrics = RankingMetrics(**payload["test_metrics"])
        ranker.scope_block, ranker.recent_block, ranker.title_block = ranker._candidate_blocks(ranker.encoder, ranker.candidate_profiles)
        return ranker

    def recommend(
        self,
        manuscript: ManuscriptProfile,
        *,
        candidate_journals: list[JournalProfile] | None = None,
        top_k: int = 15,
    ) -> list[RecommendationResult]:
        if self.encoder is None or self.model is None:
            raise ValueError("Load or train a supervised model before running recommendation.")
        candidate_map = {
            profile.journal_id: JournalProfile.model_validate(profile.journal_payload)
            for profile in self.candidate_profiles
        }
        if candidate_journals is not None:
            for journal in candidate_journals:
                candidate_map[journal.journal_id or normalize_title_key(journal.title)] = journal
            allowed_ids = {
                journal.journal_id or normalize_title_key(journal.title)
                for journal in candidate_journals
                if (journal.journal_id or normalize_title_key(journal.title)) in self.candidate_indices
            }
        else:
            allowed_ids = set(self.candidate_indices)

        sample = SupervisedArticleSample(
            sample_id="manuscript-query",
            article_id="manuscript-query",
            journal_id="manuscript-query",
            journal_title="",
            title=normalize_space(manuscript.title),
            abstract=normalize_space(manuscript.abstract or manuscript.full_text[:2000]),
            keywords=tuple(keyword for keyword in manuscript.keywords if normalize_space(keyword)),
            language=manuscript.language,
        )
        predictions = self._rank_sample(sample, allowed_ids=allowed_ids)

        recommendations: list[RecommendationResult] = []
        for prediction in predictions[:top_k]:
            journal = candidate_map[prediction["journal_id"]]
            publication_fit = prediction["publication_count_ratio"]
            keyword_fit = max(prediction["keyword_overlap"], prediction["term_overlap"])
            rationale_parts = []
            if prediction["scope_similarity"] >= 0.45:
                rationale_parts.append("Aims & Scope matching is strong")
            if prediction["recent_article_similarity"] >= 0.45:
                rationale_parts.append("recent 5-article profile is similar")
            if publication_fit >= 0.80:
                rationale_parts.append("observed annual publication volume is high")
            rationale = "; ".join(rationale_parts) if rationale_parts else "supervised score driven by scope, recent articles, and publication volume"
            recommendations.append(
                RecommendationResult(
                    journal=journal,
                    content_fit=clamp((0.58 * prediction["scope_similarity"]) + (0.42 * prediction["recent_article_similarity"])),
                    bucket_fit=0.0,
                    scope_fit=prediction["scope_similarity"],
                    article_corpus_fit=prediction["recent_article_similarity"],
                    best_article_fit=0.0,
                    keyword_fit=keyword_fit,
                    methodology_fit=0.0,
                    editorial_fit=0.0,
                    venue_quality=publication_fit,
                    feasibility=publication_fit,
                    overall_score=prediction["probability"],
                    match_probability=prediction["probability"],
                    overexposure_penalty=1.0,
                    overexposure_penalty_reason=None,
                    match_level=self._match_level(prediction["probability"]),
                    rationale=rationale,
                    manuscript_primary_subfield=None,
                    journal_primary_subfield=None,
                    matched_methodologies=[],
                    matched_editorial_signals=[],
                )
            )
        return recommendations

    def save_report(
        self,
        *,
        report_path: str | Path,
        report_payload: dict[str, Any],
        validation_rows: list[dict[str, Any]],
        test_rows: list[dict[str, Any]],
    ) -> None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        validation_path = report_file.with_name(f"{report_file.stem}.validation.csv")
        test_path = report_file.with_name(f"{report_file.stem}.test.csv")
        self._write_rows(validation_path, validation_rows)
        self._write_rows(test_path, test_rows)

    def _fit_encoder(
        self,
        train_samples: list[SupervisedArticleSample],
        candidate_profiles: list[CandidateJournalProfile],
    ) -> HybridTextEncoder:
        corpus = [sample.query_text() for sample in train_samples]
        corpus.extend(profile.scope_text for profile in candidate_profiles)
        corpus.extend(profile.recent_articles_text for profile in candidate_profiles)
        corpus.extend(profile.journal_title_text for profile in candidate_profiles)
        encoder = HybridTextEncoder()
        return encoder.fit(corpus)

    def _candidate_blocks(
        self,
        encoder: HybridTextEncoder,
        candidate_profiles: list[CandidateJournalProfile],
    ) -> tuple[EncodedTextBlock, EncodedTextBlock, EncodedTextBlock]:
        return (
            encoder.encode([profile.scope_text for profile in candidate_profiles]),
            encoder.encode([profile.recent_articles_text for profile in candidate_profiles]),
            encoder.encode([profile.journal_title_text for profile in candidate_profiles]),
        )

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray, config: dict[str, Any]) -> Pipeline:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(config["c"]),
                        class_weight="balanced",
                        max_iter=4000,
                    ),
                ),
            ]
        )
        model.fit(train_x, train_y)
        return model

    def _candidate_configs(self) -> list[dict[str, Any]]:
        return [{"c": value} for value in (0.5, 1.0, 2.0, 4.0, 8.0)]

    def _pairwise_dataset(
        self,
        *,
        samples: list[SupervisedArticleSample],
        candidate_profiles: list[CandidateJournalProfile],
        encoder: HybridTextEncoder,
        scope_block: EncodedTextBlock,
        recent_block: EncodedTextBlock,
        title_block: EncodedTextBlock,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        labels: list[int] = []
        candidate_index = {profile.journal_id: index for index, profile in enumerate(candidate_profiles)}
        max_publication_count = max((self._scaled_publication_count(profile.publication_count) for profile in candidate_profiles), default=1.0)

        for sample in samples:
            query_block = encoder.encode([sample.query_text()])
            manuscript_title_block = encoder.encode([sample.title_text()])
            scope_scores = encoder.similarity(query_block, scope_block)[0]
            recent_scores = encoder.similarity(query_block, recent_block)[0]
            title_scope_scores = encoder.similarity(manuscript_title_block, scope_block)[0]
            title_recent_scores = encoder.similarity(manuscript_title_block, recent_block)[0]
            title_title_scores = encoder.similarity(manuscript_title_block, title_block)[0]

            positive_index = candidate_index.get(sample.journal_id)
            if positive_index is not None:
                positive_profile = candidate_profiles[positive_index]
                excluded_text = positive_profile.excluded_recent_articles_text.get(sample.article_id)
                if excluded_text:
                    excluded_block = encoder.encode([excluded_text])
                    recent_scores[positive_index] = encoder.similarity(query_block, excluded_block)[0][0]
                    title_recent_scores[positive_index] = encoder.similarity(manuscript_title_block, excluded_block)[0][0]

            sample_terms = sample.keyword_pool()
            for index, profile in enumerate(candidate_profiles):
                scaled_publication_count = self._scaled_publication_count(profile.publication_count)
                publication_ratio = scaled_publication_count / max_publication_count if max_publication_count else 0.0
                rows.append(
                    [
                        float(scope_scores[index]),
                        float(recent_scores[index]),
                        float(title_scope_scores[index]),
                        float(title_recent_scores[index]),
                        float(title_title_scores[index]),
                        keyword_overlap(sample.keywords, list(profile.keyword_pool)),
                        keyword_overlap(sample_terms, list(profile.keyword_pool)),
                        publication_ratio,
                        math.log1p(scaled_publication_count),
                    ]
                )
                labels.append(1 if profile.journal_id == sample.journal_id else 0)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)

    def _rank_sample(
        self,
        sample: SupervisedArticleSample,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.encoder is None or self.model is None or self.scope_block is None or self.recent_block is None or self.title_block is None:
            raise ValueError("Supervised model is not ready.")
        candidate_profiles = self.candidate_profiles
        max_publication_count = max((self._scaled_publication_count(profile.publication_count) for profile in candidate_profiles), default=1.0)
        query_block = self.encoder.encode([sample.query_text()])
        manuscript_title_block = self.encoder.encode([sample.title_text()])
        scope_scores = self.encoder.similarity(query_block, self.scope_block)[0]
        recent_scores = self.encoder.similarity(query_block, self.recent_block)[0]
        title_scope_scores = self.encoder.similarity(manuscript_title_block, self.scope_block)[0]
        title_recent_scores = self.encoder.similarity(manuscript_title_block, self.recent_block)[0]
        title_title_scores = self.encoder.similarity(manuscript_title_block, self.title_block)[0]

        positive_index = self.candidate_indices.get(sample.journal_id)
        if positive_index is not None:
            excluded_text = self.candidate_profiles[positive_index].excluded_recent_articles_text.get(sample.article_id)
            if excluded_text:
                excluded_block = self.encoder.encode([excluded_text])
                recent_scores[positive_index] = self.encoder.similarity(query_block, excluded_block)[0][0]
                title_recent_scores[positive_index] = self.encoder.similarity(manuscript_title_block, excluded_block)[0][0]

        sample_terms = sample.keyword_pool()
        rows: list[list[float]] = []
        profile_lookup: list[CandidateJournalProfile] = []
        for index, profile in enumerate(candidate_profiles):
            if allowed_ids is not None and profile.journal_id not in allowed_ids:
                continue
            scaled_publication_count = self._scaled_publication_count(profile.publication_count)
            publication_ratio = scaled_publication_count / max_publication_count if max_publication_count else 0.0
            rows.append(
                [
                    float(scope_scores[index]),
                    float(recent_scores[index]),
                    float(title_scope_scores[index]),
                    float(title_recent_scores[index]),
                    float(title_title_scores[index]),
                    keyword_overlap(sample.keywords, list(profile.keyword_pool)),
                    keyword_overlap(sample_terms, list(profile.keyword_pool)),
                    publication_ratio,
                    math.log1p(scaled_publication_count),
                ]
            )
            profile_lookup.append(profile)
        features = np.asarray(rows, dtype=float)
        probabilities = self.model.predict_proba(features)[:, 1]
        predictions = []
        for profile, feature_row, probability in zip(profile_lookup, features, probabilities, strict=False):
            feature_map = dict(zip(self.feature_names, feature_row.tolist(), strict=False))
            feature_map.update(
                {
                    "journal_id": profile.journal_id,
                    "journal_title": profile.title,
                    "probability": float(probability),
                }
            )
            predictions.append(feature_map)
        predictions.sort(key=lambda item: (-item["probability"], item["journal_title"].lower()))
        return predictions

    def _evaluate(
        self,
        *,
        samples: list[SupervisedArticleSample],
        candidate_profiles: list[CandidateJournalProfile],
        encoder: HybridTextEncoder,
        model: Pipeline,
        scope_block: EncodedTextBlock,
        recent_block: EncodedTextBlock,
        title_block: EncodedTextBlock,
    ) -> tuple[RankingMetrics, list[dict[str, Any]]]:
        previous_state = (
            self.encoder,
            self.model,
            self.candidate_profiles,
            self.candidate_indices,
            self.scope_block,
            self.recent_block,
            self.title_block,
        )
        self.encoder = encoder
        self.model = model
        self.candidate_profiles = list(candidate_profiles)
        self.candidate_indices = {profile.journal_id: index for index, profile in enumerate(candidate_profiles)}
        self.scope_block = scope_block
        self.recent_block = recent_block
        self.title_block = title_block
        try:
            ranks: list[int] = []
            rows: list[dict[str, Any]] = []
            for sample in samples:
                predictions = self._rank_sample(sample)
                rank = next(
                    (index for index, prediction in enumerate(predictions, start=1) if prediction["journal_id"] == sample.journal_id),
                    len(predictions) + 1,
                )
                ranks.append(rank)
                top_predictions = predictions[:5]
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "true_journal": sample.journal_title,
                        "article_title": sample.title,
                        "rank": rank,
                        "top1_prediction": top_predictions[0]["journal_title"] if len(top_predictions) >= 1 else "",
                        "top2_prediction": top_predictions[1]["journal_title"] if len(top_predictions) >= 2 else "",
                        "top3_prediction": top_predictions[2]["journal_title"] if len(top_predictions) >= 3 else "",
                        "top5_predictions": " | ".join(item["journal_title"] for item in top_predictions),
                        "top1_hit": int(rank == 1),
                        "top3_hit": int(rank <= 3),
                        "top5_hit": int(rank <= 5),
                    }
                )
            return self._metrics_from_ranks(ranks), rows
        finally:
            (
                self.encoder,
                self.model,
                self.candidate_profiles,
                self.candidate_indices,
                self.scope_block,
                self.recent_block,
                self.title_block,
            ) = previous_state

    def _metrics_from_ranks(self, ranks: list[int]) -> RankingMetrics:
        total = len(ranks)
        if total <= 0:
            return RankingMetrics(sample_count=0, top1_accuracy=0.0, top3_accuracy=0.0, top5_accuracy=0.0, mrr=0.0)
        return RankingMetrics(
            sample_count=total,
            top1_accuracy=sum(1 for rank in ranks if rank == 1) / total,
            top3_accuracy=sum(1 for rank in ranks if rank <= 3) / total,
            top5_accuracy=sum(1 for rank in ranks if rank <= 5) / total,
            mrr=sum(1 / rank for rank in ranks) / total,
        )

    def _match_level(self, probability: float) -> str:
        if probability >= 0.80:
            return "High"
        if probability >= 0.65:
            return "Good"
        if probability >= 0.50:
            return "Medium"
        return "Low"

    def _write_rows(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(rows)

    def _scaled_publication_count(self, publication_count: int | None) -> float:
        if publication_count is None:
            return 0.0
        return float(min(max(publication_count, 0), PUBLICATION_COUNT_CAP))
