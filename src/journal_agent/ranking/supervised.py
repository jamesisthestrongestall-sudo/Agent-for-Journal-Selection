from __future__ import annotations

import csv
import json
import math
import pickle
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from journal_agent.models.schemas import (
    JournalProfile,
    ManuscriptProfile,
    RecommendationResult,
    is_interdisciplinary_journal_profile,
)
from journal_agent.ranking.subfields import (
    GENERAL_LAW_REVIEW_BUCKET,
    SubfieldProfile,
    bucket_label,
    bucket_similarity,
    build_law_subfield_profile,
)
from journal_agent.utils.text_processing import (
    TOKEN_PATTERN,
    clamp,
    extract_candidate_terms,
    keyword_overlap,
    normalize_space,
    normalize_title_key,
)


VALIDATION_METRICS = {"top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"}
TECHNOLOGY_BUCKET = "technology_privacy_ip"
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
NON_RESEARCH_ARTICLE_TITLES = {
    "about the authors",
    "acknowledgements",
    "back matter",
    "book review",
    "book reviews",
    "contents",
    "correction",
    "editorial",
    "editorial board",
    "erratum",
    "front matter",
    "index",
    "introduction",
    "issue information",
    "masthead",
    "notes on contributors",
    "table of contents",
}
NON_RESEARCH_ARTICLE_PREFIXES = (
    "book review:",
    "correction to:",
    "corrigendum",
    "erratum to:",
    "issue information",
)


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
    article_rank: int = 0

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
    subfield_scores: dict[str, float] = field(default_factory=dict)
    primary_subfield: str | None = None
    focus_subfields: tuple[str, ...] = ()
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
    def __init__(self, *, max_articles_per_journal: int = 8, random_seed: int = 42) -> None:
        self.max_articles_per_journal = max_articles_per_journal
        self.random_seed = random_seed
        self.skipped_non_research_articles = 0

    def build(self, journals: list[JournalProfile]) -> SupervisedDataset:
        candidate_profiles: list[CandidateJournalProfile] = []
        train_samples: list[SupervisedArticleSample] = []
        validation_samples: list[SupervisedArticleSample] = []
        test_samples: list[SupervisedArticleSample] = []
        journals_with_samples = 0
        journals_without_samples = 0
        self.skipped_non_research_articles = 0

        for journal in journals:
            selected_samples, journal_train, journal_validation, journal_test = self._journal_split(journal)
            candidate_profiles.append(self._candidate_profile(journal, selected_samples))
            if not selected_samples:
                journals_without_samples += 1
                continue
            journals_with_samples += 1
            train_samples.extend(journal_train)
            validation_samples.extend(journal_validation)
            test_samples.extend(journal_test)

        split_summary = {
            "candidate_journal_count": len(candidate_profiles),
            "journals_with_samples": journals_with_samples,
            "journals_without_samples": journals_without_samples,
            "max_articles_per_journal": self.max_articles_per_journal,
            "selection_strategy": "representative_mmr_fixed_holdouts",
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            "validation_journal_count": len({sample.journal_id for sample in validation_samples}),
            "test_journal_count": len({sample.journal_id for sample in test_samples}),
            "skipped_non_research_articles": self.skipped_non_research_articles,
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
            if self._is_non_research_article(title=title, abstract=abstract, keywords=keywords):
                self.skipped_non_research_articles += 1
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
                    article_rank=index,
                )
            )
        return usable

    def _is_non_research_article(self, *, title: str, abstract: str, keywords: tuple[str, ...]) -> bool:
        normalized_title = normalize_space(title).lower().strip(" .:-")
        if not normalized_title:
            return False
        if normalized_title in NON_RESEARCH_ARTICLE_TITLES:
            return True
        if any(normalized_title.startswith(prefix) for prefix in NON_RESEARCH_ARTICLE_PREFIXES):
            return True
        if len(normalized_title.split()) <= 2 and not abstract and len(keywords) <= 1:
            return normalized_title in {"editorial", "introduction", "preface", "foreword"}
        return False

    def _journal_split(
        self,
        journal: JournalProfile,
    ) -> tuple[
        list[SupervisedArticleSample],
        list[SupervisedArticleSample],
        list[SupervisedArticleSample],
        list[SupervisedArticleSample],
    ]:
        usable = self._journal_samples(journal)
        if not usable:
            return [], [], [], []

        ranked = self._rank_representative_samples(journal, usable)
        selected_total = min(self.max_articles_per_journal, len(ranked))
        train_count, validation_count, test_count = self._split_counts(selected_total)
        holdout_candidates = self._fixed_holdouts(journal, ranked)
        validation = holdout_candidates[:validation_count]
        test = holdout_candidates[validation_count : validation_count + test_count]
        holdout_ids = {sample.article_id for sample in [*validation, *test]}
        train = [sample for sample in ranked if sample.article_id not in holdout_ids][:train_count]

        selected_lookup = {sample.article_id: sample for sample in [*train, *validation, *test]}
        selected = [sample for sample in ranked if sample.article_id in selected_lookup]
        return selected, train, validation, test

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
        journal_subfields = self._journal_subfield_profile(journal)
        return CandidateJournalProfile(
            journal_id=journal.journal_id or normalize_title_key(journal.title),
            title=journal.title,
            language=journal.language,
            scope_text=scope_text,
            recent_articles_text=recent_articles_text,
            journal_title_text=normalize_space(journal.title),
            keyword_pool=tuple(item for item in keyword_pool if normalize_space(item)),
            publication_count=publication_count,
            subfield_scores=dict(journal_subfields.scores),
            primary_subfield=journal_subfields.primary,
            focus_subfields=tuple(journal_subfields.focus),
            excluded_recent_articles_text=excluded_recent_articles_text,
            journal_payload=journal.model_dump(mode="json"),
        )

    def _article_profile_text(self, sample: SupervisedArticleSample) -> str:
        return normalize_space("\n".join([sample.title, sample.abstract, " ".join(sample.keywords)]))

    def _rank_representative_samples(
        self,
        journal: JournalProfile,
        samples: list[SupervisedArticleSample],
    ) -> list[SupervisedArticleSample]:
        if len(samples) <= 1:
            return samples

        profile_terms = self._journal_profile_terms(journal, samples)
        sample_terms = {sample.article_id: self._sample_terms(sample) for sample in samples}
        term_frequencies: Counter[str] = Counter()
        for terms in sample_terms.values():
            term_frequencies.update(terms)

        sample_count = len(samples)
        base_scores: dict[str, float] = {}
        for sample in samples:
            terms = sample_terms[sample.article_id]
            scope_overlap = self._term_set_overlap(terms, profile_terms)
            centrality = self._corpus_centrality(terms, term_frequencies, sample_count)
            abstract_signal = clamp(len(sample.abstract) / 1200.0)
            keyword_signal = clamp(len(sample.keywords) / 6.0)
            recency_signal = 1.0 - ((sample.article_rank - 1) / max(1, sample_count - 1))
            base_scores[sample.article_id] = (
                (0.38 * scope_overlap)
                + (0.30 * centrality)
                + (0.18 * abstract_signal)
                + (0.08 * keyword_signal)
                + (0.06 * recency_signal)
            )

        selected: list[SupervisedArticleSample] = []
        remaining = list(samples)
        while remaining:
            best_sample: SupervisedArticleSample | None = None
            best_score = -1.0
            for candidate in remaining:
                candidate_terms = sample_terms[candidate.article_id]
                redundancy = max(
                    (self._term_set_overlap(candidate_terms, sample_terms[item.article_id]) for item in selected),
                    default=0.0,
                )
                mmr_score = (0.82 * base_scores[candidate.article_id]) - (0.18 * redundancy)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_sample = candidate
            if best_sample is None:
                break
            selected.append(best_sample)
            remaining = [sample for sample in remaining if sample.article_id != best_sample.article_id]

        return selected

    def _fixed_holdouts(
        self,
        journal: JournalProfile,
        ranked_samples: list[SupervisedArticleSample],
    ) -> list[SupervisedArticleSample]:
        holdouts = list(ranked_samples)
        journal_key = journal.journal_id or normalize_title_key(journal.title)
        rng = random.Random(f"{self.random_seed}:{journal_key}")
        rng.shuffle(holdouts)
        return holdouts

    def _journal_profile_terms(
        self,
        journal: JournalProfile,
        samples: list[SupervisedArticleSample],
    ) -> set[str]:
        seed_terms = [
            *journal.keywords,
            *journal.subdisciplines,
            *[sample.title for sample in samples],
            *[sample.abstract[:600] for sample in samples if sample.abstract],
        ]
        extracted = extract_candidate_terms(
            " ".join([journal.title, journal.aims_and_scope, *seed_terms]),
            top_k=48,
        )
        return {
            normalize_space(term).lower()
            for term in [*journal.keywords, *journal.subdisciplines, *extracted]
            if normalize_space(term)
        }

    def _sample_terms(self, sample: SupervisedArticleSample) -> set[str]:
        extracted = extract_candidate_terms(sample.query_text(), top_k=24)
        return {
            normalize_space(term).lower()
            for term in [*sample.keywords, *extracted]
            if normalize_space(term)
        }

    def _term_set_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        if union == 0:
            return 0.0
        return intersection / union

    def _corpus_centrality(
        self,
        terms: set[str],
        term_frequencies: Counter[str],
        sample_count: int,
    ) -> float:
        if not terms:
            return 0.0
        total = sum(term_frequencies[term] for term in terms)
        return min(total / (len(terms) * max(sample_count, 1)), 1.0)

    def _journal_subfield_profile(self, journal: JournalProfile) -> SubfieldProfile:
        article_keywords = [keyword for article in journal.recent_articles for keyword in article.keywords]
        article_titles = [article.title for article in journal.recent_articles]
        article_abstracts = [
            article.abstract_snippet[:1200]
            for article in journal.recent_articles
            if normalize_space(article.abstract_snippet)
        ]
        return build_law_subfield_profile(
            title=journal.title,
            keywords=[
                *journal.keywords,
                *journal.subdisciplines,
                *article_keywords,
                *extract_candidate_terms(" ".join(article_titles), top_k=20),
            ],
            text_segments=[journal.aims_and_scope, *article_titles, *article_abstracts],
            subdisciplines=journal.subdisciplines,
        )


class SupervisedJournalRanker:
    def __init__(
        self,
        *,
        regularization_values: list[float] | None = None,
        negative_samples_per_query: int | None = None,
        hard_negative_samples_per_query: int = 0,
        random_seed: int = 42,
    ) -> None:
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
        self.regularization_values = regularization_values or [0.5, 1.0, 2.0, 4.0, 8.0]
        self.negative_samples_per_query = negative_samples_per_query
        self.hard_negative_samples_per_query = hard_negative_samples_per_query
        self.random_seed = random_seed

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
                negative_samples_per_query=self.negative_samples_per_query,
                hard_negative_samples_per_query=self.hard_negative_samples_per_query,
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
            negative_samples_per_query=self.negative_samples_per_query,
            hard_negative_samples_per_query=self.hard_negative_samples_per_query,
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
            "regularization_values": self.regularization_values,
            "negative_samples_per_query": self.negative_samples_per_query,
            "hard_negative_samples_per_query": self.hard_negative_samples_per_query,
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
        ranker.candidate_profiles = [ranker._coerce_candidate_profile(profile) for profile in payload["candidate_profiles"]]
        ranker.candidate_indices = {profile.journal_id: index for index, profile in enumerate(ranker.candidate_profiles)}
        ranker.encoder = payload["encoder"]
        ranker.model = payload["model"]
        ranker.feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        ranker.best_config = payload.get("best_config", {})
        ranker.split_summary = payload.get("split_summary", {})
        ranker.regularization_values = list(payload.get("regularization_values") or ranker.regularization_values)
        ranker.negative_samples_per_query = payload.get("negative_samples_per_query")
        ranker.hard_negative_samples_per_query = int(payload.get("hard_negative_samples_per_query") or 0)
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
        manuscript_subfields = self._manuscript_subfield_profile(manuscript)
        predictions = self._rank_sample(
            sample,
            allowed_ids=allowed_ids,
            manuscript_subfields=manuscript_subfields,
        )
        predictions = self._ensure_interdisciplinary_top5(
            predictions,
            candidate_map=candidate_map,
            top_k=top_k,
        )

        recommendations: list[RecommendationResult] = []
        for prediction in predictions[:top_k]:
            journal = candidate_map[prediction["journal_id"]]
            publication_fit = prediction["publication_count_ratio"]
            keyword_fit = max(prediction["keyword_overlap"], prediction["term_overlap"])
            rationale_parts = []
            if prediction["bucket_fit"] >= 0.52 and prediction["journal_primary_subfield"]:
                rationale_parts.append(
                    f"subfield bucket match: {bucket_label(prediction['journal_primary_subfield'])}"
                )
            if prediction["scope_similarity"] >= 0.45:
                rationale_parts.append("Aims & Scope matching is strong")
            if prediction["recent_article_similarity"] >= 0.45:
                rationale_parts.append("recent article profile is similar")
            if publication_fit >= 0.80:
                rationale_parts.append("observed annual publication volume is high")
            if prediction["generic_penalty"] < 1.0 and prediction["generic_penalty_reason"]:
                rationale_parts.append(prediction["generic_penalty_reason"])
            if prediction.get("interdisciplinary_promoted"):
                rationale_parts.append("included to ensure interdisciplinary coverage in top-5")
            rationale = (
                "; ".join(rationale_parts)
                if rationale_parts
                else "supervised score driven by subfield prior, scope, recent articles, and publication volume"
            )
            recommendations.append(
                RecommendationResult(
                    journal=journal,
                    content_fit=clamp((0.58 * prediction["scope_similarity"]) + (0.42 * prediction["recent_article_similarity"])),
                    bucket_fit=prediction["bucket_fit"],
                    scope_fit=prediction["scope_similarity"],
                    article_corpus_fit=prediction["recent_article_similarity"],
                    best_article_fit=0.0,
                    keyword_fit=keyword_fit,
                    methodology_fit=0.0,
                    editorial_fit=0.0,
                    venue_quality=publication_fit,
                    feasibility=publication_fit,
                    overall_score=prediction["rerank_score"],
                    match_probability=prediction["rerank_score"],
                    overexposure_penalty=prediction["generic_penalty"],
                    overexposure_penalty_reason=prediction["generic_penalty_reason"],
                    match_level=self._match_level(prediction["rerank_score"]),
                    rationale=rationale,
                    manuscript_primary_subfield=bucket_label(manuscript_subfields.primary) or None,
                    journal_primary_subfield=bucket_label(prediction["journal_primary_subfield"]) or None,
                    matched_methodologies=[],
                    matched_editorial_signals=[],
                )
            )
        return recommendations

    def _ensure_interdisciplinary_top5(
        self,
        predictions: list[dict[str, Any]],
        *,
        candidate_map: dict[str, JournalProfile],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k < 5 or len(predictions) <= 5:
            return predictions
        if any(self._is_interdisciplinary_journal(candidate_map[item["journal_id"]]) for item in predictions[:5]):
            return predictions

        best_interdisciplinary = next(
            (
                item
                for item in predictions[5:]
                if self._is_interdisciplinary_journal(candidate_map[item["journal_id"]])
            ),
            None,
        )
        if best_interdisciplinary is None:
            return predictions

        promoted = dict(best_interdisciplinary)
        promoted["interdisciplinary_promoted"] = True
        selected = [*predictions[:4], promoted]
        selected_ids = {item["journal_id"] for item in selected}
        remaining = [item for item in predictions[4:] if item["journal_id"] not in selected_ids]
        return [*selected, *remaining]

    def _is_interdisciplinary_journal(self, journal: JournalProfile) -> bool:
        return is_interdisciplinary_journal_profile(journal)

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
        return [{"c": value} for value in self.regularization_values]

    def _pairwise_dataset(
        self,
        *,
        samples: list[SupervisedArticleSample],
        candidate_profiles: list[CandidateJournalProfile],
        encoder: HybridTextEncoder,
        scope_block: EncodedTextBlock,
        recent_block: EncodedTextBlock,
        title_block: EncodedTextBlock,
        negative_samples_per_query: int | None = None,
        hard_negative_samples_per_query: int = 0,
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
            candidate_indices = self._training_candidate_indices(
                sample,
                candidate_profiles=candidate_profiles,
                candidate_index=candidate_index,
                negative_samples_per_query=negative_samples_per_query,
                hard_negative_samples_per_query=hard_negative_samples_per_query,
                scope_scores=scope_scores,
                recent_scores=recent_scores,
                title_scope_scores=title_scope_scores,
                title_recent_scores=title_recent_scores,
                title_title_scores=title_title_scores,
                sample_terms=sample_terms,
            )
            for index in candidate_indices:
                profile = candidate_profiles[index]
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

    def _training_candidate_indices(
        self,
        sample: SupervisedArticleSample,
        *,
        candidate_profiles: list[CandidateJournalProfile],
        candidate_index: dict[str, int],
        negative_samples_per_query: int | None,
        hard_negative_samples_per_query: int,
        scope_scores: np.ndarray,
        recent_scores: np.ndarray,
        title_scope_scores: np.ndarray,
        title_recent_scores: np.ndarray,
        title_title_scores: np.ndarray,
        sample_terms: list[str],
    ) -> list[int]:
        positive_index = candidate_index.get(sample.journal_id)
        all_indices = list(range(len(candidate_profiles)))
        if negative_samples_per_query is None or negative_samples_per_query <= 0:
            return all_indices
        negative_indices = [index for index in all_indices if index != positive_index]
        hard_count = min(max(hard_negative_samples_per_query, 0), negative_samples_per_query, len(negative_indices))
        selected_hard: list[int] = []
        if hard_count:
            hard_scores = []
            for index in negative_indices:
                profile = candidate_profiles[index]
                hard_score = (
                    (0.30 * float(scope_scores[index]))
                    + (0.30 * float(recent_scores[index]))
                    + (0.14 * float(title_scope_scores[index]))
                    + (0.14 * float(title_recent_scores[index]))
                    + (0.06 * float(title_title_scores[index]))
                    + (0.03 * keyword_overlap(sample.keywords, list(profile.keyword_pool)))
                    + (0.03 * keyword_overlap(sample_terms, list(profile.keyword_pool)))
                )
                hard_scores.append((hard_score, index))
            selected_hard = [
                index
                for _, index in sorted(hard_scores, key=lambda item: (-item[0], candidate_profiles[item[1]].title.lower()))[:hard_count]
            ]
        hard_lookup = set(selected_hard)
        random_pool = [index for index in negative_indices if index not in hard_lookup]
        random_count = min(negative_samples_per_query - len(selected_hard), len(random_pool))
        rng = random.Random(f"{self.random_seed}:{sample.sample_id}:{sample.article_id}")
        selected_random = rng.sample(random_pool, k=random_count) if random_count > 0 else []
        selected_negatives = [*selected_hard, *selected_random]
        if positive_index is None:
            return sorted(selected_negatives)
        return [positive_index, *sorted(selected_negatives)]

    def _rank_sample(
        self,
        sample: SupervisedArticleSample,
        *,
        allowed_ids: set[str] | None = None,
        manuscript_subfields: SubfieldProfile | None = None,
    ) -> list[dict[str, Any]]:
        if self.encoder is None or self.model is None or self.scope_block is None or self.recent_block is None or self.title_block is None:
            raise ValueError("Supervised model is not ready.")
        candidate_profiles = self.candidate_profiles
        manuscript_subfields = manuscript_subfields or self._sample_subfield_profile(sample)
        bucket_weight = self._bucket_rerank_weight(manuscript_subfields)
        bucket_confidence = max(manuscript_subfields.scores.values(), default=0.0)
        strong_specialized_bucket = (
            manuscript_subfields.primary is not None
            and manuscript_subfields.primary != GENERAL_LAW_REVIEW_BUCKET
            and bucket_confidence >= 0.34
        )
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
            journal_subfields = self._candidate_subfield_profile(profile)
            bucket_fit = self._bucket_prior_score(manuscript_subfields, journal_subfields)
            generic_penalty, generic_reason = self._generic_law_review_penalty(
                manuscript_subfields,
                journal_subfields,
            )
            if generic_penalty < 1.0:
                bucket_fit = clamp(bucket_fit * generic_penalty)
            overlap = any(bucket in journal_subfields.focus for bucket in manuscript_subfields.focus)
            same_primary = bool(
                manuscript_subfields.primary
                and manuscript_subfields.primary == journal_subfields.primary
            )
            rerank_score = ((1.0 - bucket_weight) * float(probability)) + (bucket_weight * bucket_fit)
            if same_primary:
                rerank_score += 0.025 if manuscript_subfields.primary == TECHNOLOGY_BUCKET else 0.03
            elif overlap:
                rerank_score += 0.014
            elif strong_specialized_bucket:
                rerank_score *= 0.84 if manuscript_subfields.primary == TECHNOLOGY_BUCKET else 0.84
            if generic_penalty < 1.0:
                rerank_score *= generic_penalty
            rerank_score = clamp(rerank_score)
            feature_map = dict(zip(self.feature_names, feature_row.tolist(), strict=False))
            feature_map.update(
                {
                    "journal_id": profile.journal_id,
                    "journal_title": profile.title,
                    "model_probability": float(probability),
                    "probability": float(probability),
                    "bucket_fit": float(bucket_fit),
                    "rerank_score": float(rerank_score),
                    "journal_primary_subfield": profile.primary_subfield,
                    "generic_penalty": float(generic_penalty),
                    "generic_penalty_reason": generic_reason,
                }
            )
            predictions.append(feature_map)
        predictions.sort(
            key=lambda item: (
                -item["rerank_score"],
                -item["bucket_fit"],
                -item["probability"],
                item["journal_title"].lower(),
            )
        )
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

    def _coerce_candidate_profile(self, profile: CandidateJournalProfile | dict[str, Any]) -> CandidateJournalProfile:
        if isinstance(profile, CandidateJournalProfile):
            payload = {
                "journal_id": profile.journal_id,
                "title": profile.title,
                "language": profile.language,
                "scope_text": profile.scope_text,
                "recent_articles_text": profile.recent_articles_text,
                "journal_title_text": profile.journal_title_text,
                "keyword_pool": profile.keyword_pool,
                "publication_count": profile.publication_count,
                "excluded_recent_articles_text": profile.excluded_recent_articles_text,
                "journal_payload": profile.journal_payload,
            }
        else:
            payload = dict(profile)
        journal = JournalProfile.model_validate(payload["journal_payload"])
        journal_subfields = self._journal_subfield_profile(journal)
        return CandidateJournalProfile(
            journal_id=payload["journal_id"],
            title=payload["title"],
            language=payload.get("language"),
            scope_text=payload["scope_text"],
            recent_articles_text=payload["recent_articles_text"],
            journal_title_text=payload["journal_title_text"],
            keyword_pool=tuple(payload.get("keyword_pool", ())),
            publication_count=int(payload.get("publication_count") or 1),
            subfield_scores=dict(payload.get("subfield_scores") or journal_subfields.scores),
            primary_subfield=payload.get("primary_subfield") or journal_subfields.primary,
            focus_subfields=tuple(payload.get("focus_subfields") or journal_subfields.focus),
            excluded_recent_articles_text=dict(payload.get("excluded_recent_articles_text", {})),
            journal_payload=payload["journal_payload"],
        )

    def _journal_subfield_profile(self, journal: JournalProfile) -> SubfieldProfile:
        article_keywords = [keyword for article in journal.recent_articles for keyword in article.keywords]
        article_titles = [article.title for article in journal.recent_articles]
        article_abstracts = [
            article.abstract_snippet[:1200]
            for article in journal.recent_articles
            if normalize_space(article.abstract_snippet)
        ]
        return build_law_subfield_profile(
            title=journal.title,
            keywords=[
                *journal.keywords,
                *journal.subdisciplines,
                *article_keywords,
                *extract_candidate_terms(" ".join(article_titles), top_k=20),
            ],
            text_segments=[journal.aims_and_scope, *article_titles, *article_abstracts],
            subdisciplines=journal.subdisciplines,
        )

    def _candidate_subfield_profile(self, profile: CandidateJournalProfile) -> SubfieldProfile:
        return SubfieldProfile(
            scores=dict(profile.subfield_scores),
            primary=profile.primary_subfield,
            focus=tuple(profile.focus_subfields),
        )

    def _sample_subfield_profile(self, sample: SupervisedArticleSample) -> SubfieldProfile:
        query_terms = extract_candidate_terms(sample.query_text(), top_k=18)
        return build_law_subfield_profile(
            title=sample.title,
            keywords=[*sample.keywords, *query_terms],
            text_segments=[sample.abstract, sample.query_text()],
        )

    def _manuscript_subfield_profile(self, manuscript: ManuscriptProfile) -> SubfieldProfile:
        if manuscript.subfield_scores or manuscript.primary_subfield or manuscript.focus_subfields:
            return SubfieldProfile(
                scores=dict(manuscript.subfield_scores),
                primary=manuscript.primary_subfield,
                focus=tuple(manuscript.focus_subfields),
            )
        return build_law_subfield_profile(
            title=manuscript.title,
            keywords=[*manuscript.keywords, *manuscript.extracted_terms, *manuscript.legal_terms],
            text_segments=[manuscript.abstract, manuscript.full_text[:6000]],
        )

    def _bucket_prior_score(self, manuscript_subfields: SubfieldProfile, journal_subfields: SubfieldProfile) -> float:
        if not manuscript_subfields.scores or not journal_subfields.scores:
            return 0.0
        base_score = bucket_similarity(manuscript_subfields, journal_subfields)
        overlap = any(bucket in journal_subfields.focus for bucket in manuscript_subfields.focus)
        same_primary = bool(
            manuscript_subfields.primary
            and manuscript_subfields.primary == journal_subfields.primary
        )
        technology_focus = TECHNOLOGY_BUCKET in manuscript_subfields.focus
        if same_primary:
            base_score += 0.10 if technology_focus else 0.12
        elif overlap:
            base_score += 0.06 if technology_focus else 0.08
        if technology_focus and journal_subfields.primary == TECHNOLOGY_BUCKET:
            base_score += 0.03
        if (
            manuscript_subfields.primary
            and manuscript_subfields.primary != GENERAL_LAW_REVIEW_BUCKET
            and journal_subfields.primary == GENERAL_LAW_REVIEW_BUCKET
            and not overlap
        ):
            base_score *= 0.88
        return clamp(base_score)

    def _bucket_rerank_weight(self, manuscript_subfields: SubfieldProfile) -> float:
        if not manuscript_subfields.primary or manuscript_subfields.primary == GENERAL_LAW_REVIEW_BUCKET:
            return 0.10
        top_score = max(manuscript_subfields.scores.values(), default=0.0)
        if manuscript_subfields.primary == TECHNOLOGY_BUCKET:
            if top_score >= 0.42:
                return 0.30
            return 0.24
        if top_score >= 0.50 and len(manuscript_subfields.focus) <= 2:
            return 0.28
        if top_score >= 0.36:
            return 0.20
        return 0.14

    def _generic_law_review_penalty(
        self,
        manuscript_subfields: SubfieldProfile,
        journal_subfields: SubfieldProfile,
    ) -> tuple[float, str | None]:
        manuscript_primary = manuscript_subfields.primary
        if not manuscript_primary or manuscript_primary == GENERAL_LAW_REVIEW_BUCKET:
            return 1.0, None

        general_score = journal_subfields.scores.get(GENERAL_LAW_REVIEW_BUCKET, 0.0)
        if general_score < 0.38:
            return 1.0, None

        top_specialized = max(
            (score for bucket, score in journal_subfields.scores.items() if bucket != GENERAL_LAW_REVIEW_BUCKET),
            default=0.0,
        )
        specialized_gap = top_specialized - general_score
        overlap = any(bucket in journal_subfields.focus for bucket in manuscript_subfields.focus)

        if journal_subfields.primary == GENERAL_LAW_REVIEW_BUCKET:
            penalty = 0.72 if not overlap else 0.80
            return penalty, "generic law review penalty"

        if specialized_gap < 0.08:
            penalty = 0.76 if overlap else 0.72
            return penalty, "generic law review penalty"
        if specialized_gap < 0.16:
            penalty = 0.84 if overlap else 0.78
            return penalty, "generic law review penalty"
        if specialized_gap < 0.24:
            return 0.92, "generic law review penalty"
        return 1.0, None
