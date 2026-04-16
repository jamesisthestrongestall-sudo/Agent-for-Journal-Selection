from __future__ import annotations

from dataclasses import dataclass

from journal_agent.models.schemas import JournalProfile, ManuscriptProfile, RecommendationResult, TaxonomyEntry, TaxonomyProfile
from journal_agent.ranking.subfields import (
    GENERAL_LAW_REVIEW_BUCKET,
    SubfieldProfile,
    bucket_label,
    bucket_similarity,
    build_law_subfield_profile,
)
from journal_agent.utils.text_processing import (
    clamp,
    extract_candidate_terms,
    hybrid_similarity_scores,
    keyword_overlap,
    normalize_space,
    normalize_title_key,
    safe_mean,
    top_k_mean,
)


INDEXING_WEIGHTS = {
    "jcr": 0.90,
    "ssci": 0.95,
    "sci": 0.95,
    "cssci": 0.85,
    "scopus": 0.72,
}

QUARTILE_WEIGHTS = {
    "Q1": 1.00,
    "Q2": 0.82,
    "Q3": 0.64,
    "Q4": 0.48,
}

LEGAL_SIGNAL_GROUPS = [
    {
        "name": "legal_core",
        "weight": 0.28,
        "aliases": ["law", "legal", "jurisprudence", "juridical", "rule of law"],
    },
    {
        "name": "judicial_process",
        "weight": 0.18,
        "aliases": ["court", "judicial", "litigation", "adjudication", "due process", "tribunal"],
    },
    {
        "name": "legislation_regulation",
        "weight": 0.16,
        "aliases": ["legislation", "statute", "regulation", "regulatory", "compliance"],
    },
    {
        "name": "rights_public_law",
        "weight": 0.14,
        "aliases": ["constitutional", "human rights", "administrative law", "public law"],
    },
    {
        "name": "private_commercial_law",
        "weight": 0.14,
        "aliases": ["civil law", "contract", "corporate law", "commercial law", "private law"],
    },
    {
        "name": "criminal_law",
        "weight": 0.14,
        "aliases": [
            "criminal law",
            "crime",
            "criminal justice",
            "penology",
            "policing",
            "rape",
            "marital rape",
            "sexual assault",
            "sexual violence",
            "consent",
            "coercion",
            "intimate partner violence",
            "violence against women",
        ],
    },
    {
        "name": "comparative_international_law",
        "weight": 0.12,
        "aliases": ["comparative law", "international law", "transnational law"],
    },
    {
        "name": "digital_law",
        "weight": 0.12,
        "aliases": [
            "platform governance",
            "ai regulation",
            "algorithmic governance",
            "data protection",
            "cyber law",
        ],
    },
]

LAW_METHOD_SIGNAL_NAMES = {"doctrinal", "comparative", "case_study"}
LAW_EDITORIAL_SIGNAL_NAMES = {
    "judicial_practice",
    "legislation_policy",
    "international_rule_of_law",
    "commercial_finance",
    "criminal_justice",
    "civil_social",
    "regional_china",
}
DIRECT_LAW_JOURNAL_TERMS = [
    "law",
    "legal",
    "jurisprudence",
    "court",
    "judicial",
    "justice",
    "human rights",
    "public law",
    "private law",
    "constitutional",
]
LAW_ADJACENT_JOURNAL_TERMS = [
    "governance",
    "regulation",
    "policy",
    "state",
    "institutions",
    "criminology",
    "penology",
]


@dataclass(frozen=True)
class OverexposedJournalRule:
    penalty_multiplier: float
    protected_buckets: tuple[str, ...]
    release_bucket_fit: float = 0.56
    release_scope_fit: float = 0.54
    release_best_article_fit: float = 0.52


OVEREXPOSED_JOURNAL_RULES = {
    normalize_title_key("EUROPEAN JOURNAL OF INTERNATIONAL LAW"): OverexposedJournalRule(
        penalty_multiplier=0.82,
        protected_buckets=("international_comparative_law",),
        release_bucket_fit=0.60,
        release_scope_fit=0.58,
        release_best_article_fit=0.56,
    ),
    normalize_title_key("JOURNAL OF INTERNATIONAL CRIMINAL JUSTICE"): OverexposedJournalRule(
        penalty_multiplier=0.84,
        protected_buckets=("international_comparative_law", "criminal_justice_criminology"),
        release_bucket_fit=0.58,
        release_scope_fit=0.55,
        release_best_article_fit=0.54,
    ),
    normalize_title_key("UNIVERSITY OF PENNSYLVANIA LAW REVIEW"): OverexposedJournalRule(
        penalty_multiplier=0.84,
        protected_buckets=(GENERAL_LAW_REVIEW_BUCKET, "constitutional_human_rights"),
        release_bucket_fit=0.57,
        release_scope_fit=0.56,
        release_best_article_fit=0.54,
    ),
    normalize_title_key("YALE LAW JOURNAL"): OverexposedJournalRule(
        penalty_multiplier=0.84,
        protected_buckets=(GENERAL_LAW_REVIEW_BUCKET, "constitutional_human_rights", "regulation_governance_legislation"),
        release_bucket_fit=0.57,
        release_scope_fit=0.56,
        release_best_article_fit=0.54,
    ),
    normalize_title_key("FEMINIST LEGAL STUDIES"): OverexposedJournalRule(
        penalty_multiplier=0.86,
        protected_buckets=("constitutional_human_rights", "family_labor_social_law", "socio_legal_behavioral"),
        release_bucket_fit=0.55,
        release_scope_fit=0.52,
        release_best_article_fit=0.50,
    ),
}


@dataclass(frozen=True)
class CorpusWeightProfile:
    scope_weight: float = 0.24
    article_corpus_weight: float = 0.33
    best_article_weight: float = 0.28
    keyword_weight: float = 0.15

    def normalized(self) -> "CorpusWeightProfile":
        total = self.scope_weight + self.article_corpus_weight + self.best_article_weight + self.keyword_weight
        if total <= 0:
            return DEFAULT_CORE_WEIGHTS
        return CorpusWeightProfile(
            scope_weight=self.scope_weight / total,
            article_corpus_weight=self.article_corpus_weight / total,
            best_article_weight=self.best_article_weight / total,
            keyword_weight=self.keyword_weight / total,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "scope_weight": self.scope_weight,
            "article_corpus_weight": self.article_corpus_weight,
            "best_article_weight": self.best_article_weight,
            "keyword_weight": self.keyword_weight,
        }


DEFAULT_CORE_WEIGHTS = CorpusWeightProfile(0.24, 0.33, 0.28, 0.15)


def build_core_weight_candidates() -> list[CorpusWeightProfile]:
    unique_profiles: dict[tuple[float, float, float, float], CorpusWeightProfile] = {}
    for scope_weight in (2, 3, 4):
        for article_corpus_weight in (3, 4, 5):
            for best_article_weight in (2, 3, 4):
                for keyword_weight in (1, 2):
                    profile = CorpusWeightProfile(
                        scope_weight=float(scope_weight),
                        article_corpus_weight=float(article_corpus_weight),
                        best_article_weight=float(best_article_weight),
                        keyword_weight=float(keyword_weight),
                    ).normalized()
                    key = tuple(round(value, 4) for value in profile.as_dict().values())
                    unique_profiles[key] = profile
    default_profile = DEFAULT_CORE_WEIGHTS.normalized()
    unique_profiles[tuple(round(value, 4) for value in default_profile.as_dict().values())] = default_profile
    return list(unique_profiles.values())


class CorpusScoringEngine:
    def __init__(
        self,
        taxonomy: TaxonomyProfile,
        *,
        core_weights: CorpusWeightProfile | None = None,
        stage_one_candidate_cap: int | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.core_weights = (core_weights or DEFAULT_CORE_WEIGHTS).normalized()
        self.stage_one_candidate_cap = stage_one_candidate_cap

    def enrich_manuscript(self, manuscript: ManuscriptProfile) -> ManuscriptProfile:
        combined = manuscript.combined_text()
        methodology_scores = self._match_taxonomy_entries(combined, self.taxonomy.methodologies)
        editorial_scores = self._match_taxonomy_entries(combined, self.taxonomy.editorial_signals)
        legal_terms, legal_keyword_score = self._extract_legal_signals(combined)
        law_method_bonus = 0.08 * len(set(methodology_scores).intersection(LAW_METHOD_SIGNAL_NAMES))
        law_editorial_bonus = 0.06 * len(set(editorial_scores).intersection(LAW_EDITORIAL_SIGNAL_NAMES))
        manuscript.methodologies = methodology_scores
        manuscript.editorial_signals = editorial_scores
        manuscript.extracted_terms = list(
            dict.fromkeys(
                [
                    *manuscript.keywords,
                    *extract_candidate_terms(combined, top_k=30),
                ]
            )
        )[:30]
        manuscript.legal_terms = legal_terms
        manuscript.legal_topic_score = clamp(legal_keyword_score + law_method_bonus + law_editorial_bonus)

        manuscript_subfields = build_law_subfield_profile(
            title=manuscript.title,
            keywords=[*manuscript.keywords, *manuscript.extracted_terms],
            text_segments=[manuscript.abstract, manuscript.full_text[:6000]],
        )
        manuscript.subfield_scores = manuscript_subfields.scores
        manuscript.primary_subfield = manuscript_subfields.primary
        manuscript.focus_subfields = list(manuscript_subfields.focus)
        return manuscript

    def score(self, manuscript: ManuscriptProfile, journals: list[JournalProfile]) -> list[RecommendationResult]:
        manuscript = self.enrich_manuscript(manuscript)
        manuscript_subfields = self._manuscript_subfield_profile(manuscript)
        journal_subfield_profiles = {
            self._journal_key(journal): self._journal_subfield_profile(journal)
            for journal in journals
        }

        manuscript_query = self._manuscript_query(manuscript)
        scope_scores = hybrid_similarity_scores(manuscript_query, [self._journal_scope_text(journal) for journal in journals])
        article_corpus_scores = hybrid_similarity_scores(
            manuscript_query,
            [self._journal_article_corpus(journal) for journal in journals],
        )

        signal_cache: list[dict[str, object]] = []
        for journal, scope_fit, article_corpus_fit in zip(journals, scope_scores, article_corpus_scores, strict=False):
            journal_key = self._journal_key(journal)
            journal_subfields = journal_subfield_profiles[journal_key]
            best_article_fit = self._best_article_fit(manuscript_query, journal)
            keyword_fit = self._keyword_fit(manuscript, journal)
            core_fit = self._core_fit(
                scope_fit=scope_fit,
                article_corpus_fit=article_corpus_fit,
                best_article_fit=best_article_fit,
                keyword_fit=keyword_fit,
            )
            signal_cache.append(
                {
                    "journal": journal,
                    "journal_key": journal_key,
                    "journal_subfields": journal_subfields,
                    "scope_fit": scope_fit,
                    "article_corpus_fit": article_corpus_fit,
                    "best_article_fit": best_article_fit,
                    "keyword_fit": keyword_fit,
                    "core_fit": core_fit,
                }
            )

        bucket_stage_scores = self._bucket_stage_scores(signal_cache)
        bucket_confidence = max(bucket_stage_scores.values(), default=0.0)
        stage_focus_buckets = self._focus_stage_buckets(bucket_stage_scores)
        bucket_signal_weight = self._bucket_signal_weight(
            manuscript,
            bucket_confidence=bucket_confidence,
            stage_focus_count=len(stage_focus_buckets),
        )

        recommendations: list[RecommendationResult] = []
        for item in signal_cache:
            journal = item["journal"]
            journal_subfields = item["journal_subfields"]
            scope_fit = float(item["scope_fit"])
            article_corpus_fit = float(item["article_corpus_fit"])
            best_article_fit = float(item["best_article_fit"])
            keyword_fit = float(item["keyword_fit"])
            core_fit = float(item["core_fit"])

            bucket_fit = self._bucket_fit_from_stage_scores(journal_subfields, bucket_stage_scores)
            heuristic_bucket_fit = bucket_similarity(manuscript_subfields, journal_subfields)
            stage_one_score = clamp(
                (0.78 * bucket_fit)
                + (0.22 * heuristic_bucket_fit)
                + (0.04 if journal_subfields.primary and journal_subfields.primary in stage_focus_buckets else 0.0)
            )
            methodology_fit, matched_methodologies = self._preference_fit(
                manuscript.methodologies,
                journal.methodology_preferences,
                neutral_default=0.50,
            )
            editorial_fit, matched_editorials = self._preference_fit(
                manuscript.editorial_signals,
                journal.editorial_preferences,
                neutral_default=0.50,
            )
            venue_quality = self._venue_quality(journal)
            feasibility = self._feasibility(journal)
            legal_alignment = self._legal_alignment(manuscript, journal)

            if manuscript.legal_topic_score >= 0.35:
                legal_weight = 0.12
                bucket_weight = bucket_signal_weight
                core_weight = max(0.0, 1.0 - legal_weight - bucket_weight)
                content_fit = clamp((core_weight * core_fit) + (bucket_weight * bucket_fit) + (legal_weight * legal_alignment))
            else:
                bucket_weight = min(bucket_signal_weight, 0.06)
                content_fit = clamp(((1.0 - bucket_weight) * core_fit) + (bucket_weight * bucket_fit))

            stage_two_score = clamp(
                (0.82 * content_fit)
                + (0.05 * methodology_fit)
                + (0.03 * editorial_fit)
                + (0.06 * venue_quality)
                + (0.04 * feasibility)
            )
            overall_score = clamp((0.94 * stage_two_score) + (0.06 * stage_one_score))
            match_probability = clamp((0.92 * content_fit) + (0.04 * stage_one_score) + (0.02 * methodology_fit) + (0.02 * editorial_fit))

            overall_score = self._apply_bucket_adjustments(
                manuscript_subfields=manuscript_subfields,
                journal_subfields=journal_subfields,
                bucket_fit=bucket_fit,
                bucket_confidence=bucket_confidence,
                stage_focus_buckets=stage_focus_buckets,
                base_score=overall_score,
            )
            match_probability = self._apply_bucket_adjustments(
                manuscript_subfields=manuscript_subfields,
                journal_subfields=journal_subfields,
                bucket_fit=bucket_fit,
                bucket_confidence=bucket_confidence,
                stage_focus_buckets=stage_focus_buckets,
                base_score=match_probability,
            )
            if manuscript.legal_topic_score >= 0.55:
                law_multiplier = 0.86 + (0.14 * legal_alignment)
                overall_score = clamp(overall_score * law_multiplier)
                match_probability = clamp(match_probability * law_multiplier)

            overexposure_penalty, overexposure_reason = self._overexposure_penalty(
                manuscript_subfields=manuscript_subfields,
                journal=journal,
                journal_subfields=journal_subfields,
                bucket_fit=bucket_fit,
                bucket_confidence=bucket_confidence,
                stage_focus_buckets=stage_focus_buckets,
                scope_fit=scope_fit,
                article_corpus_fit=article_corpus_fit,
                best_article_fit=best_article_fit,
                legal_alignment=legal_alignment,
            )
            if overexposure_penalty < 1.0:
                overall_score = clamp(overall_score * overexposure_penalty)
                match_probability = clamp(match_probability * overexposure_penalty)

            rationale = self._build_rationale(
                manuscript=manuscript,
                journal=journal,
                manuscript_subfields=manuscript_subfields,
                journal_subfields=journal_subfields,
                bucket_fit=bucket_fit,
                scope_fit=scope_fit,
                article_corpus_fit=article_corpus_fit,
                best_article_fit=best_article_fit,
                keyword_fit=keyword_fit,
                methodology_fit=methodology_fit,
                editorial_fit=editorial_fit,
                legal_alignment=legal_alignment,
                matched_methodologies=matched_methodologies,
                matched_editorials=matched_editorials,
            )
            if overexposure_reason:
                rationale = f"{rationale}; overexposure control: {overexposure_reason}"

            recommendations.append(
                RecommendationResult(
                    journal=journal,
                    content_fit=content_fit,
                    bucket_fit=bucket_fit,
                    scope_fit=scope_fit,
                    article_corpus_fit=article_corpus_fit,
                    best_article_fit=best_article_fit,
                    keyword_fit=keyword_fit,
                    methodology_fit=methodology_fit,
                    editorial_fit=editorial_fit,
                    venue_quality=venue_quality,
                    feasibility=feasibility,
                    overall_score=overall_score,
                    match_probability=match_probability,
                    overexposure_penalty=overexposure_penalty,
                    overexposure_penalty_reason=overexposure_reason,
                    match_level=self._match_level(match_probability),
                    rationale=rationale,
                    manuscript_primary_subfield=bucket_label(manuscript_subfields.primary) or None,
                    journal_primary_subfield=bucket_label(journal_subfields.primary) or None,
                    matched_methodologies=matched_methodologies,
                    matched_editorial_signals=matched_editorials,
                )
            )
        recommendations.sort(key=lambda item: (-item.overall_score, -item.match_probability, item.journal.title.lower()))
        return recommendations

    def _bucket_stage_scores(self, signal_cache: list[dict[str, object]]) -> dict[str, float]:
        bucket_values: dict[str, list[float]] = {}
        for item in signal_cache:
            journal_subfields = item["journal_subfields"]
            core_fit = float(item["core_fit"])
            for bucket in journal_subfields.focus:
                bucket_values.setdefault(bucket, []).append(core_fit)
        stage_scores: dict[str, float] = {}
        for bucket, values in bucket_values.items():
            stage_scores[bucket] = clamp((0.60 * top_k_mean(values, k=5, default=0.0)) + (0.40 * max(values, default=0.0)))
        return stage_scores

    def _focus_stage_buckets(self, stage_scores: dict[str, float]) -> set[str]:
        if not stage_scores:
            return set()
        top_score = max(stage_scores.values())
        threshold = max(0.32, top_score * 0.86)
        return {bucket for bucket, score in stage_scores.items() if score >= threshold}

    def _bucket_fit_from_stage_scores(self, journal_subfields: SubfieldProfile, stage_scores: dict[str, float]) -> float:
        if not journal_subfields.focus:
            return 0.0
        stage_fit = max((stage_scores.get(bucket, 0.0) for bucket in journal_subfields.focus), default=0.0)
        if journal_subfields.primary == GENERAL_LAW_REVIEW_BUCKET:
            stage_fit *= 0.97
        return clamp(stage_fit)

    def _apply_bucket_adjustments(
        self,
        *,
        manuscript_subfields: SubfieldProfile,
        journal_subfields: SubfieldProfile,
        bucket_fit: float,
        bucket_confidence: float,
        stage_focus_buckets: set[str],
        base_score: float,
    ) -> float:
        adjusted = base_score
        if bucket_confidence >= 0.44 and journal_subfields.primary and journal_subfields.primary in stage_focus_buckets:
            adjusted *= 1.01
        if (
            bucket_confidence >= 0.52
            and manuscript_subfields.primary
            and manuscript_subfields.primary != GENERAL_LAW_REVIEW_BUCKET
            and journal_subfields.primary == GENERAL_LAW_REVIEW_BUCKET
            and bucket_fit < 0.28
        ):
            adjusted *= 0.97
        return clamp(adjusted)

    def _match_taxonomy_entries(self, text: str, entries: list[TaxonomyEntry]) -> dict[str, float]:
        lowered = normalize_space(text).lower()
        scores: dict[str, float] = {}
        for entry in entries:
            hits = 0
            for alias in entry.aliases:
                if normalize_space(alias).lower() in lowered:
                    hits += 1
            if hits:
                scores[entry.name] = clamp(0.35 + (0.2 * hits), 0.0, 1.0)
        return scores

    def _manuscript_query(self, manuscript: ManuscriptProfile) -> str:
        return "\n".join(
            [
                manuscript.title,
                manuscript.abstract,
                " ".join(manuscript.keywords),
                " ".join(manuscript.extracted_terms[:20]),
                manuscript.full_text[:6000],
            ]
        )

    def _journal_scope_text(self, journal: JournalProfile) -> str:
        return "\n".join(
            [
                journal.title,
                journal.aims_and_scope,
                " ".join(journal.subdisciplines),
                " ".join(journal.keywords),
            ]
        )

    def _journal_article_corpus(self, journal: JournalProfile) -> str:
        article_text = " ".join(self._article_text(article) for article in journal.recent_articles)
        fallback = " ".join([journal.title, journal.aims_and_scope, " ".join(journal.keywords)])
        return article_text or fallback

    def _article_text(self, article) -> str:
        return "\n".join(
            [
                article.title,
                " ".join(article.keywords),
                article.abstract_snippet,
            ]
        )

    def _best_article_fit(self, manuscript_query: str, journal: JournalProfile) -> float:
        article_texts = [self._article_text(article) for article in journal.recent_articles if normalize_space(self._article_text(article))]
        if not article_texts:
            return 0.0
        similarities = hybrid_similarity_scores(manuscript_query, article_texts)
        return clamp(top_k_mean(similarities, k=2, default=0.0))

    def _keyword_fit(self, manuscript: ManuscriptProfile, journal: JournalProfile) -> float:
        flattened_terms: list[str] = [*journal.keywords, *journal.subdisciplines]
        for article in journal.recent_articles:
            flattened_terms.extend(article.keywords)
            flattened_terms.extend(extract_candidate_terms(article.title, top_k=6))
        flattened_terms = list(dict.fromkeys(term for term in flattened_terms if normalize_space(term)))
        manuscript_terms = [*manuscript.keywords, *manuscript.extracted_terms, *manuscript.legal_terms]
        return keyword_overlap(manuscript_terms, flattened_terms)

    def _core_fit(
        self,
        *,
        scope_fit: float,
        article_corpus_fit: float,
        best_article_fit: float,
        keyword_fit: float,
    ) -> float:
        return clamp(
            (self.core_weights.scope_weight * scope_fit)
            + (self.core_weights.article_corpus_weight * article_corpus_fit)
            + (self.core_weights.best_article_weight * best_article_fit)
            + (self.core_weights.keyword_weight * keyword_fit)
        )

    def _preference_fit(
        self,
        manuscript_scores: dict[str, float],
        preferences: list[str],
        *,
        neutral_default: float,
    ) -> tuple[float, list[str]]:
        if not preferences:
            return neutral_default, []
        matched = [pref for pref in preferences if manuscript_scores.get(pref, 0.0) > 0.0]
        if not matched:
            return 0.22, []
        scores = [manuscript_scores[pref] for pref in matched]
        return clamp(safe_mean(scores, default=0.0)), matched

    def _venue_quality(self, journal: JournalProfile) -> float:
        quartile_score = QUARTILE_WEIGHTS.get((journal.jcr_quartile or "").upper()) if journal.jcr_quartile else None
        impact_score = None
        if journal.impact_factor is not None:
            impact_score = clamp(journal.impact_factor / 10.0)
        indexing_score = None
        if journal.indexing:
            weights = [INDEXING_WEIGHTS[item.lower()] for item in journal.indexing if item.lower() in INDEXING_WEIGHTS]
            if weights:
                indexing_score = max(weights)
        scores = [score for score in [quartile_score, impact_score, indexing_score] if score is not None]
        return clamp(safe_mean(scores, default=0.50))

    def _feasibility(self, journal: JournalProfile) -> float:
        acceptance_score = journal.acceptance_rate if journal.acceptance_rate is not None else 0.45
        cycle_score = 0.50
        if journal.review_cycle_months is not None:
            cycle_score = clamp(1.0 - (journal.review_cycle_months / 12.0), 0.10, 1.0)
        return clamp((0.62 * acceptance_score) + (0.38 * cycle_score))

    def _build_rationale(
        self,
        *,
        manuscript: ManuscriptProfile,
        journal: JournalProfile,
        manuscript_subfields: SubfieldProfile,
        journal_subfields: SubfieldProfile,
        bucket_fit: float,
        scope_fit: float,
        article_corpus_fit: float,
        best_article_fit: float,
        keyword_fit: float,
        methodology_fit: float,
        editorial_fit: float,
        legal_alignment: float,
        matched_methodologies: list[str],
        matched_editorials: list[str],
    ) -> str:
        reasons: list[str] = []
        if manuscript_subfields.primary and journal_subfields.primary and bucket_fit >= 0.48:
            reasons.append(f"subfield bucket match: {bucket_label(journal_subfields.primary)}")
        if scope_fit >= 0.60:
            reasons.append("aims and scope fit is strong")
        elif scope_fit >= 0.45:
            reasons.append("aims and scope fit is moderate")
        if article_corpus_fit >= 0.60:
            reasons.append("recent published article corpus is highly similar")
        elif best_article_fit >= 0.58:
            reasons.append("one or more recent published articles are closely aligned")
        if keyword_fit >= 0.25:
            reasons.append("keyword overlap with recent articles is strong")
        if manuscript.legal_topic_score >= 0.55:
            if legal_alignment >= 0.70:
                reasons.append("journal remains strongly law-aligned")
            elif legal_alignment < 0.40:
                reasons.append("journal is not strongly law-aligned for this manuscript")
        if matched_methodologies:
            reasons.append(f"method fit: {', '.join(matched_methodologies)}")
        elif methodology_fit < 0.30:
            reasons.append("method profile overlap is limited")
        if matched_editorials:
            reasons.append(f"editorial fit: {', '.join(matched_editorials)}")
        elif editorial_fit < 0.30:
            reasons.append("editorial emphasis overlap is limited")
        if journal.jcr_quartile:
            reasons.append(f"JCR {journal.jcr_quartile}")
        return "; ".join(reasons) if reasons else "journal selected through bucket-first reranking and article corpus similarity"

    def _match_level(self, probability: float) -> str:
        if probability >= 0.80:
            return "High"
        if probability >= 0.65:
            return "Good"
        if probability >= 0.50:
            return "Medium"
        return "Low"

    def _extract_legal_signals(self, text: str) -> tuple[list[str], float]:
        lowered = normalize_space(text).lower()
        matched_terms: list[str] = []
        weighted_score = 0.0
        for group in LEGAL_SIGNAL_GROUPS:
            if any(alias.lower() in lowered for alias in group["aliases"]):
                matched_terms.append(group["name"])
                weighted_score += group["weight"]
        return matched_terms, clamp(weighted_score)

    def _journal_legal_terms(self, journal: JournalProfile) -> list[str]:
        source_text = " ".join(
            [
                journal.title,
                journal.discipline,
                " ".join(journal.subdisciplines),
                " ".join(journal.keywords),
                journal.aims_and_scope,
            ]
        ).lower()
        matched_terms: list[str] = []
        for term in [*DIRECT_LAW_JOURNAL_TERMS, *LAW_ADJACENT_JOURNAL_TERMS]:
            if term in source_text:
                matched_terms.append(term)
        return list(dict.fromkeys(matched_terms))

    def _journal_legal_affinity(self, journal: JournalProfile) -> float:
        journal_text = " ".join(
            [
                journal.title,
                journal.discipline,
                " ".join(journal.subdisciplines),
                " ".join(journal.keywords),
                journal.aims_and_scope,
            ]
        ).lower()
        direct_hits = sum(1 for term in DIRECT_LAW_JOURNAL_TERMS if term in journal_text)
        adjacent_hits = sum(1 for term in LAW_ADJACENT_JOURNAL_TERMS if term in journal_text)
        if direct_hits:
            return clamp(0.72 + (0.10 * min(direct_hits, 3)))
        if adjacent_hits:
            return clamp(0.42 + (0.08 * min(adjacent_hits, 3)))
        return 0.18

    def _legal_alignment(self, manuscript: ManuscriptProfile, journal: JournalProfile) -> float:
        if manuscript.legal_topic_score < 0.20:
            return 0.50
        journal_affinity = self._journal_legal_affinity(journal)
        legal_overlap = keyword_overlap(manuscript.legal_terms, self._journal_legal_terms(journal))
        return clamp((0.65 * journal_affinity) + (0.35 * legal_overlap))

    def _manuscript_subfield_profile(self, manuscript: ManuscriptProfile) -> SubfieldProfile:
        return SubfieldProfile(
            scores=manuscript.subfield_scores,
            primary=manuscript.primary_subfield,
            focus=tuple(manuscript.focus_subfields),
        )

    def _journal_subfield_profile(self, journal: JournalProfile) -> SubfieldProfile:
        article_keywords = [keyword for article in journal.recent_articles for keyword in article.keywords]
        article_titles = [article.title for article in journal.recent_articles]
        article_abstracts = [article.abstract_snippet[:1200] for article in journal.recent_articles if normalize_space(article.abstract_snippet)]
        return build_law_subfield_profile(
            title=journal.title,
            keywords=[*journal.keywords, *journal.subdisciplines, *article_keywords, *extract_candidate_terms(" ".join(article_titles), top_k=20)],
            text_segments=[journal.aims_and_scope, *article_titles, *article_abstracts],
            subdisciplines=journal.subdisciplines,
        )

    def _journal_key(self, journal: JournalProfile) -> str:
        return journal.journal_id or journal.title.lower()

    def _bucket_signal_weight(
        self,
        manuscript: ManuscriptProfile,
        *,
        bucket_confidence: float,
        stage_focus_count: int,
    ) -> float:
        if not manuscript.primary_subfield or manuscript.primary_subfield == GENERAL_LAW_REVIEW_BUCKET:
            return 0.03
        if bucket_confidence >= 0.52 and stage_focus_count <= 2:
            return 0.08
        if bucket_confidence >= 0.42:
            return 0.05
        return 0.03

    def _overexposure_penalty(
        self,
        *,
        manuscript_subfields: SubfieldProfile,
        journal: JournalProfile,
        journal_subfields: SubfieldProfile,
        bucket_fit: float,
        bucket_confidence: float,
        stage_focus_buckets: set[str],
        scope_fit: float,
        article_corpus_fit: float,
        best_article_fit: float,
        legal_alignment: float,
    ) -> tuple[float, str | None]:
        rule = OVEREXPOSED_JOURNAL_RULES.get(normalize_title_key(journal.title))
        if rule is None:
            return 1.0, None

        manuscript_primary = manuscript_subfields.primary
        journal_primary = journal_subfields.primary
        protected_match = (
            manuscript_primary is not None
            and manuscript_primary in rule.protected_buckets
            and (
                journal_primary == manuscript_primary
                or manuscript_primary in journal_subfields.focus
                or bucket_fit >= 0.42
            )
        )
        strong_release = (
            bucket_fit >= rule.release_bucket_fit
            and (
                scope_fit >= rule.release_scope_fit
                or best_article_fit >= rule.release_best_article_fit
                or article_corpus_fit >= (rule.release_scope_fit - 0.02)
            )
        )
        if protected_match or strong_release:
            return 1.0, None

        penalty = rule.penalty_multiplier
        reasons: list[str] = []

        if manuscript_primary and manuscript_primary not in rule.protected_buckets:
            penalty *= 0.95
            reasons.append("primary subfield mismatch")
        if stage_focus_buckets and journal_primary and journal_primary not in stage_focus_buckets and bucket_confidence >= 0.46:
            penalty *= 0.96
            reasons.append("outside stage-one focus")
        if legal_alignment < 0.66:
            penalty *= 0.97
        if scope_fit < rule.release_scope_fit and best_article_fit < rule.release_best_article_fit:
            penalty *= 0.97
        if journal_primary == GENERAL_LAW_REVIEW_BUCKET and manuscript_primary and manuscript_primary != GENERAL_LAW_REVIEW_BUCKET and bucket_fit < 0.38:
            penalty *= 0.95
            reasons.append("general law review overreach")
        if (
            journal_primary == "international_comparative_law"
            and manuscript_primary
            and manuscript_primary != "international_comparative_law"
            and bucket_fit < 0.46
        ):
            penalty *= 0.95
            reasons.append("international-law drift")

        penalty = clamp(penalty, 0.68, 1.0)
        if penalty >= 0.995:
            return 1.0, None
        reason_text = ", ".join(reasons) if reasons else "historically over-predicted venue"
        return penalty, reason_text
