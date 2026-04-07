from __future__ import annotations

from dataclasses import dataclass

from journal_agent.models.schemas import JournalProfile, ManuscriptProfile, RecommendationResult, TaxonomyEntry, TaxonomyProfile
from journal_agent.utils.text_processing import clamp, cosine_similarity_scores, extract_candidate_terms, keyword_overlap, normalize_space, safe_mean


INDEXING_WEIGHTS = {
    "jcr": 0.90,
    "ssci": 0.95,
    "sci": 0.95,
    "cssci": 0.85,
    "北大核心": 0.75,
    "scopus": 0.72,
}

QUARTILE_WEIGHTS = {
    "Q1": 1.00,
    "Q2": 0.82,
    "Q3": 0.64,
    "Q4": 0.48,
}

CONCEPT_ALIASES = {
    "platform_governance": ["platform governance", "online platform", "平台治理", "平台规制", "平台监管"],
    "algorithmic_regulation": ["algorithm", "algorithmic", "算法", "自动化决策", "算法治理"],
    "due_process": ["due process", "procedural fairness", "procedural justice", "正当程序", "程序保障"],
    "digital_regulation": ["digital regulation", "digital governance", "数字监管", "数字治理", "网络治理"],
    "comparative_law": ["comparative law", "cross-jurisdiction", "比较法", "域外法", "比较研究"],
    "artificial_intelligence": ["artificial intelligence", "ai", "人工智能"],
    "data_protection": ["data protection", "data compliance", "数据保护", "数据合规", "个人信息保护"],
    "human_rights": ["human rights", "rights-based", "人权", "基本权利"],
    "judicial_process": ["judicial", "court", "litigation", "法院", "司法", "裁判", "诉讼"],
    "regulation_policy": ["regulation", "policy", "legislation", "规制", "政策", "立法"],
}


@dataclass
class ManuscriptSignalBundle:
    methodologies: dict[str, float]
    editorial_signals: dict[str, float]
    extracted_terms: list[str]


class HeuristicScoringEngine:
    def __init__(self, taxonomy: TaxonomyProfile) -> None:
        self.taxonomy = taxonomy

    def enrich_manuscript(self, manuscript: ManuscriptProfile) -> ManuscriptProfile:
        combined = manuscript.combined_text()
        methodology_scores = self._match_taxonomy_entries(combined, self.taxonomy.methodologies)
        editorial_scores = self._match_taxonomy_entries(combined, self.taxonomy.editorial_signals)
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
        return manuscript

    def score(self, manuscript: ManuscriptProfile, journals: list[JournalProfile]) -> list[RecommendationResult]:
        manuscript = self.enrich_manuscript(manuscript)
        journal_corpora = [self._journal_corpus(journal) for journal in journals]
        similarity_scores = cosine_similarity_scores(manuscript.combined_text(), journal_corpora)
        recommendations: list[RecommendationResult] = []
        for journal, similarity in zip(journals, similarity_scores, strict=False):
            content_fit = self._content_fit(manuscript, journal, similarity)
            methodology_fit, matched_methodologies = self._preference_fit(
                manuscript.methodologies,
                journal.methodology_preferences,
                neutral_default=0.58,
            )
            editorial_fit, matched_editorials = self._preference_fit(
                manuscript.editorial_signals,
                journal.editorial_preferences,
                neutral_default=0.56,
            )
            venue_quality = self._venue_quality(journal)
            feasibility = self._feasibility(journal)
            fit_total = 0.55 * content_fit + 0.25 * methodology_fit + 0.20 * editorial_fit
            overall_score = clamp(0.72 * fit_total + 0.16 * venue_quality + 0.12 * feasibility)
            match_probability = clamp(0.82 * fit_total + 0.18 * feasibility)
            recommendations.append(
                RecommendationResult(
                    journal=journal,
                    content_fit=content_fit,
                    methodology_fit=methodology_fit,
                    editorial_fit=editorial_fit,
                    venue_quality=venue_quality,
                    feasibility=feasibility,
                    overall_score=overall_score,
                    match_probability=match_probability,
                    match_level=self._match_level(match_probability),
                    rationale=self._build_rationale(
                        journal,
                        content_fit,
                        methodology_fit,
                        editorial_fit,
                        matched_methodologies,
                        matched_editorials,
                    ),
                    matched_methodologies=matched_methodologies,
                    matched_editorial_signals=matched_editorials,
                )
            )
        recommendations.sort(key=lambda item: (-item.overall_score, -item.match_probability, item.journal.title.lower()))
        return recommendations

    def _match_taxonomy_entries(self, text: str, entries: list[TaxonomyEntry]) -> dict[str, float]:
        lowered = normalize_space(text).lower()
        scores: dict[str, float] = {}
        for entry in entries:
            hits = 0
            for alias in entry.aliases:
                if normalize_space(alias).lower() in lowered:
                    hits += 1
            if hits:
                scores[entry.name] = clamp(0.35 + 0.2 * hits, 0.0, 1.0)
        return scores

    def _journal_corpus(self, journal: JournalProfile) -> str:
        recent_articles = " ".join(
            f"{article.title} {' '.join(article.keywords)} {article.abstract_snippet}"
            for article in journal.recent_articles
        )
        return "\n".join(
            [
                journal.title,
                journal.aims_and_scope,
                " ".join(journal.subdisciplines),
                " ".join(journal.keywords),
                " ".join(journal.methodology_preferences),
                " ".join(journal.editorial_preferences),
                recent_articles,
            ]
        )

    def _content_fit(self, manuscript: ManuscriptProfile, journal: JournalProfile, similarity: float) -> float:
        journal_terms = list(
            dict.fromkeys(
                [
                    *journal.keywords,
                    *journal.subdisciplines,
                    *(article.title for article in journal.recent_articles),
                    *(keyword for article in journal.recent_articles for keyword in article.keywords),
                ]
            )
        )
        overlap = keyword_overlap(manuscript.keywords + manuscript.extracted_terms, journal_terms)
        concept_overlap = keyword_overlap(
            self._extract_concepts(manuscript.combined_text()),
            self._extract_concepts(self._journal_corpus(journal)),
        )
        return clamp(0.50 * similarity + 0.20 * overlap + 0.30 * concept_overlap)

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
            return 0.18, []
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
        return clamp(0.62 * acceptance_score + 0.38 * cycle_score)

    def _build_rationale(
        self,
        journal: JournalProfile,
        content_fit: float,
        methodology_fit: float,
        editorial_fit: float,
        matched_methodologies: list[str],
        matched_editorials: list[str],
    ) -> str:
        reasons: list[str] = []
        if content_fit >= 0.68:
            reasons.append("scope and article topics are strongly aligned")
        elif content_fit >= 0.50:
            reasons.append("scope alignment is moderate")
        if matched_methodologies:
            reasons.append(f"method fit: {', '.join(matched_methodologies)}")
        elif methodology_fit < 0.30:
            reasons.append("method profile is not a strong match")
        if matched_editorials:
            reasons.append(f"editorial preference fit: {', '.join(matched_editorials)}")
        elif editorial_fit < 0.30:
            reasons.append("editorial emphasis overlap is limited")
        if journal.jcr_quartile:
            reasons.append(f"JCR {journal.jcr_quartile}")
        if journal.review_cycle_months is not None:
            reasons.append(f"review cycle about {journal.review_cycle_months:g} months")
        return "; ".join(reasons) if reasons else "general scope fit based recommendation"

    def _match_level(self, probability: float) -> str:
        if probability >= 0.80:
            return "High"
        if probability >= 0.65:
            return "Good"
        if probability >= 0.50:
            return "Medium"
        return "Low"

    def _extract_concepts(self, text: str) -> list[str]:
        lowered = normalize_space(text).lower()
        concepts: list[str] = []
        for concept, aliases in CONCEPT_ALIASES.items():
            if any(alias.lower() in lowered for alias in aliases):
                concepts.append(concept)
        return concepts
