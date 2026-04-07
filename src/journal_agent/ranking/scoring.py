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

LEGAL_SIGNAL_GROUPS = [
    {
        "name": "legal_core",
        "weight": 0.26,
        "aliases": [
            "law",
            "legal",
            "jurisprudence",
            "juridical",
            "\u6cd5\u5b66",
            "\u6cd5\u5f8b",
            "\u6cd5\u7406",
            "\u6cd5\u6cbb",
        ],
    },
    {
        "name": "judicial_process",
        "weight": 0.18,
        "aliases": [
            "court",
            "judicial",
            "litigation",
            "adjudication",
            "due process",
            "tribunal",
            "\u6cd5\u9662",
            "\u53f8\u6cd5",
            "\u8bc9\u8bbc",
            "\u88c1\u5224",
            "\u6b63\u5f53\u7a0b\u5e8f",
        ],
    },
    {
        "name": "legislation_regulation",
        "weight": 0.16,
        "aliases": [
            "legislation",
            "statute",
            "regulation",
            "regulatory",
            "compliance",
            "\u7acb\u6cd5",
            "\u89c4\u5236",
            "\u76d1\u7ba1",
            "\u6cd5\u89c4",
            "\u5408\u89c4",
        ],
    },
    {
        "name": "rights_public_law",
        "weight": 0.14,
        "aliases": [
            "constitutional",
            "human rights",
            "administrative law",
            "public law",
            "\u5baa\u6cd5",
            "\u4eba\u6743",
            "\u6743\u5229",
            "\u884c\u653f\u6cd5",
            "\u516c\u6cd5",
        ],
    },
    {
        "name": "private_commercial_law",
        "weight": 0.14,
        "aliases": [
            "civil law",
            "contract",
            "corporate law",
            "commercial law",
            "private law",
            "\u6c11\u6cd5",
            "\u5408\u540c",
            "\u516c\u53f8\u6cd5",
            "\u5546\u6cd5",
            "\u79c1\u6cd5",
        ],
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
            "\u5211\u6cd5",
            "\u72af\u7f6a",
            "\u5211\u4e8b",
            "\u6cbb\u5b89",
        ],
    },
    {
        "name": "comparative_international_law",
        "weight": 0.12,
        "aliases": [
            "comparative law",
            "international law",
            "transnational law",
            "\u6bd4\u8f83\u6cd5",
            "\u56fd\u9645\u6cd5",
            "\u8de8\u5883",
            "\u57df\u5916\u6cd5",
        ],
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
            "\u5e73\u53f0\u6cbb\u7406",
            "\u7b97\u6cd5\u6cbb\u7406",
            "\u6570\u636e\u4fdd\u62a4",
            "\u7f51\u7edc\u6cd5",
            "\u4eba\u5de5\u667a\u80fd\u76d1\u7ba1",
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
    "criminology",
    "penology",
    "human rights",
    "public law",
    "private law",
    "constitutional",
]
LAW_ADJACENT_JOURNAL_TERMS = [
    "political science",
    "public administration",
    "international relations",
    "governance",
    "regulation",
    "policy",
    "state",
    "institutions",
]

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
            legal_alignment = self._legal_alignment(manuscript, journal)
            if manuscript.legal_topic_score >= 0.35:
                fit_total = 0.45 * content_fit + 0.22 * methodology_fit + 0.15 * editorial_fit + 0.18 * legal_alignment
            else:
                fit_total = 0.55 * content_fit + 0.25 * methodology_fit + 0.20 * editorial_fit
            overall_score = clamp(0.72 * fit_total + 0.16 * venue_quality + 0.12 * feasibility)
            match_probability = clamp(0.82 * fit_total + 0.18 * feasibility)
            if manuscript.legal_topic_score >= 0.55:
                law_multiplier = 0.75 + 0.35 * legal_alignment
                overall_score = clamp(overall_score * law_multiplier)
                match_probability = clamp(match_probability * law_multiplier)
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
                        manuscript,
                        journal,
                        content_fit,
                        methodology_fit,
                        editorial_fit,
                        legal_alignment,
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
            f"{article.title} {' '.join(article.keywords)} {article.abstract_snippet} {article.full_text}"
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
        article_section_fit = self._article_section_fit(manuscript, journal)
        reference_fit = self._reference_similarity_fit(manuscript, journal)
        weighted_parts = [(similarity, 0.35), (overlap, 0.15), (concept_overlap, 0.20), (article_section_fit, 0.20)]
        if manuscript.legal_topic_score >= 0.35:
            legal_overlap = keyword_overlap(manuscript.legal_terms, self._journal_legal_terms(journal))
            weighted_parts.append((legal_overlap, 0.10))
        if reference_fit is not None:
            weighted_parts.append((reference_fit, 0.10))
        total_weight = sum(weight for _, weight in weighted_parts)
        if not total_weight:
            return 0.0
        blended = sum(score * weight for score, weight in weighted_parts) / total_weight
        return clamp(blended)

    def _article_section_fit(self, manuscript: ManuscriptProfile, journal: JournalProfile) -> float:
        if not journal.recent_articles:
            return 0.0
        article_scores: list[float] = []
        for article in journal.recent_articles:
            section_scores: list[tuple[float, float]] = [
                (cosine_similarity_scores(manuscript.title, [article.title])[0], 0.25),
                (cosine_similarity_scores(manuscript.abstract, [article.abstract_snippet])[0], 0.25),
                (keyword_overlap(manuscript.keywords, article.keywords), 0.20),
            ]
            has_oa_full_text = article.is_oa is not False and normalize_space(article.full_text)
            if has_oa_full_text:
                section_scores.append((cosine_similarity_scores(manuscript.full_text, [article.full_text])[0], 0.30))
            weight_sum = sum(weight for _, weight in section_scores)
            if not weight_sum:
                continue
            article_score = sum(score * weight for score, weight in section_scores) / weight_sum
            article_scores.append(article_score)
        return clamp(max(article_scores, default=0.0))

    def _reference_similarity_fit(self, manuscript: ManuscriptProfile, journal: JournalProfile) -> float | None:
        if not normalize_space(manuscript.references_text):
            return None
        article_references = [
            normalize_space(article.references_text)
            for article in journal.recent_articles
            if normalize_space(article.references_text)
        ]
        if not article_references:
            return None
        similarity_scores = cosine_similarity_scores(manuscript.references_text, article_references)
        return clamp(max(similarity_scores, default=0.0))

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
        manuscript: ManuscriptProfile,
        journal: JournalProfile,
        content_fit: float,
        methodology_fit: float,
        editorial_fit: float,
        legal_alignment: float,
        matched_methodologies: list[str],
        matched_editorials: list[str],
    ) -> str:
        reasons: list[str] = []
        if content_fit >= 0.68:
            reasons.append("scope and article topics are strongly aligned")
        elif content_fit >= 0.50:
            reasons.append("scope alignment is moderate")
        if manuscript.legal_topic_score >= 0.55:
            if legal_alignment >= 0.70:
                reasons.append("legal-profile alignment is strong")
            elif legal_alignment < 0.40:
                reasons.append("journal is weakly aligned with a law-focused manuscript")
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
            return clamp(0.72 + 0.10 * min(direct_hits, 3))
        if adjacent_hits:
            return clamp(0.42 + 0.08 * min(adjacent_hits, 3))
        return 0.18

    def _legal_alignment(self, manuscript: ManuscriptProfile, journal: JournalProfile) -> float:
        if manuscript.legal_topic_score < 0.20:
            return 0.50
        journal_affinity = self._journal_legal_affinity(journal)
        legal_overlap = keyword_overlap(manuscript.legal_terms, self._journal_legal_terms(journal))
        return clamp(0.65 * journal_affinity + 0.35 * legal_overlap)
