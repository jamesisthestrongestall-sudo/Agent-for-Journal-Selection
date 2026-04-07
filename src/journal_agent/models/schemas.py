from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JournalArticleExample(BaseModel):
    title: str
    keywords: list[str] = Field(default_factory=list)
    abstract_snippet: str = ""
    full_text: str = ""
    references_text: str = ""
    is_oa: bool | None = None


class JournalProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    journal_id: str | None = None
    title: str
    issn: str | None = None
    eissn: str | None = None
    discipline: str = "law"
    subdisciplines: list[str] = Field(default_factory=list)
    language: str | None = None
    country: str | None = None
    website: str | None = None
    submission_url: str | None = None
    publisher: str | None = None
    aims_and_scope: str = ""
    keywords: list[str] = Field(default_factory=list)
    methodology_preferences: list[str] = Field(default_factory=list)
    editorial_preferences: list[str] = Field(default_factory=list)
    indexing: list[str] = Field(default_factory=list)
    recent_articles: list[JournalArticleExample] = Field(default_factory=list)
    jcr_quartile: str | None = None
    impact_factor: float | None = None
    review_cycle_months: float | None = None
    acceptance_rate: float | None = None
    source_tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class ManuscriptProfile(BaseModel):
    title: str
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    language: str = "unknown"
    full_text: str = ""
    references_text: str = ""
    footnotes_text: str = ""
    discipline: str = "law"
    methodologies: dict[str, float] = Field(default_factory=dict)
    editorial_signals: dict[str, float] = Field(default_factory=dict)
    extracted_terms: list[str] = Field(default_factory=list)

    def combined_text(self) -> str:
        return "\n".join(
            [
                self.title,
                self.abstract,
                " ".join(self.keywords),
                self.full_text,
            ]
        ).strip()


class TaxonomyEntry(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)


class TaxonomyProfile(BaseModel):
    discipline: str
    methodologies: list[TaxonomyEntry] = Field(default_factory=list)
    editorial_signals: list[TaxonomyEntry] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    journal: JournalProfile
    content_fit: float
    methodology_fit: float
    editorial_fit: float
    venue_quality: float
    feasibility: float
    overall_score: float
    match_probability: float
    match_level: str
    rationale: str
    matched_methodologies: list[str] = Field(default_factory=list)
    matched_editorial_signals: list[str] = Field(default_factory=list)

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "journal_name": self.journal.title,
            "issn": self.journal.issn or "",
            "eissn": self.journal.eissn or "",
            "journal_language": self.journal.language or "",
            "website": self.journal.website or "",
            "publisher": self.journal.publisher or "",
            "discipline": self.journal.discipline,
            "match_level": self.match_level,
            "overall_score": round(self.overall_score * 100, 2),
            "match_probability": round(self.match_probability * 100, 2),
            "content_fit": round(self.content_fit * 100, 2),
            "methodology_fit": round(self.methodology_fit * 100, 2),
            "editorial_fit": round(self.editorial_fit * 100, 2),
            "venue_quality": round(self.venue_quality * 100, 2),
            "feasibility": round(self.feasibility * 100, 2),
            "jcr_quartile": self.journal.jcr_quartile or "",
            "impact_factor": self.journal.impact_factor if self.journal.impact_factor is not None else "",
            "review_cycle_months": self.journal.review_cycle_months if self.journal.review_cycle_months is not None else "",
            "acceptance_rate": round(self.journal.acceptance_rate * 100, 2) if self.journal.acceptance_rate is not None else "",
            "indexing": "; ".join(self.journal.indexing),
            "methodology_preferences": "; ".join(self.journal.methodology_preferences),
            "editorial_preferences": "; ".join(self.journal.editorial_preferences),
            "matched_methodologies": "; ".join(self.matched_methodologies),
            "matched_editorial_signals": "; ".join(self.matched_editorial_signals),
            "submission_url": self.journal.submission_url or "",
            "notes": self.journal.notes or "",
            "rationale": self.rationale,
        }
