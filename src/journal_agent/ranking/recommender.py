from __future__ import annotations

import csv
from pathlib import Path

from journal_agent.data.repository import JournalRepository
from journal_agent.ingestion.manuscript_parser import ManuscriptParser
from journal_agent.models.schemas import ManuscriptProfile, RecommendationResult
from journal_agent.ranking.scoring import CorpusScoringEngine
from journal_agent.ranking.supervised import SupervisedJournalRanker
from journal_agent.utils.text_processing import detect_language


DIRECT_LAW_JOURNAL_TERMS = {
    "law",
    "legal",
    "jurisprudence",
    "court",
    "judicial",
    "justice",
    "criminology",
    "penology",
    "human rights",
    "constitutional",
    "public law",
}
LAW_ADJACENT_JOURNAL_TERMS = {
    "political science",
    "public administration",
    "international relations",
    "governance",
    "regulation",
    "policy",
    "state",
    "institutions",
}


class JournalRecommendationAgent:
    def __init__(self) -> None:
        self.repository = JournalRepository()
        self.parser = ManuscriptParser()

    def recommend(
        self,
        *,
        dataset_path: str | Path,
        taxonomy_path: str | Path,
        model_path: str | Path | None = None,
        manuscript_path: str | Path | None = None,
        title: str | None = None,
        abstract: str | None = None,
        keywords: list[str] | str | None = None,
        discipline: str = "law",
        top_k: int = 15,
    ) -> tuple[ManuscriptProfile, list[RecommendationResult]]:
        manuscript = self.parser.parse(
            manuscript_path,
            title=title,
            abstract=abstract,
            keywords=keywords,
            discipline=discipline,
        )
        journals = self.repository.load_journals(dataset_path, discipline=discipline)
        if model_path:
            ranker = SupervisedJournalRanker.load(model_path)
            journals = self._select_candidate_journals(journals, manuscript, discipline=discipline)
            if not journals:
                raise ValueError(
                    "No candidate journals remain after language filtering. "
                    "Check that the dataset contains journals in the manuscript language."
                )
            recommendations = ranker.recommend(manuscript, candidate_journals=journals, top_k=top_k)
            return manuscript, recommendations[:top_k]

        taxonomy = self.repository.load_taxonomy(taxonomy_path)
        engine = CorpusScoringEngine(taxonomy)
        manuscript = engine.enrich_manuscript(manuscript)
        journals = self._select_candidate_journals(journals, manuscript, discipline=discipline)
        if not journals:
            raise ValueError(
                "No candidate journals remain after language filtering. "
                "Check that the dataset contains journals in the manuscript language."
            )
        recommendations = engine.score(manuscript, journals)
        return manuscript, recommendations[:top_k]

    def export_results(self, recommendations: list[RecommendationResult], output_path: str | Path) -> None:
        path = Path(output_path)
        if path.suffix.lower() == ".xlsx":
            raise ValueError("Only CSV export is supported. Please use a .csv output path.")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [item.as_csv_row() for item in recommendations]
        if not rows:
            raise ValueError("No recommendations to export.")
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _select_candidate_journals(
        self,
        journals: list,
        manuscript: ManuscriptProfile,
        *,
        discipline: str,
    ) -> list:
        manuscript_language = manuscript.language
        broad_candidates = []
        narrow_candidates = []
        legal_focus = manuscript.legal_topic_score >= 0.55
        for journal in journals:
            journal_language = self._journal_language(journal)
            if discipline == "law":
                if not self._is_law_related_journal(journal):
                    continue
                if manuscript_language in {"zh", "en"} and journal_language != manuscript_language:
                    continue
                broad_candidates.append(journal)
                narrow_candidates.append(journal)
                continue
            if manuscript_language == "zh":
                if journal_language != "zh":
                    continue
                broad_candidates.append(journal)
                if self._is_law_related_journal(journal):
                    narrow_candidates.append(journal)
                continue
            if manuscript_language == "en":
                if journal_language != "en":
                    continue
                if self._is_humanities_social_sciences(journal) or journal.discipline == discipline or self._is_ssci(journal):
                    broad_candidates.append(journal)
                if self._is_law_related_journal(journal):
                    narrow_candidates.append(journal)
                continue
            broad_candidates.append(journal)
            if self._is_law_related_journal(journal):
                narrow_candidates.append(journal)
        if legal_focus and narrow_candidates:
            return narrow_candidates
        return broad_candidates

    def _journal_language(self, journal) -> str:
        if journal.language:
            normalized = journal.language.strip().lower()
            if normalized.startswith("zh") or "chinese" in normalized:
                return "zh"
            if normalized.startswith("en") or "english" in normalized:
                return "en"
        return detect_language("\n".join([journal.title, journal.aims_and_scope]))

    def _is_ssci(self, journal) -> bool:
        return any(index.strip().upper() == "SSCI" for index in journal.indexing)

    def _is_humanities_social_sciences(self, journal) -> bool:
        discipline = (journal.discipline or "").strip().lower()
        if discipline in {
            "law",
            "political science",
            "public policy",
            "sociology",
            "anthropology",
            "economics",
            "history",
            "philosophy",
            "international relations",
            "area studies",
            "criminology",
            "communication",
            "education",
            "social sciences",
            "humanities",
        }:
            return True
        index_set = {item.strip().upper() for item in journal.indexing}
        return bool(index_set.intersection({"SSCI", "AHCI", "ESCI"}))

    def _is_law_related_journal(self, journal) -> bool:
        journal_text = " ".join(
            [
                journal.title,
                journal.discipline or "",
                " ".join(journal.subdisciplines),
                " ".join(journal.keywords),
                journal.aims_and_scope,
            ]
        ).lower()
        if any(term in journal_text for term in DIRECT_LAW_JOURNAL_TERMS):
            return True
        if any(term in journal_text for term in LAW_ADJACENT_JOURNAL_TERMS):
            return True
        return False
