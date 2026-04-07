from __future__ import annotations

import csv
from pathlib import Path

from journal_agent.data.repository import JournalRepository
from journal_agent.ingestion.manuscript_parser import ManuscriptParser
from journal_agent.models.schemas import ManuscriptProfile, RecommendationResult
from journal_agent.ranking.scoring import HeuristicScoringEngine
from journal_agent.utils.text_processing import detect_language


class JournalRecommendationAgent:
    def __init__(self) -> None:
        self.repository = JournalRepository()
        self.parser = ManuscriptParser()

    def recommend(
        self,
        *,
        dataset_path: str | Path,
        taxonomy_path: str | Path,
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
        journals = self.repository.load_journals(dataset_path)
        journals = self._select_candidate_journals(journals, manuscript, discipline=discipline)
        if not journals:
            raise ValueError(
                "No candidate journals remain after language filtering. "
                "Check that the dataset contains journals in the manuscript language."
            )
        taxonomy = self.repository.load_taxonomy(taxonomy_path)
        engine = HeuristicScoringEngine(taxonomy)
        recommendations = engine.score(manuscript, journals)
        return manuscript, recommendations[:top_k]

    def export_csv(self, recommendations: list[RecommendationResult], output_path: str | Path) -> None:
        path = Path(output_path)
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
        candidates = []
        for journal in journals:
            journal_language = self._journal_language(journal)
            if manuscript_language == "zh":
                if journal_language != "zh":
                    continue
                if journal.discipline != discipline:
                    continue
                candidates.append(journal)
                continue
            if manuscript_language == "en":
                if journal_language != "en":
                    continue
                if journal.discipline == discipline or self._is_ssci(journal):
                    candidates.append(journal)
                continue
            if journal.discipline == discipline:
                candidates.append(journal)
        return candidates

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
