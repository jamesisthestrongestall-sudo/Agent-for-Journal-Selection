from __future__ import annotations

import json
from pathlib import Path

from journal_agent.models.schemas import JournalProfile, TaxonomyProfile
from journal_agent.utils.text_processing import normalize_title_key


class JournalRepository:
    def load_journals(self, dataset_path: str | Path, *, discipline: str | None = None) -> list[JournalProfile]:
        payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        journals = [JournalProfile.model_validate(item) for item in payload]
        if discipline:
            journals = [journal for journal in journals if journal.discipline == discipline]
        for journal in journals:
            if not journal.journal_id:
                journal.journal_id = normalize_title_key(journal.title)
        return journals

    def load_taxonomy(self, taxonomy_path: str | Path) -> TaxonomyProfile:
        payload = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        return TaxonomyProfile.model_validate(payload)
