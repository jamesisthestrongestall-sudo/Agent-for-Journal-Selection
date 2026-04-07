from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from journal_agent.models.schemas import JournalProfile
from journal_agent.utils.text_processing import extract_candidate_terms, normalize_space, normalize_title_key, parse_keyword_string


def _apply_field_mapping(raw_record: dict[str, Any], field_mapping: dict[str, str]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for target_field, source_field in field_mapping.items():
        mapped[target_field] = raw_record.get(source_field)
    return mapped


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_space(str(item)) for item in value if normalize_space(str(item))]
    return parse_keyword_string(str(value))


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None


def _normalize_language_name(value: str | None) -> str | None:
    normalized = normalize_space(value)
    if not normalized:
        return None
    lowered = normalized.lower()
    if "english" in lowered:
        return "en"
    if "chinese" in lowered:
        return "zh"
    return normalized


def _split_categories(value: str | None) -> list[str]:
    if not value:
        return []
    return [normalize_space(item) for item in value.split("|") if normalize_space(item)]


def _extract_country(address: str | None) -> str | None:
    normalized = normalize_space(address)
    if not normalized:
        return None
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return parts[-2] if len(parts) >= 2 else parts[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class JournalSource(ABC):
    def __init__(self, config: dict[str, Any], base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir

    @abstractmethod
    def fetch(self) -> list[JournalProfile]:
        raise NotImplementedError

    def _build_profile(self, record: dict[str, Any]) -> JournalProfile:
        static_fields = self.config.get("static_fields", {})
        payload = {**record, **static_fields}
        payload["keywords"] = _coerce_list(payload.get("keywords"))
        payload["subdisciplines"] = _coerce_list(payload.get("subdisciplines"))
        payload["indexing"] = _coerce_list(payload.get("indexing"))
        payload["methodology_preferences"] = _coerce_list(payload.get("methodology_preferences"))
        payload["editorial_preferences"] = _coerce_list(payload.get("editorial_preferences"))
        payload["source_tags"] = _coerce_list(payload.get("source_tags"))
        payload["impact_factor"] = _coerce_float(payload.get("impact_factor"))
        payload["review_cycle_months"] = _coerce_float(payload.get("review_cycle_months"))
        payload["acceptance_rate"] = _coerce_float(payload.get("acceptance_rate"))
        if not payload.get("journal_id") and payload.get("title"):
            payload["journal_id"] = normalize_title_key(payload["title"])
        return JournalProfile.model_validate(payload)


class JsonSource(JournalSource):
    def fetch(self) -> list[JournalProfile]:
        path = (self.base_dir / self.config["path"]).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [self._build_profile(item) for item in payload]


class CsvSource(JournalSource):
    def fetch(self) -> list[JournalProfile]:
        path = (self.base_dir / self.config["path"]).resolve()
        delimiter = self.config.get("delimiter", ",")
        field_mapping = self.config.get("field_mapping", {})
        records: list[JournalProfile] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                mapped = _apply_field_mapping(row, field_mapping)
                records.append(self._build_profile(mapped))
        return records


class ClarivateMjlCsvSource(CsvSource):
    DEFAULT_FIELD_MAPPING = {
        "title": "Journal name",
        "website": "Journal website",
        "publisher": "Publisher",
        "discipline": "Web of Science Categories",
        "language": "Languages",
        "jcr_quartile": "Journal Impact Factor Quartile",
        "impact_factor": "Journal Impact Factor",
    }

    def fetch(self) -> list[JournalProfile]:
        self.config.setdefault("field_mapping", self.DEFAULT_FIELD_MAPPING)
        self.config.setdefault("static_fields", {})
        static_fields = self.config["static_fields"]
        static_fields.setdefault("source_tags", ["clarivate_master_journal_list"])
        return super().fetch()


class SsciCsvLookupSource(CsvSource):
    DEFAULT_FIELD_MAPPING = {
        "title": "Journal title",
        "issn": "ISSN",
        "eissn": "eISSN",
        "publisher": "Publisher name",
        "language": "Languages",
        "discipline": "Web of Science Categories",
    }
    CATEGORY_RULES = {
        "law": {
            "methodology_preferences": ["doctrinal", "comparative", "empirical"],
            "editorial_preferences": ["theoretical_innovation", "judicial_practice", "legislation_policy"],
            "keywords": ["law", "legal studies", "regulation", "judicial", "comparative law"],
        },
        "criminology": {
            "methodology_preferences": ["empirical", "qualitative", "quantitative"],
            "editorial_preferences": ["criminal_justice", "judicial_practice"],
            "keywords": ["crime", "criminal justice", "penology", "policing"],
        },
        "political science": {
            "methodology_preferences": ["comparative", "qualitative", "empirical"],
            "editorial_preferences": ["legislation_policy", "international_rule_of_law", "regional_china"],
            "keywords": ["governance", "public policy", "state", "institutions"],
        },
        "public administration": {
            "methodology_preferences": ["empirical", "qualitative", "interdisciplinary"],
            "editorial_preferences": ["legislation_policy", "regional_china"],
            "keywords": ["governance", "public administration", "policy", "institutions"],
        },
        "international relations": {
            "methodology_preferences": ["comparative", "qualitative", "historical"],
            "editorial_preferences": ["international_rule_of_law", "regional_china"],
            "keywords": ["international relations", "global governance", "transnational", "institutions"],
        },
        "communication": {
            "methodology_preferences": ["empirical", "qualitative", "interdisciplinary"],
            "editorial_preferences": ["digital_governance", "theoretical_innovation"],
            "keywords": ["communication", "digital media", "platforms", "algorithms"],
        },
        "business": {
            "methodology_preferences": ["empirical", "quantitative", "interdisciplinary"],
            "editorial_preferences": ["commercial_finance", "legislation_policy"],
            "keywords": ["business", "management", "governance", "strategy"],
        },
        "management": {
            "methodology_preferences": ["empirical", "quantitative", "interdisciplinary"],
            "editorial_preferences": ["commercial_finance", "legislation_policy"],
            "keywords": ["management", "organizations", "governance", "institutions"],
        },
        "finance": {
            "methodology_preferences": ["empirical", "quantitative"],
            "editorial_preferences": ["commercial_finance"],
            "keywords": ["finance", "markets", "regulation", "corporate governance"],
        },
        "sociology": {
            "methodology_preferences": ["qualitative", "empirical", "interdisciplinary"],
            "editorial_preferences": ["theoretical_innovation", "regional_china"],
            "keywords": ["society", "institutions", "governance", "inequality"],
        },
        "psychology": {
            "methodology_preferences": ["empirical", "quantitative", "qualitative"],
            "editorial_preferences": ["theoretical_innovation"],
            "keywords": ["behavior", "decision making", "social cognition"],
        },
        "education": {
            "methodology_preferences": ["empirical", "qualitative", "interdisciplinary"],
            "editorial_preferences": ["regional_china", "theoretical_innovation"],
            "keywords": ["education", "learning", "institutions", "policy"],
        },
        "philosophy": {
            "methodology_preferences": ["historical", "doctrinal", "qualitative"],
            "editorial_preferences": ["theoretical_innovation"],
            "keywords": ["theory", "ethics", "philosophy", "history"],
        },
        "history": {
            "methodology_preferences": ["historical", "qualitative"],
            "editorial_preferences": ["theoretical_innovation", "regional_china"],
            "keywords": ["history", "institutions", "archives", "historical analysis"],
        },
    }

    def fetch(self) -> list[JournalProfile]:
        path = (self.base_dir / self.config["path"]).resolve()
        delimiter = self.config.get("delimiter", ",")
        records: list[JournalProfile] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                records.append(self._build_profile(self._lookup_profile(row)))
        return records

    def _lookup_profile(self, row: dict[str, Any]) -> dict[str, Any]:
        title = normalize_space(row.get("Journal title"))
        publisher = normalize_space(row.get("Publisher name"))
        publisher_address = normalize_space(row.get("Publisher address"))
        categories = _split_categories(row.get("Web of Science Categories"))
        language = _normalize_language_name(row.get("Languages"))
        keywords = self._derive_keywords(title, categories)
        methodology_preferences = self._derive_preferences(categories, "methodology_preferences")
        editorial_preferences = self._derive_preferences(categories, "editorial_preferences")
        aims_and_scope = self._build_aims_scope(title, publisher, language, categories, keywords)
        notes = (
            "Portrait generated from SSCI CSV metadata and category lookup rules. "
            "Use external enrichment later if you need publisher website, review cycle, or acceptance rate."
        )
        return {
            "title": title,
            "issn": normalize_space(row.get("ISSN")),
            "eissn": normalize_space(row.get("eISSN")),
            "publisher": publisher,
            "country": _extract_country(publisher_address),
            "language": language,
            "discipline": categories[0] if categories else "social sciences",
            "subdisciplines": categories,
            "keywords": keywords,
            "methodology_preferences": methodology_preferences,
            "editorial_preferences": editorial_preferences,
            "aims_and_scope": aims_and_scope,
            "indexing": ["SSCI"],
            "source_tags": ["ssci_csv_lookup", "ssci_local_csv"],
            "notes": notes,
        }

    def _derive_keywords(self, title: str, categories: list[str]) -> list[str]:
        title_terms = extract_candidate_terms(title, top_k=8)
        category_terms: list[str] = []
        for category in categories:
            lowered = category.lower()
            for key, rule in self.CATEGORY_RULES.items():
                if key in lowered:
                    category_terms.extend(rule["keywords"])
        raw_keywords = [*title_terms, *categories, *category_terms]
        return list(dict.fromkeys(item for item in raw_keywords if normalize_space(item)))[:15]

    def _derive_preferences(self, categories: list[str], target_key: str) -> list[str]:
        collected: list[str] = []
        for category in categories:
            lowered = category.lower()
            for key, rule in self.CATEGORY_RULES.items():
                if key in lowered:
                    collected.extend(rule[target_key])
        if not collected and target_key == "methodology_preferences":
            collected = ["empirical", "interdisciplinary"]
        if not collected and target_key == "editorial_preferences":
            collected = ["theoretical_innovation"]
        return list(dict.fromkeys(collected))

    def _build_aims_scope(
        self,
        title: str,
        publisher: str,
        language: str | None,
        categories: list[str],
        keywords: list[str],
    ) -> str:
        category_text = ", ".join(categories[:4]) if categories else "social science research"
        keyword_text = ", ".join(keywords[:6]) if keywords else "interdisciplinary social-science topics"
        language_text = "English" if language == "en" else ("Chinese" if language == "zh" else (language or "not specified"))
        publisher_text = publisher or "an indexed publisher"
        return (
            f"{title} is an SSCI-indexed journal published by {publisher_text}. "
            f"Based on Web of Science categories {category_text}, its profile centers on {keyword_text}. "
            f"Primary publication language: {language_text}."
        )


class HtmlListSource(JournalSource):
    def fetch(self) -> list[JournalProfile]:
        response = requests.get(self.config["url"], timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        item_selector = self.config["item_selector"]
        field_mapping = self.config.get("field_mapping", {})
        records: list[JournalProfile] = []
        for item in soup.select(item_selector):
            extracted: dict[str, Any] = {}
            for target_field, selector in field_mapping.items():
                extracted[target_field] = self._extract_selector_value(item, selector, self.config["url"])
            if normalize_space(str(extracted.get("title", ""))):
                records.append(self._build_profile(extracted))
        return records

    def _extract_selector_value(self, item: Any, selector: str, page_url: str) -> str:
        selector = selector.strip()
        attribute = None
        if "@" in selector:
            selector, attribute = selector.split("@", 1)
            selector = selector.strip()
            attribute = attribute.strip()
        node = item.select_one(selector) if selector else item
        if node is None:
            return ""
        if attribute:
            value = node.get(attribute, "")
            if attribute == "href":
                return urljoin(page_url, value)
            return normalize_space(value)
        return normalize_space(node.get_text(" ", strip=True))


class DatasetBuilder:
    SOURCE_TYPES = {
        "json": JsonSource,
        "csv": CsvSource,
        "clarivate_mjl_csv": ClarivateMjlCsvSource,
        "ssci_csv_lookup": SsciCsvLookupSource,
        "html_list": HtmlListSource,
    }

    def build_from_manifest(self, manifest_path: str | Path) -> list[JournalProfile]:
        manifest_file = Path(manifest_path)
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        sources = payload.get("sources", [])
        merged_profiles: dict[str, JournalProfile] = {}
        for config in sources:
            source_type = config.get("type")
            source_cls = self.SOURCE_TYPES.get(source_type)
            if source_cls is None:
                raise ValueError(f"Unsupported source type: {source_type}")
            source = source_cls(config, manifest_file.parent)
            for profile in source.fetch():
                key = profile.journal_id or normalize_title_key(profile.title)
                merged_profiles[key] = self._merge(merged_profiles.get(key), profile)
        return list(merged_profiles.values())

    def save(self, profiles: list[JournalProfile], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [profile.model_dump(mode="json") for profile in profiles]
        path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_from_ssci_csv(self, csv_path: str | Path) -> list[JournalProfile]:
        path = Path(csv_path)
        config = {
            "type": "ssci_csv_lookup",
            "path": path.name,
        }
        source = SsciCsvLookupSource(config, path.parent)
        return source.fetch()

    def _merge(self, current: JournalProfile | None, incoming: JournalProfile) -> JournalProfile:
        if current is None:
            return incoming
        payload = current.model_dump()
        incoming_payload = incoming.model_dump()
        for key, value in incoming_payload.items():
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                existing = payload.get(key, [])
                payload[key] = list(dict.fromkeys([*existing, *value]))
                continue
            if not payload.get(key):
                payload[key] = value
        return JournalProfile.model_validate(payload)
