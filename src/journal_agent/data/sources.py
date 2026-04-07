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
from journal_agent.utils.text_processing import normalize_space, normalize_title_key, parse_keyword_string


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
