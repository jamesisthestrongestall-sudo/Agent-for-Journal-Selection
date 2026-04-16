from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from journal_agent.models.schemas import ManuscriptProfile
from journal_agent.utils.text_processing import (
    detect_language,
    extract_candidate_terms,
    normalize_space,
    parse_keyword_string,
)


class ManuscriptParser:
    TITLE_PATTERN = re.compile(r"^(?:title)\s*[:：]?\s*(.*)$", re.IGNORECASE)
    ABSTRACT_PATTERNS = [
        re.compile(
            r"(?:\u6458\u8981|abstract)\s*[:\uff1a]\s*(.+?)(?=(?:\u5173\u952e\u8bcd|\u5173\u952e\u5b57|introduction|\u5f15\u8a00|\n\d+[.\u3001]))",
            re.IGNORECASE | re.DOTALL,
        ),
    ]
    KEYWORD_PATTERNS = [
        re.compile(r"(?:\u5173\u952e\u8bcd|\u5173\u952e\u5b57|keywords?)\s*[:\uff1a]\s*(.+)", re.IGNORECASE),
    ]
    KEYWORD_HEADINGS = {"keywords", "keyword", "\u5173\u952e\u8bcd", "\u5173\u952e\u5b57"}
    REFERENCE_HEADING_PATTERNS = [
        re.compile(r"^(?:\u53c2\u8003\u6587\u732e|\u53c2\u8003\u4e66\u76ee|references|bibliography|works cited)\s*$", re.IGNORECASE),
    ]
    INLINE_CITATION_PATTERNS = [
        re.compile(r"\[\s*\d+(?:\s*[-,，]\s*\d+)*\s*\]"),
        re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+"),
        re.compile(r"\((?:[^()]{0,60}\d{4}[a-z]?(?:[^()]{0,30}))\)"),
        re.compile(r"（(?:[^（）]{0,60}\d{4}[a-z]?(?:[^（）]{0,30}))）"),
    ]
    FOOTNOTE_LINE_PATTERN = re.compile(
        r"^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳*]+)[.)]?\s*(?:\u53c2\u89c1|\u89c1|See|see|Available at|DOI|doi|https?://)",
        re.IGNORECASE,
    )

    def parse(
        self,
        manuscript_path: str | Path | None = None,
        *,
        title: str | None = None,
        abstract: str | None = None,
        keywords: list[str] | str | None = None,
        discipline: str = "law",
    ) -> ManuscriptProfile:
        raw_text = ""
        footnotes_text = ""
        provided_references_text = ""
        if manuscript_path:
            raw_text, footnotes_text, provided_references_text = self._read_path(Path(manuscript_path))
        body_text, references_text = self._split_references(raw_text)
        references_text = normalize_space("\n".join([provided_references_text, references_text]))
        body_text = self._strip_inline_citations(body_text)
        body_text, inline_footnotes = self._strip_footnote_like_lines(body_text)
        footnotes_text = normalize_space("\n".join([footnotes_text, inline_footnotes]))

        title = normalize_space(title) or self._extract_title(body_text)
        abstract = normalize_space(abstract) or self._extract_abstract(body_text)
        keyword_list = self._normalize_keywords(keywords) or self._extract_keywords(body_text)
        language = detect_language("\n".join([title, abstract, " ".join(keyword_list), body_text]))
        cleaned_body = normalize_space(body_text)
        return ManuscriptProfile(
            title=title or "Untitled Manuscript",
            abstract=abstract,
            keywords=keyword_list,
            language=language,
            full_text=cleaned_body,
            references_text=normalize_space(references_text),
            footnotes_text=footnotes_text,
            discipline=discipline,
            extracted_terms=extract_candidate_terms(
                " ".join([title or "", abstract or "", " ".join(keyword_list), cleaned_body])
            ),
        )

    def _normalize_keywords(self, keywords: list[str] | str | None) -> list[str]:
        if keywords is None:
            return []
        if isinstance(keywords, str):
            return parse_keyword_string(keywords)
        return parse_keyword_string("; ".join(keywords))

    def _read_path(self, path: Path) -> tuple[str, str, str]:
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8"), "", ""
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            full_text = "\n".join(
                [
                    payload.get("title", ""),
                    payload.get("abstract", ""),
                    " ".join(payload.get("keywords", [])),
                    payload.get("full_text", ""),
                ]
            )
            references_text = payload.get("references_text", "")
            footnotes_text = payload.get("footnotes_text", "")
            return normalize_space(full_text), normalize_space(footnotes_text), normalize_space(references_text)
        raise ValueError(f"Unsupported manuscript format: {path.suffix}")

    def _read_docx(self, path: Path) -> tuple[str, str, str]:
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
            footnotes_bytes = archive.read("word/footnotes.xml") if "word/footnotes.xml" in archive.namelist() else None

        root = ElementTree.fromstring(xml_bytes)
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            joined = normalize_space("".join(texts))
            if joined:
                paragraphs.append(joined)

        footnotes: list[str] = []
        if footnotes_bytes:
            footnotes_root = ElementTree.fromstring(footnotes_bytes)
            for footnote in footnotes_root.findall(".//w:footnote", namespace):
                footnote_type = footnote.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type")
                if footnote_type:
                    continue
                texts = [node.text or "" for node in footnote.findall(".//w:t", namespace)]
                joined = normalize_space("".join(texts))
                if joined:
                    footnotes.append(joined)

        return "\n".join(paragraphs), "\n".join(footnotes), ""

    def _split_references(self, text: str) -> tuple[str, str]:
        lines = [line.rstrip() for line in text.splitlines()]
        body_lines: list[str] = []
        reference_lines: list[str] = []
        in_references = False
        for line in lines:
            normalized = normalize_space(line)
            if not normalized:
                if in_references:
                    reference_lines.append("")
                else:
                    body_lines.append("")
                continue
            if any(pattern.match(normalized) for pattern in self.REFERENCE_HEADING_PATTERNS):
                in_references = True
                reference_lines.append(normalized)
                continue
            if in_references:
                reference_lines.append(line)
            else:
                body_lines.append(line)
        return "\n".join(body_lines), "\n".join(reference_lines)

    def _strip_inline_citations(self, text: str) -> str:
        cleaned = text
        for pattern in self.INLINE_CITATION_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        return cleaned

    def _strip_footnote_like_lines(self, text: str) -> tuple[str, str]:
        body_lines: list[str] = []
        footnote_lines: list[str] = []
        for line in text.splitlines():
            normalized = normalize_space(line)
            if normalized and self.FOOTNOTE_LINE_PATTERN.match(normalized):
                footnote_lines.append(normalized)
                continue
            body_lines.append(line)
        return "\n".join(body_lines), "\n".join(footnote_lines)

    def _extract_title(self, text: str) -> str:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            candidate = normalize_space(line)
            if not candidate:
                continue
            title_match = self.TITLE_PATTERN.match(candidate)
            if title_match:
                inline_title = normalize_space(title_match.group(1))
                if inline_title:
                    return inline_title
                for fallback_line in lines[index + 1 :]:
                    fallback_candidate = normalize_space(fallback_line)
                    if fallback_candidate:
                        return fallback_candidate
                continue
            lower = candidate.lower()
            if lower.startswith(("abstract", "keywords")):
                continue
            if candidate.startswith(("\u6458\u8981", "\u5173\u952e\u8bcd", "\u5173\u952e\u5b57")):
                continue
            return candidate
        return ""

    def _extract_abstract(self, text: str) -> str:
        for pattern in self.ABSTRACT_PATTERNS:
            match = pattern.search(text)
            if match:
                return normalize_space(match.group(1))
        lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
        if len(lines) >= 2:
            return lines[1][:600]
        return ""

    def _extract_keywords(self, text: str) -> list[str]:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            normalized_line = normalize_space(line).lower()
            for pattern in self.KEYWORD_PATTERNS:
                match = pattern.search(line)
                if match:
                    inline_keywords = parse_keyword_string(match.group(1))
                    if inline_keywords:
                        return inline_keywords
                    for fallback_line in lines[index + 1 :]:
                        fallback_candidate = normalize_space(fallback_line)
                        if fallback_candidate:
                            return parse_keyword_string(fallback_candidate)
            if normalized_line in self.KEYWORD_HEADINGS:
                for fallback_line in lines[index + 1 :]:
                    fallback_candidate = normalize_space(fallback_line)
                    if fallback_candidate:
                        return parse_keyword_string(fallback_candidate)
        return []
