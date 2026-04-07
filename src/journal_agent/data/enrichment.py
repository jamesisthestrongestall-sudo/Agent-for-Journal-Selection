from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from journal_agent.models.schemas import JournalArticleExample, JournalProfile
from journal_agent.utils.text_processing import normalize_space, normalize_title_key


OPENALEX_API_BASE = "https://api.openalex.org"
MAX_ARTICLE_ABSTRACT_CHARS = 4000
DEFAULT_HEADERS = {
    "User-Agent": (
        "legal-journal-agent/0.1 "
        "(journal profile enrichment; contact: local-enrichment@example.com)"
    )
}
AIMS_TEXT_PATTERN = re.compile(
    r"(aims?\s*(?:&|and)?\s*scope|about\s+this\s+journal|about\s+the\s+journal|"
    r"journal\s+overview|focus\s+and\s+scope|scope)",
    re.IGNORECASE,
)
AIMS_LINK_PATTERN = re.compile(
    r"(aims?|scope|about|overview|journal-information|focus)",
    re.IGNORECASE,
)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    ordered_tokens: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            ordered_tokens.append((position, token))
    ordered_tokens.sort(key=lambda item: item[0])
    return normalize_space(" ".join(token for _, token in ordered_tokens))


def _same_domain(base_url: str, candidate_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def _best_title_match(results: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    normalized_title = normalize_title_key(title)
    if not results:
        return None
    exact = next(
        (
            result
            for result in results
            if normalize_title_key(result.get("display_name", "")) == normalized_title
        ),
        None,
    )
    if exact:
        return exact
    prefix = next(
        (
            result
            for result in results
            if normalized_title in normalize_title_key(result.get("display_name", ""))
            or normalize_title_key(result.get("display_name", "")) in normalized_title
        ),
        None,
    )
    return prefix or results[0]


class OpenAlexClient:
    def __init__(self, *, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def find_source(
        self,
        *,
        title: str,
        issn: str | None,
        eissn: str | None,
    ) -> dict[str, Any] | None:
        for external_id in [issn, eissn]:
            if not normalize_space(external_id):
                continue
            payload = self._get_json(f"/sources/issn:{normalize_space(external_id)}")
            if payload and payload.get("id"):
                return payload
        payload = self._get_json(
            "/sources",
            params={
                "search": title,
                "filter": "type:journal",
                "per-page": 5,
            },
        )
        results = payload.get("results", []) if payload else []
        return _best_title_match(results, title)

    def fetch_recent_articles(self, *, source_id: str, limit: int = 5) -> list[JournalArticleExample]:
        payload = self._get_json(
            "/works",
            params={
                "filter": f"primary_location.source.id:{source_id},type:article",
                "sort": "publication_date:desc",
                "per-page": limit,
            },
        )
        results = payload.get("results", []) if payload else []
        articles: list[JournalArticleExample] = []
        for result in results:
            abstract_text = reconstruct_abstract(result.get("abstract_inverted_index"))
            article_keywords = [
                normalize_space(item.get("display_name", ""))
                for item in result.get("keywords", [])[:8]
                if normalize_space(item.get("display_name", ""))
            ]
            if not article_keywords:
                article_keywords = [
                    normalize_space(item.get("display_name", ""))
                    for item in result.get("concepts", [])[:8]
                    if normalize_space(item.get("display_name", ""))
                ]
            articles.append(
                JournalArticleExample(
                    title=normalize_space(result.get("display_name", "")) or "Untitled Article",
                    keywords=list(dict.fromkeys(article_keywords)),
                    abstract_snippet=abstract_text[:MAX_ARTICLE_ABSTRACT_CHARS],
                    full_text="",
                    references_text="",
                    is_oa=(result.get("open_access") or {}).get("is_oa"),
                )
            )
        return articles

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        response = self.session.get(f"{OPENALEX_API_BASE}{path}", params=params, timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


class JournalSiteCrawler:
    def __init__(self, *, timeout: int = 25) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def fetch_aims_and_scope(self, homepage_url: str | None) -> tuple[str | None, str | None]:
        if not normalize_space(homepage_url):
            return None, None
        homepage_url = normalize_space(homepage_url)
        try:
            homepage_html = self._get_html(homepage_url)
        except requests.RequestException:
            return None, None
        candidate_urls = self._discover_candidate_urls(homepage_url, homepage_html)
        for candidate_url in candidate_urls:
            try:
                html = self._get_html(candidate_url)
            except requests.RequestException:
                continue
            extracted = self._extract_aims_text(html)
            if extracted:
                return extracted, candidate_url
        extracted = self._extract_aims_text(homepage_html)
        if extracted:
            return extracted, homepage_url
        return None, None

    def _get_html(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _discover_candidate_urls(self, homepage_url: str, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[tuple[int, str]] = []
        for anchor in soup.find_all("a", href=True):
            href = normalize_space(anchor.get("href"))
            text = normalize_space(anchor.get_text(" ", strip=True))
            if not href:
                continue
            absolute_url = urljoin(homepage_url, href)
            if not _same_domain(homepage_url, absolute_url):
                continue
            score = 0
            if AIMS_TEXT_PATTERN.search(text):
                score += 6
            if AIMS_LINK_PATTERN.search(text):
                score += 3
            if AIMS_LINK_PATTERN.search(absolute_url):
                score += 2
            if score:
                candidates.append((score, absolute_url))
        for suffix in [
            "aims-and-scope",
            "aims-and-scope/",
            "about",
            "overview",
            "about-this-journal",
            "journal-information",
            "journal-information/aims-and-scope",
        ]:
            candidates.append((1, urljoin(homepage_url.rstrip("/") + "/", suffix)))
        ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
        unique_urls: list[str] = []
        seen: set[str] = set()
        for _, url in ordered:
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)
        return unique_urls[:8]

    def _extract_aims_text(self, html: str) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
            tag.decompose()

        structured_candidates: list[str] = []
        for selector in [
            "[id*='aim']",
            "[class*='aim']",
            "[id*='scope']",
            "[class*='scope']",
            "[id*='about']",
            "[class*='about']",
        ]:
            for node in soup.select(selector):
                text = self._clean_text(node)
                if self._looks_like_aims_text(text):
                    structured_candidates.append(text)

        for heading in soup.find_all(re.compile(r"^h[1-6]$")):
            heading_text = normalize_space(heading.get_text(" ", strip=True))
            if not AIMS_TEXT_PATTERN.search(heading_text):
                continue
            for candidate in [heading.parent, heading.parent.parent if heading.parent else None]:
                if not isinstance(candidate, Tag):
                    continue
                text = self._clean_text(candidate)
                if self._looks_like_aims_text(text):
                    structured_candidates.append(text)

        main_like = soup.find(["main", "article"])
        if main_like:
            text = self._clean_text(main_like)
            if self._looks_like_aims_text(text):
                structured_candidates.append(text)

        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            structured_candidates.append(normalize_space(meta_description["content"]))

        if not structured_candidates:
            return None

        best = max(structured_candidates, key=len)
        return best[:2400]

    def _clean_text(self, node: Tag) -> str:
        text = normalize_space(node.get_text(" ", strip=True))
        text = re.sub(r"\s+", " ", text)
        return text

    def _looks_like_aims_text(self, text: str) -> bool:
        if not text or len(text) < 120:
            return False
        if "cookie" in text.lower() and len(text) < 300:
            return False
        return bool(AIMS_TEXT_PATTERN.search(text) or len(text) >= 220)


class JournalProfileEnricher:
    def __init__(
        self,
        *,
        recent_article_count: int = 5,
        request_delay_sec: float = 0.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.recent_article_count = recent_article_count
        self.request_delay_sec = request_delay_sec
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.openalex = OpenAlexClient()
        self.site_crawler = JournalSiteCrawler()

    def enrich(self, profile: JournalProfile) -> JournalProfile:
        cached = self._load_cache(profile.journal_id or normalize_title_key(profile.title))
        if cached is not None:
            return self._merge_cached(profile, cached)

        source = self.openalex.find_source(title=profile.title, issn=profile.issn, eissn=profile.eissn)
        homepage_url = normalize_space((source or {}).get("homepage_url", "")) or profile.website
        aims_and_scope, aims_url = self.site_crawler.fetch_aims_and_scope(homepage_url)
        recent_articles: list[JournalArticleExample] = []
        if source and source.get("id"):
            recent_articles = self.openalex.fetch_recent_articles(
                source_id=source["id"],
                limit=self.recent_article_count,
            )
        if self.request_delay_sec > 0:
            time.sleep(self.request_delay_sec)

        notes_bits = [profile.notes or ""]
        if source and source.get("id"):
            notes_bits.append("Enriched using OpenAlex source metadata and recent works.")
        if aims_and_scope:
            notes_bits.append(f"Aims & Scope crawled from {aims_url}.")

        enriched_payload = profile.model_dump()
        enriched_payload["website"] = homepage_url or profile.website
        if aims_and_scope:
            enriched_payload["aims_and_scope"] = aims_and_scope
        if recent_articles:
            enriched_payload["recent_articles"] = [article.model_dump() for article in recent_articles]
        enriched_payload["notes"] = normalize_space(" ".join(bit for bit in notes_bits if normalize_space(bit)))
        enriched_payload["source_tags"] = list(
            dict.fromkeys([*profile.source_tags, "openalex_enriched", "site_crawled_aims_scope"])
        )
        enriched_profile = JournalProfile.model_validate(enriched_payload)
        self._save_cache(
            profile.journal_id or normalize_title_key(profile.title),
            {
                "website": enriched_profile.website,
                "aims_and_scope": enriched_profile.aims_and_scope,
                "recent_articles": [article.model_dump() for article in enriched_profile.recent_articles],
                "notes": enriched_profile.notes,
                "source_tags": enriched_profile.source_tags,
            },
        )
        return enriched_profile

    def _load_cache(self, cache_key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))

    def _save_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _merge_cached(self, profile: JournalProfile, cached: dict[str, Any]) -> JournalProfile:
        payload = profile.model_dump()
        payload["website"] = cached.get("website") or profile.website
        payload["aims_and_scope"] = cached.get("aims_and_scope") or profile.aims_and_scope
        payload["recent_articles"] = cached.get("recent_articles") or [article.model_dump() for article in profile.recent_articles]
        payload["notes"] = cached.get("notes") or profile.notes
        payload["source_tags"] = cached.get("source_tags") or profile.source_tags
        return JournalProfile.model_validate(payload)
