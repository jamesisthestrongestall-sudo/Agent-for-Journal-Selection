from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KEYWORD_SPLIT_PATTERN = re.compile(r"[;,；，、|\n]+")
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{1,}|[\u4e00-\u9fff]{2,}")


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_title_key(text: str) -> str:
    compact = normalize_space(text).lower()
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", compact)
    return compact


def parse_keyword_string(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [normalize_space(part) for part in KEYWORD_SPLIT_PATTERN.split(raw)]
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return unique


def detect_language(text: str) -> str:
    normalized = normalize_space(text)
    if not normalized:
        return "unknown"
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", normalized))
    if chinese_chars >= max(12, latin_words):
        return "zh"
    if latin_words > 0:
        return "en"
    return "unknown"


def extract_candidate_terms(text: str, top_k: int = 20) -> list[str]:
    normalized = normalize_space(text).lower()
    counter: Counter[str] = Counter()
    for token in TOKEN_PATTERN.findall(normalized):
        if len(token) <= 1:
            continue
        counter[token] += 1
    scored = sorted(counter.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _ in scored[:top_k]]


def keyword_overlap(left: list[str], right: list[str]) -> float:
    left_set = {normalize_space(item).lower() for item in left if normalize_space(item)}
    right_set = {normalize_space(item).lower() for item in right if normalize_space(item)}
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def cosine_similarity_scores(query_text: str, documents: list[str]) -> list[float]:
    if not normalize_space(query_text) or not documents:
        return [0.0 for _ in documents]
    cleaned_docs = [normalize_space(doc) or "empty" for doc in documents]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([normalize_space(query_text)] + cleaned_docs)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(score) for score in similarities]


def hybrid_similarity_scores(query_text: str, documents: list[str]) -> list[float]:
    if not normalize_space(query_text) or not documents:
        return [0.0 for _ in documents]
    cleaned_docs = [normalize_space(doc) or "empty" for doc in documents]
    normalized_query = normalize_space(query_text)

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    char_matrix = char_vectorizer.fit_transform([normalized_query] + cleaned_docs)
    char_scores = cosine_similarity(char_matrix[0:1], char_matrix[1:]).flatten()

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda text: TOKEN_PATTERN.findall(normalize_space(text).lower()),
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
    )
    word_matrix = word_vectorizer.fit_transform([normalized_query] + cleaned_docs)
    word_scores = cosine_similarity(word_matrix[0:1], word_matrix[1:]).flatten()

    blended = (0.45 * char_scores) + (0.55 * word_scores)
    return [float(score) for score in blended]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def safe_mean(values: list[float], default: float = 0.0) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return default
    return sum(filtered) / len(filtered)


def top_k_mean(values: Iterable[float], k: int = 3, default: float = 0.0) -> float:
    ordered = sorted((value for value in values if not math.isnan(value)), reverse=True)
    if not ordered:
        return default
    top_values = ordered[: max(1, k)]
    return sum(top_values) / len(top_values)
