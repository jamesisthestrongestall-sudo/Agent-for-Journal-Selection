from __future__ import annotations

import re
from dataclasses import dataclass

from journal_agent.utils.text_processing import clamp, normalize_space


GENERAL_LAW_REVIEW_BUCKET = "general_law_review"


@dataclass(frozen=True)
class LawSubfieldBucket:
    name: str
    label: str
    aliases: tuple[str, ...]
    title_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubfieldProfile:
    scores: dict[str, float]
    primary: str | None
    focus: tuple[str, ...]


LAW_SUBFIELD_BUCKETS: tuple[LawSubfieldBucket, ...] = (
    LawSubfieldBucket(
        name="international_comparative_law",
        label="International / Comparative Law",
        aliases=(
            "international law",
            "international legal",
            "transnational",
            "comparative law",
            "comparative legal",
            "world trade",
            "trade law",
            "investment law",
            "investment treaty",
            "arbitration",
            "wto",
            "marine law",
            "coastal law",
            "ocean law",
            "international dispute",
            "international criminal",
            "international humanitarian",
            "migration law",
            "european law",
            "common market",
        ),
        title_aliases=(
            "international",
            "comparative",
            "transnational",
            "world trade",
            "european law",
            "international criminal",
            "marine",
            "coastal",
            "ocean",
            "icsid",
        ),
    ),
    LawSubfieldBucket(
        name="criminal_justice_criminology",
        label="Criminal Law / Criminology",
        aliases=(
            "criminal law",
            "criminal justice",
            "criminology",
            "crime",
            "penology",
            "policing",
            "sentencing",
            "evidence",
            "proof",
            "forensic",
            "offender",
            "detention",
            "punishment",
            "juvenile justice",
            "prosecution",
        ),
        title_aliases=(
            "criminal",
            "criminology",
            "crime",
            "justice",
            "evidence",
            "proof",
            "forensic",
            "juvenile",
        ),
    ),
    LawSubfieldBucket(
        name="law_and_economics_business",
        label="Law & Economics / Business Law",
        aliases=(
            "law and economics",
            "economic analysis",
            "economics",
            "competition law",
            "antitrust",
            "corporate law",
            "corporate governance",
            "business law",
            "commercial law",
            "bankruptcy",
            "insolvency",
            "finance",
            "organization",
            "merger",
            "market regulation",
            "securities",
            "tax law",
            "trade finance",
        ),
        title_aliases=(
            "economics",
            "economic",
            "business",
            "corporate",
            "bankruptcy",
            "insolvency",
            "competition",
            "organization",
            "trade",
        ),
    ),
    LawSubfieldBucket(
        name="health_medicine_bioethics",
        label="Health / Medical / Bioethics Law",
        aliases=(
            "medical law",
            "medicine",
            "medical",
            "health law",
            "healthcare",
            "health policy",
            "bioethics",
            "biosciences",
            "food and drug",
            "psychiatry",
            "mental health",
            "biomedical",
            "forensic medicine",
            "embryo",
            "genomic",
            "neurotechnology",
            "neuroscience",
        ),
        title_aliases=(
            "medicine",
            "medical",
            "health",
            "biosciences",
            "psychiatry",
            "food and drug",
            "bio",
        ),
    ),
    LawSubfieldBucket(
        name="environmental_energy_resources",
        label="Environmental / Energy / Natural Resources Law",
        aliases=(
            "environmental law",
            "environmental",
            "ecology",
            "energy law",
            "energy regulation",
            "natural resources",
            "climate",
            "sustainability",
            "water law",
            "biodiversity",
            "resource law",
            "resource governance",
            "coastal governance",
            "marine environment",
            "ocean governance",
        ),
        title_aliases=(
            "environmental",
            "ecology",
            "energy",
            "natural resources",
            "marine",
            "coastal",
            "ocean",
            "resource",
        ),
    ),
    LawSubfieldBucket(
        name="constitutional_human_rights",
        label="Constitutional / Human Rights Law",
        aliases=(
            "constitutional law",
            "constitutional",
            "human rights",
            "civil rights",
            "civil liberties",
            "public law",
            "rights review",
            "constitutionalism",
            "rule of law",
            "liberty",
            "freedom of expression",
            "equal protection",
            "judicial review",
            "supreme court",
        ),
        title_aliases=(
            "constitutional",
            "human rights",
            "civil rights",
            "civil liberties",
            "public policy",
            "supreme court",
            "rule of law",
        ),
    ),
    LawSubfieldBucket(
        name="technology_privacy_ip",
        label="Technology / Privacy / Intellectual Property",
        aliases=(
            "artificial intelligence",
            "ai regulation",
            "digital law",
            "digital governance",
            "computer law",
            "cyber law",
            "cybersecurity",
            "data privacy",
            "data protection",
            "privacy law",
            "algorithmic",
            "platform governance",
            "technology law",
            "intellectual property",
            "copyright",
            "information law",
            "internet regulation",
        ),
        title_aliases=(
            "artificial intelligence",
            "computer",
            "security",
            "data privacy",
            "privacy",
            "copyright",
            "intellectual property",
            "technology",
        ),
    ),
    LawSubfieldBucket(
        name="regulation_governance_legislation",
        label="Regulation / Governance / Legislation",
        aliases=(
            "regulation",
            "regulatory",
            "governance",
            "legislation",
            "legislative",
            "rulemaking",
            "administrative law",
            "administrative governance",
            "public policy",
            "policy design",
            "institutional design",
            "regulatory state",
            "public administration",
        ),
        title_aliases=(
            "regulation",
            "governance",
            "legislation",
            "policy",
            "administrative",
        ),
    ),
    LawSubfieldBucket(
        name="family_labor_social_law",
        label="Family / Labor / Social Law",
        aliases=(
            "family law",
            "labor law",
            "labour law",
            "employment law",
            "industrial relations",
            "juvenile",
            "family court",
            "marriage",
            "workers",
            "dismissal",
            "social welfare",
            "social problems",
            "gender equality",
            "feminist legal",
        ),
        title_aliases=(
            "family",
            "labor",
            "labour",
            "industrial",
            "juvenile",
            "feminist",
            "social problems",
        ),
    ),
    LawSubfieldBucket(
        name="socio_legal_behavioral",
        label="Socio-Legal / Behavioral / Psychology",
        aliases=(
            "law and society",
            "social inquiry",
            "social science",
            "social studies",
            "behavioral",
            "psychology",
            "psychiatry and law",
            "human behavior",
            "empirical legal studies",
            "law in context",
            "socio-legal",
            "law and social science",
        ),
        title_aliases=(
            "social",
            "behavioral",
            "psychology",
            "human behavior",
            "empirical",
            "context",
        ),
    ),
    LawSubfieldBucket(
        name="legal_theory_history",
        label="Legal Theory / History / Education",
        aliases=(
            "legal theory",
            "jurisprudence",
            "philosophy of law",
            "law and philosophy",
            "legal history",
            "history of law",
            "historical",
            "legal analysis",
            "legal education",
            "law library",
            "legal studies",
            "legal scholarship",
        ),
        title_aliases=(
            "philosophy",
            "history",
            "historical",
            "education",
            "library",
            "legal studies",
            "analysis",
        ),
    ),
)


LAW_SUBFIELD_BUCKET_BY_NAME = {bucket.name: bucket for bucket in LAW_SUBFIELD_BUCKETS}

GENERAL_LAW_REVIEW_PATTERNS = (
    " law review",
    " law journal",
    " law quarterly",
    " law review-",
    " university law review",
)


def bucket_label(bucket_name: str | None) -> str:
    if not bucket_name:
        return ""
    if bucket_name == GENERAL_LAW_REVIEW_BUCKET:
        return "General Law Review"
    bucket = LAW_SUBFIELD_BUCKET_BY_NAME.get(bucket_name)
    return bucket.label if bucket else bucket_name.replace("_", " ").title()


def build_law_subfield_profile(
    *,
    title: str,
    keywords: list[str] | None = None,
    text_segments: list[str] | None = None,
    subdisciplines: list[str] | None = None,
) -> SubfieldProfile:
    keywords = keywords or []
    text_segments = text_segments or []
    subdisciplines = subdisciplines or []
    title_text = normalize_space(title).lower()
    keyword_text = normalize_space(" ".join([*keywords, *subdisciplines])).lower()
    body_text = normalize_space(" ".join(text_segments)).lower()

    scores: dict[str, float] = {}
    for bucket in LAW_SUBFIELD_BUCKETS:
        title_hits = _matched_aliases(title_text, [*bucket.title_aliases, *bucket.aliases])
        keyword_hits = _matched_aliases(keyword_text, bucket.aliases)
        body_hits = _matched_aliases(body_text, bucket.aliases)
        score = (
            (0.46 * _scaled_hit_score(len(title_hits), target=1))
            + (0.24 * _scaled_hit_score(len(keyword_hits), target=2))
            + (0.30 * _scaled_hit_score(len(body_hits), target=3))
        )
        if score >= 0.18:
            scores[bucket.name] = clamp(score)

    general_score = _general_law_review_score(title_text=title_text, keyword_text=keyword_text, specialized_scores=scores)
    if general_score > 0:
        scores[GENERAL_LAW_REVIEW_BUCKET] = general_score

    primary = _primary_bucket(scores)
    focus = _focus_buckets(scores)
    return SubfieldProfile(scores=scores, primary=primary, focus=focus)


def bucket_similarity(manuscript_profile: SubfieldProfile, journal_profile: SubfieldProfile) -> float:
    if not manuscript_profile.scores or not journal_profile.scores:
        return 0.0
    manuscript_normalized = _normalize_scores(manuscript_profile.scores)
    journal_normalized = _normalize_scores(journal_profile.scores)
    shared_buckets = set(manuscript_normalized) | set(journal_normalized)
    shared_strength = max(
        (min(manuscript_normalized.get(bucket, 0.0), journal_normalized.get(bucket, 0.0)) for bucket in shared_buckets),
        default=0.0,
    )
    overlap = sum(
        manuscript_normalized.get(bucket, 0.0) * journal_normalized.get(bucket, 0.0)
        for bucket in shared_buckets
    ) / max(1, len(manuscript_profile.focus))
    focus_overlap = 1.0 if any(bucket in journal_profile.focus for bucket in manuscript_profile.focus) else 0.0
    primary_match = 1.0 if manuscript_profile.primary and manuscript_profile.primary == journal_profile.primary else 0.0
    general_penalty = 0.0
    if (
        manuscript_profile.primary
        and manuscript_profile.primary != GENERAL_LAW_REVIEW_BUCKET
        and journal_profile.primary == GENERAL_LAW_REVIEW_BUCKET
        and journal_profile.focus == (GENERAL_LAW_REVIEW_BUCKET,)
    ):
        general_penalty = 0.18
    return clamp((0.38 * shared_strength) + (0.32 * overlap) + (0.18 * focus_overlap) + (0.12 * primary_match) - general_penalty)


def stage_one_bucket_score(manuscript_profile: SubfieldProfile, journal_profile: SubfieldProfile) -> float:
    bucket_fit = bucket_similarity(manuscript_profile, journal_profile)
    primary_bonus = 0.12 if manuscript_profile.primary and manuscript_profile.primary in journal_profile.focus else 0.0
    focus_bonus = 0.08 if any(bucket in journal_profile.focus for bucket in manuscript_profile.focus) else 0.0
    return clamp(bucket_fit + primary_bonus + focus_bonus)


def _general_law_review_score(
    *,
    title_text: str,
    keyword_text: str,
    specialized_scores: dict[str, float],
) -> float:
    if not title_text:
        return 0.0
    generic_title_hit = any(pattern in title_text for pattern in GENERAL_LAW_REVIEW_PATTERNS)
    if not generic_title_hit and " law review" not in keyword_text and " law journal" not in keyword_text:
        return 0.0
    top_specialized = max((score for bucket, score in specialized_scores.items() if bucket != GENERAL_LAW_REVIEW_BUCKET), default=0.0)
    base_score = 0.60 if generic_title_hit else 0.42
    if top_specialized >= 0.65:
        base_score *= 0.55
    elif top_specialized >= 0.45:
        base_score *= 0.72
    return clamp(base_score)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values(), default=0.0)
    if max_score <= 0:
        return {}
    return {bucket: value / max_score for bucket, value in scores.items()}


def _primary_bucket(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if (
        ordered[0][0] == GENERAL_LAW_REVIEW_BUCKET
        and len(ordered) > 1
        and ordered[1][1] >= ordered[0][1] - 0.10
    ):
        return ordered[1][0]
    return ordered[0][0]


def _focus_buckets(scores: dict[str, float]) -> tuple[str, ...]:
    if not scores:
        return (GENERAL_LAW_REVIEW_BUCKET,)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_score = ordered[0][1]
    threshold = max(0.28, top_score * 0.58)
    focus = [bucket for bucket, score in ordered if score >= threshold]
    if not focus:
        focus = [ordered[0][0]]
    if (
        GENERAL_LAW_REVIEW_BUCKET in scores
        and GENERAL_LAW_REVIEW_BUCKET not in focus
        and scores[GENERAL_LAW_REVIEW_BUCKET] >= 0.34
        and len(focus) < 3
    ):
        focus.append(GENERAL_LAW_REVIEW_BUCKET)
    return tuple(focus[:3])


def _matched_aliases(text: str, aliases: list[str] | tuple[str, ...]) -> set[str]:
    matched: set[str] = set()
    for alias in aliases:
        normalized_alias = normalize_space(alias).lower()
        if not normalized_alias:
            continue
        if _contains_alias(text, normalized_alias):
            matched.add(normalized_alias)
    return matched


def _contains_alias(text: str, alias: str) -> bool:
    if not text or not alias:
        return False
    if re.fullmatch(r"[a-z0-9\-& ]+", alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))
    return alias in text


def _scaled_hit_score(hit_count: int, *, target: int) -> float:
    if target <= 0:
        return 0.0
    return clamp(hit_count / target)
