"""Normalization and fuzzy deduplication utilities for extracted entities."""

import re
import string

from fuzzywuzzy import fuzz

COMMON_FALSE_NAME_WORDS = {
    "the",
    "this",
    "they",
    "it",
    "we",
    "inc",
    "ltd",
    "analysis",
    "innovation",
    "report",
    "breach",
    "incident",
    "technology",
    "artificial",
    "intelligence",
    "cybersecurity",
    "industry",
    "data",
    "major",
    "present",
    "profile",
    "skills",
    "software",
    "languages",
    "experience",
    "references",
    "resource",
    "coordination",
    "coordinalion",
    "strategic",
    "planning",
    "leadership",
    "internal",
    "interal",
}
COMMON_GENERIC_ORG_WORDS = {"company", "organization", "agency", "institution", "limited", "inc", "ltd"}


def normalize_name(name: str) -> str:
    """Normalize a person name string and remove common invalid forms."""

    normalized = name.strip().strip(string.punctuation).title()
    lowered = normalized.lower()
    if len(normalized) < 2:
        return ""
    if any(ch.isdigit() for ch in normalized):
        return ""
    if re.fullmatch(r"\d+", normalized):
        return ""
    if len(normalized.split()) == 1 and lowered in {"inc", "ltd"}:
        return ""
    token_list = [token.strip(string.punctuation) for token in normalized.split() if token.strip(string.punctuation)]
    if not (2 <= len(token_list) <= 4):
        return ""
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'\-.]*", token) for token in token_list):
        return ""

    tokens = {token.strip(string.punctuation) for token in lowered.split()}
    if lowered in COMMON_FALSE_NAME_WORDS or tokens & COMMON_FALSE_NAME_WORDS:
        return ""
    return normalized


def normalize_date(date: str) -> str:
    """Preserve source date format and reject orphaned year numbers."""

    normalized = date.strip()
    # Reject bare 4-digit numbers that don't look like years (1900-2099)
    if re.fullmatch(r"\d{4}", normalized):
        year_int = int(normalized)
        if year_int < 1900 or year_int > 2099:
            return ""
    return normalized


def normalize_amount(amount: str) -> str:
    """Preserve source currency format with whitespace normalization only."""

    return amount.strip()


def deduplicate_fuzzy(items: list[str]) -> list[str]:
    """Deduplicate entity values using fuzzy ratio while preserving order."""

    kept: list[str] = []
    for item in items:
        candidate = item.strip()
        if not candidate:
            continue
        if any(fuzz.ratio(candidate.lower(), existing.lower()) > 85 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def deduplicate_exact_casefold(items: list[str]) -> list[str]:
    """Deduplicate strings by exact case-insensitive identity while preserving order."""

    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        candidate = item.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
    return kept


def filter_false_positives(entities: list[str], entity_type: str) -> list[str]:
    """Filter noisy entities based on lightweight type-specific heuristics."""

    filtered: list[str] = []
    for raw in entities:
        value = raw.strip()
        if not value:
            continue

        lowered = value.lower()
        if entity_type == "names":
            if len(value) < 2 or re.fullmatch(r"\d+", value) or lowered in COMMON_FALSE_NAME_WORDS:
                continue
            if any(ch.isdigit() for ch in value):
                continue
            tokens = [t for t in re.split(r"\s+", value) if t]
            if not (2 <= len(tokens) <= 4):
                continue
            if any(t.casefold() in COMMON_FALSE_NAME_WORDS for t in tokens):
                continue
        elif entity_type == "organizations":
            if lowered in COMMON_GENERIC_ORG_WORDS:
                continue
        elif entity_type == "dates":
            if not any(ch.isdigit() for ch in value):
                continue
        elif entity_type == "amounts":
            if not any(ch.isdigit() for ch in value):
                continue
        elif entity_type == "emails":
            if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
                continue
        elif entity_type == "phones":
            digits = re.sub(r"\D", "", value)
            if len(digits) < 10 or len(digits) > 15:
                continue

        filtered.append(value)

    return filtered
