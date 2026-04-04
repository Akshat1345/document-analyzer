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
ORG_SINGLE_WORD_STOPWORDS = {
    "company",
    "organization",
    "agency",
    "institution",
    "department",
    "office",
    "division",
    "team",
    "group",
    "limited",
    "inc",
    "ltd",
    "llc",
    "corp",
    "co",
    "ceo",
    "cto",
    "cfo",
    "cio",
    "hr",
    "it",
    "qa",
    "ui",
    "ux",
    "usa",
    "us",
    "uk",
    "eu",
    "un",
    "ai",
    "ml",
    "dl",
    "ny",
    "nyc",
}
NAME_NOISE_TERMS = {
    "software",
    "excel",
    "outlook",
    "word",
    "office",
    "onenote",
    "sharepoint",
    "concur",
    "catalyst",
    "kronos",
    "ultimate",
    "uitimate",
    "profile",
    "work",
    "experience",
}
ORG_NOISE_TERMS = {
    "figma",
    "adobe creative suite",
    "graphic designer",
    "skills",
    "portfolio",
    "interests",
    "bachelor",
    "fine arts",
    "social media campaign",
    "company corporate",
    "corporate",
    "graphic",
    "design",
    "photoshop",
    "brand",
    "social",
    "suite",
    "interests social media",
    "adobe creative suite",
}
ORG_HINTS = {
    "agency",
    "school",
    "university",
    "college",
    "institute",
    "company",
    "corp",
    "corporate",
    "inc",
    "ltd",
    "llc",
    "group",
    "bank",
    "co",
    "foundation",
    "association",
    "society",
    "hospital",
    "clinic",
    "council",
    "committee",
    "board",
    "department",
    "office",
    "center",
    "centre",
    "labs",
    "lab",
}
ORG_ACTION_WORDS = {
    "led",
    "designed",
    "managed",
    "improved",
    "boosted",
    "created",
    "developed",
    "building",
    "built",
    "combining",
    "combined",
    "driving",
    "driven",
    "delivering",
    "delivered",
    "supporting",
    "supported",
    "enabling",
    "enabled",
    "leveraging",
    "leveraged",
    "using",
    "used",
}
ORG_ROLE_WORDS = {
    "management",
    "improvement",
    "leadership",
    "coordination",
    "project",
    "manager",
    "designer",
    "ceo",
    "co-founder",
    "founder",
}
ORG_GENERIC_MULTIWORD_WORDS = {
    "global",
    "advanced",
    "technology",
    "technologies",
    "innovation",
    "industry",
    "industrial",
    "data",
    "science",
    "research",
    "engineering",
    "software",
    "hardware",
    "digital",
    "business",
    "enterprise",
    "solutions",
    "services",
    "system",
    "systems",
    "platform",
    "platforms",
    "security",
    "cybersecurity",
    "development",
    "analytics",
    "computing",
    "intelligence",
}
LOCATION_TERMS = {"new york", "york", "city", "cty", "manhattan", "la"}
LOCATION_TERMS = LOCATION_TERMS | {"brooklyn", "ny", "nyc"}
MONTH_PATTERN = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"


def _word_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"\s+", value) if token]


def _normalized_word_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9&]+", value.lower()) if token}


def _has_org_hint(value: str) -> bool:
    word_set = _normalized_word_set(value)
    return any(hint in word_set for hint in ORG_HINTS)


def _looks_like_brand_token(token: str) -> bool:
    if not token:
        return False
    return bool(
        re.fullmatch(r"[A-Z0-9&.-]{2,10}", token)
        or re.fullmatch(r"[A-Z][A-Za-z0-9&.'-]*[A-Z][A-Za-z0-9&.'-]*", token)
        or re.fullmatch(r"[A-Za-z]+\d+[A-Za-z0-9&.'-]*", token)
        or re.fullmatch(r"\d+[A-Za-z][A-Za-z0-9&.'-]*", token)
    )


def _looks_like_title_case_org(tokens: list[str]) -> bool:
    filtered = [token for token in tokens if token.lower() not in {"of", "and", "the", "for", "&"}]
    if not filtered:
        return False
    return all(
        re.fullmatch(r"[A-Z][A-Za-z0-9&.'-]*", token) is not None
        or re.fullmatch(r"[A-Z0-9&.-]{2,10}", token) is not None
        or _looks_like_brand_token(token)
        for token in filtered
    )


def _is_generic_org_phrase(tokens: list[str]) -> bool:
    normalized = {token.lower().strip(string.punctuation) for token in tokens if token.strip(string.punctuation)}
    return normalized and normalized.issubset(ORG_GENERIC_MULTIWORD_WORDS)


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
    if not (2 <= len(token_list) <= 3):
        return ""
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'\-.]*", token) for token in token_list):
        return ""

    tokens = {token.strip(string.punctuation) for token in lowered.split()}
    if lowered in COMMON_FALSE_NAME_WORDS or tokens & COMMON_FALSE_NAME_WORDS:
        return ""
    if any(token.lower() in {"manager", "designer", "agency", "company", "profile", "skills"} for token in token_list):
        return ""
    if any(len(token) == 1 for token in token_list):
        return ""
    if any(token.lower() in NAME_NOISE_TERMS for token in token_list):
        return ""
    return normalized


def normalize_date(date: str) -> str:
    """Preserve source date format and reject orphaned year numbers."""

    normalized = date.strip()
    lowered = normalized.lower()
    if any(ch.isalpha() for ch in lowered):
        months = {
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
        }
        words = re.findall(r"[a-zA-Z]+", lowered)
        if words and not any(word in months for word in words):
            return ""
    if re.search(r"[a-zA-Z]\d|\d[a-zA-Z]", normalized):
        return ""
    # Reject bare 4-digit numbers that don't look like years (1900-2099)
    if re.fullmatch(r"\d{4}", normalized):
        year_int = int(normalized)
        if year_int < 1900 or year_int > 2099:
            return ""
    return normalized


def normalize_amount(amount: str) -> str:
    """Preserve source currency format with whitespace normalization only."""

    return amount.strip()


def normalize_organization(org: str) -> str:
    """Normalize organization and reject common non-organization phrases."""

    normalized = re.sub(r"\s+", " ", org.strip().strip(string.punctuation))
    # OCR resumes often prefix orgs with role titles; strip those while keeping the company tail.
    normalized = re.sub(
        r"^(?:(?:Senior|Junior|Lead|Principal|Associate)\s+)?(?:[A-Z][a-z]+\s+){0,3}(?:Designer|Engineer|Developer|Manager|Analyst|Consultant|Specialist|Coordinator|Officer|Executive)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"^(?:Experience|Phone|Contact|Role)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:and|or|the|a|an|for|from|in|at|with)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+(?:and|or|the|a|an)$", "", normalized, flags=re.IGNORECASE)
    lowered = normalized.lower()
    if len(normalized) < 2:
        return ""
    if lowered in COMMON_GENERIC_ORG_WORDS:
        return ""
    if lowered.startswith("company ") and any(term in lowered for term in {"corporate", "profile", "project"}):
        return ""
    if len(normalized.split()) == 1 and lowered in {"co", "corp", "inc", "ltd", "llc"}:
        return ""
    if re.search(r"\d{3,}", normalized):
        return ""
    if re.search(r"[\d][^\s]*[\d]", normalized):
        return ""
    # Reject lines likely coming from OCR-garbled section headers.
    if re.search(r"[{}<>|\\/]{2,}", normalized):
        return ""
    if any(word in lowered for word in ORG_ACTION_WORDS):
        return ""
    if any(word in lowered for word in ORG_ROLE_WORDS):
        return ""
    if any(term in lowered for term in NAME_NOISE_TERMS):
        return ""
    if any(term in lowered for term in LOCATION_TERMS) and len(normalized.split()) > 3:
        return ""
    if any(term in lowered for term in ORG_NOISE_TERMS) and not _has_org_hint(normalized):
        return ""
    # Keep compact title-cased names and common suffixes.
    if len(normalized.split()) > 8:
        return ""
    words = _word_tokens(normalized)
    title_words = sum(1 for w in words if re.fullmatch(r"[A-Z][A-Za-z&.'-]*", w) is not None)
    if len(words) == 1:
        single = words[0]
        single_lower = single.lower()
        if (
            single_lower in COMMON_FALSE_NAME_WORDS
            or single_lower in COMMON_GENERIC_ORG_WORDS
            or single_lower in ORG_NOISE_TERMS
            or single_lower in NAME_NOISE_TERMS
            or single_lower in LOCATION_TERMS
            or single_lower in ORG_SINGLE_WORD_STOPWORDS
        ):
            return ""
        if _looks_like_brand_token(single) or (single[0].isupper() and len(single) >= 3):
            return normalized
        return ""
    if any(token.lower() in ORG_SINGLE_WORD_STOPWORDS for token in words):
        return ""
    if _is_generic_org_phrase(words):
        return ""
    if _has_org_hint(normalized):
        if title_words < 1:
            return ""
        return normalized
    if 2 <= len(words) <= 4 and _looks_like_title_case_org(words):
        return normalized
    if any(_looks_like_brand_token(token) for token in words):
        return normalized
    if title_words >= 2 and len(words) <= 5:
        return normalized
    return ""


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
            if not (2 <= len(tokens) <= 3):
                continue
            if any(t.casefold() in COMMON_FALSE_NAME_WORDS for t in tokens):
                continue
            if any(t.lower() in {"manager", "designer", "agency", "company", "profile", "skills"} for t in tokens):
                continue
            if any(t.lower() in NAME_NOISE_TERMS for t in tokens):
                continue
        elif entity_type == "organizations":
            if not normalize_organization(value):
                continue
        elif entity_type == "dates":
            if not any(ch.isdigit() for ch in value):
                continue
            # Reject phone/date mashups and overly noisy date spans.
            if re.search(r"\d{3}[-\s]?\d{4}", value):
                continue
            if len(value) > 30:
                continue
            lowered_date = value.lower()
            looks_like_month = re.search(MONTH_PATTERN, lowered_date) is not None
            looks_like_numeric = re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", value) is not None
            looks_like_year = re.fullmatch(r"(?:19|20)\d{2}", value.strip()) is not None
            if not (looks_like_month or looks_like_numeric or looks_like_year):
                continue
        elif entity_type == "amounts":
            if not any(ch.isdigit() for ch in value):
                continue
            has_currency_or_percent = any(sym in value for sym in ["$", "₹", "€", "£", "%"])
            has_grouped_amount = re.search(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", value) is not None
            has_large_suffix = re.search(r"\b\d+(?:\.\d+)?\s*(?:k|m|b|thousand|million|billion)\b", value, re.IGNORECASE) is not None
            if not (has_currency_or_percent or has_grouped_amount or has_large_suffix):
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
