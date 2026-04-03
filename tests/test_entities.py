"""Entity normalization and filtering tests."""

from app.processors.entity_normalizer import (
    deduplicate_exact_casefold,
    deduplicate_fuzzy,
    filter_false_positives,
    normalize_name,
)


def test_normalize_name_filters_noise() -> None:
    """Single-char and generic noise tokens should be removed."""

    assert normalize_name("1") == ""
    assert normalize_name("inc") == ""
    assert normalize_name("Resource Coordination") == ""
    assert normalize_name("Matt Connors") == "Matt Connors"


def test_deduplicate_fuzzy_removes_close_duplicates() -> None:
    """Fuzzy dedup should keep one representative for near-identical phrases."""

    values = ["Microsoft", "Micro soft", "NVIDIA"]
    out = deduplicate_fuzzy(values)
    assert "NVIDIA" in out
    assert len(out) <= 3


def test_exact_casefold_dedup_for_sensitive_identifiers() -> None:
    """Email/phone dedup should use exact case-insensitive identity."""

    values = ["Jane@Email.com", "jane@email.com", "john@email.com"]
    out = deduplicate_exact_casefold(values)
    assert out == ["Jane@Email.com", "john@email.com"]


def test_filter_false_positives_for_names() -> None:
    """Name filtering should remove noisy tokens and keep plausible names."""

    values = ["Interal", "Project", "John Smith", "1234"]
    out = filter_false_positives(values, "names")
    assert "John Smith" in out
    assert "Interal" not in out
    assert "1234" not in out
