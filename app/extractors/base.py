"""Abstract extractor interface and shared utility methods."""

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Base contract for all format-specific text extractors."""

    @abstractmethod
    def extract(self, content: bytes) -> tuple[str, dict]:
        """Extract plain text and metadata from raw bytes."""

    def get_word_count(self, text: str) -> int:
        """Return approximate word count from normalized whitespace splitting."""

        return len([word for word in text.split() if word.strip()])
