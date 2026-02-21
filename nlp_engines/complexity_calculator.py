"""
nlp_engines/complexity_calculator.py
=====================================
Linguistic complexity metrics for tracking rhetoric "simplification" over time.

Metrics implemented:
  - Type-Token Ratio (TTR): vocabulary richness relative to total words
  - Corrected TTR (CTTR): TTR normalised for text length (more stable)
  - Average Sentence Length (ASL): words per sentence
  - Average Word Length (AWL): characters per word
  - Flesch Reading Ease (FRE): higher = easier to read
  - Flesch-Kincaid Grade Level (FKGL): US school grade equivalent
  - Gunning Fog Index (GFI): readability estimate

Note on multilingualism:
  Flesch-Kincaid was developed for English.  For other languages, the
  absolute values shift, but the *trend over time* within a single language
  corpus remains valid.  All scores are language-tagged in the output so
  cross-language comparisons are never made naively.
"""

from __future__ import annotations

import re
from collections import Counter

from loguru import logger


class ComplexityCalculator:
    """
    Computes linguistic complexity metrics from raw text.

    All metrics are returned as a dict[str, float] keyed by metric name,
    suitable for direct storage in a Neo4j Speech node.
    """

    def compute(self, text: str, language: str = "unk") -> dict[str, float]:
        """
        Compute all complexity metrics for the given text.

        Parameters
        ----------
        text:
            Raw speech text.
        language:
            BCP-47 language code (used for textstat locale setting).

        Returns
        -------
        dict[str, float]
            All computed metric values.
        """
        if not text or not text.strip():
            return {}

        try:
            import textstat

            # Set language for textstat's syllable counter
            textstat.textstat.set_lang(_map_lang(language))
        except ImportError:
            logger.warning("[Complexity] textstat not installed, using basic metrics only.")
            return self._basic_metrics(text)

        try:
            return {
                "ttr": self._ttr(text),
                "cttr": self._cttr(text),
                "avg_sentence_length": self._avg_sentence_length(text),
                "avg_word_length": self._avg_word_length(text),
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "lexicon_count": float(textstat.lexicon_count(text, removepunct=True)),
                "sentence_count": float(textstat.sentence_count(text)),
            }
        except Exception as e:
            logger.debug(f"[Complexity] textstat failed: {e}. Using basic metrics.")
            return self._basic_metrics(text)

    # ── Core metric implementations ───────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return [
            w.lower()
            for w in re.findall(r"\b[a-zA-ZÀ-ÿ]{2,}\b", text)
        ]

    def _ttr(self, text: str) -> float:
        """Type-Token Ratio: unique_words / total_words."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)

    def _cttr(self, text: str) -> float:
        """Carroll's Corrected TTR: unique / sqrt(2 * total)."""
        import math
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / math.sqrt(2 * len(tokens)), 4)

    def _avg_sentence_length(self, text: str) -> float:
        """Average number of words per sentence."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        word_counts = [len(s.split()) for s in sentences]
        return round(sum(word_counts) / len(word_counts), 2)

    def _avg_word_length(self, text: str) -> float:
        """Average word length in characters."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return round(sum(len(t) for t in tokens) / len(tokens), 2)

    def _basic_metrics(self, text: str) -> dict[str, float]:
        """Fallback metrics that don't require textstat."""
        return {
            "ttr": self._ttr(text),
            "cttr": self._cttr(text),
            "avg_sentence_length": self._avg_sentence_length(text),
            "avg_word_length": self._avg_word_length(text),
        }


def _map_lang(bcp47: str) -> str:
    """Map BCP-47 language code to textstat's language key."""
    _MAP = {
        "de": "de",
        "fr": "fr",
        "es": "es",
        "it": "it",
        "nl": "nl",
        "pl": "pl",
        "pt": "pt",
        "en": "en_US",
        "fi": "fi",
        "et": "et",
        "lv": "lv",
        "lt": "lt",
    }
    return _MAP.get(bcp47.lower().split("-")[0], "en_US")
