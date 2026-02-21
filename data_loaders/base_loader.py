"""
data_loaders/base_loader.py
===========================
Abstract Base Class for all data source loaders.

Design principle (Clean Architecture):
    The pipeline ONLY depends on this interface.  Adding a new data source
    (Twitter, Europarl, Hansard, etc.) requires implementing this class —
    zero changes to the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator


from config.models import RawDocument


class BaseLoader(ABC):
    """
    Abstract loader contract.  All concrete loaders must implement
    `load()` and `get_metadata_schema()`.

    Usage
    -----
    >>> loader = ParlaMintLoader(country_code="DE")
    >>> for doc in loader.load("data/raw/ParlaMint-DE"):
    ...     process(doc)
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source, e.g. 'parlamint', 'twitter'."""
        ...

    @abstractmethod
    def load(self, source_path: str | Path) -> Iterator[RawDocument]:
        """
        Yield normalised RawDocument objects from the given source path.

        Parameters
        ----------
        source_path:
            A directory or file path (semantics are source-specific).

        Yields
        ------
        RawDocument
            One document per speech / post / article.
        """
        ...

    @abstractmethod
    def get_metadata_schema(self) -> Dict[str, type]:
        """
        Return a description of metadata fields this loader produces.

        Returns
        -------
        dict[str, type]
            e.g. {"session_id": str, "subcorpus": str | None, ...}
        """
        ...

    def validate_source(self, source_path: str | Path) -> Path:
        """Utility: ensure source path exists and return a Path object."""
        p = Path(source_path)
        if not p.exists():
            raise FileNotFoundError(f"Source path does not exist: {p}")
        return p

    def batch(self, source_path: str | Path, batch_size: int = 100) -> Iterator[list[RawDocument]]:
        """
        Convenience wrapper that yields fixed-size batches of RawDocuments.
        Useful for bulk Neo4j / Qdrant writes.
        """
        buf: list[RawDocument] = []
        for doc in self.load(source_path):
            buf.append(doc)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf
