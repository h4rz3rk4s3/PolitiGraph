"""
config/models.py
================
Shared Pydantic v2 domain models used across loaders, extractors,
graph connectors, and analyzers.  These are the canonical data contracts
for the PolitiGraph pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Raw document from any data source ────────────────────────────────────────

class RawDocument(BaseModel):
    """Normalised, source-agnostic document output by any BaseLoader."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # e.g. "parlamint", "twitter"
    country_code: str  # ISO 3166-1 alpha-2
    language: str  # BCP-47, e.g. "de", "fr"
    text: str
    speaker_id: str
    speaker_name: str
    party: str
    role: str  # e.g. "MP", "President", "Minister"
    date: date
    session_id: str
    subcorpus: str | None = None  # e.g. "COVID", "Regular"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── NLP Extraction outputs ────────────────────────────────────────────────────

class NamedEntity(BaseModel):
    """A single entity extracted by GLiNER (Track A)."""

    text: str
    label: str   # PERSON | ORGANIZATION | LOCATION | LAW_TREATY
    start: int
    end: int
    score: float


class Triple(BaseModel):
    """A semantic triple (subject, relation, object) extracted by LLM (Track B)."""

    subject: str
    relation: str
    obj: str  # 'object' is a Python builtin, use 'obj'
    subject_type: str | None = None
    obj_type: str | None = None


class AbstractTopic(BaseModel):
    """A high-level topic extracted/inferred by the LLM."""

    label: str     # e.g. "Immigration Control"
    keywords: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Full output of the hybrid NLP pipeline for a single document."""

    doc_id: str
    named_entities: list[NamedEntity] = Field(default_factory=list)
    topics: list[AbstractTopic] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    sentiment_score: float | None = None  # [-1.0, 1.0]
    complexity_metrics: dict[str, float] = Field(default_factory=dict)


# ── Graph node / edge schemas (mirrors Neo4j ontology) ───────────────────────

class PoliticianNode(BaseModel):
    politician_id: str  # ParlaMint speaker ID
    name: str
    party: str
    country_code: str
    role: str
    social_media_handles: dict[str, str] = Field(default_factory=dict)  # {"twitter": "@...", "mastodon": "..."}


class PoliticalPartyNode(BaseModel):
    party_id: str
    name: str
    country_code: str
    abbreviation: str | None = None
    political_family: str | None = None  # e.g. "EPP", "S&D", "ID"


class SpeechNode(BaseModel):
    speech_id: str
    session_id: str
    date: date
    timestamp: datetime
    language: str
    subcorpus: str | None = None
    word_count: int
    raw_text: str
    sentiment_score: float | None = None
    ttr: float | None = None               # Type-Token Ratio
    flesch_kincaid_grade: float | None = None
    avg_sentence_length: float | None = None


class TopicNode(BaseModel):
    topic_id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    keywords: list[str] = Field(default_factory=list)
    language: str = "multilingual"


class NamedEntityNode(BaseModel):
    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    label: str
    canonical_text: str | None = None  # after normalisation


# ── Social media stub (future extension) ─────────────────────────────────────

class SocialMediaPostNode(BaseModel):
    """Future node type for Twitter/X or Mastodon posts."""

    post_id: str
    platform: str   # "twitter", "mastodon", "bluesky"
    text: str
    timestamp: datetime
    politician_id: str
    language: str
    sentiment_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Embedding record ──────────────────────────────────────────────────────────

class EmbeddingRecord(BaseModel):
    doc_id: str
    vector: list[float]
    model_name: str
    party: str
    country_code: str
    year: int
    language: str
