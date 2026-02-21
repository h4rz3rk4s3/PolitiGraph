"""
graph_connectors/schema.py
===========================
Ontology documentation and Cypher schema setup for the PolitiGraph
Knowledge Graph.

Ontological foundations:
  - FOAF (Friend of a Friend):   Politician → foaf:Person
  - ORG  (Organization Ontology): PoliticalParty → org:Organization
  - PODIO (Political Discourse Ontology): Speech, Topic, rhetorical relations
  - ParlaMint TEI metadata maps directly to Speech / PlenarySession nodes

Node Type Reference:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Label            │ Key Property    │ Ontology Mapping                 │
  ├──────────────────┼─────────────────┼──────────────────────────────────┤
  │ :Politician      │ politician_id   │ foaf:Person                      │
  │ :PoliticalParty  │ party_id        │ org:Organization                 │
  │ :PlenarySession  │ session_id      │ podio:PlenarySession             │
  │ :Speech          │ speech_id       │ podio:DiscourseUnit              │
  │ :Topic           │ label           │ podio:Topic / skos:Concept       │
  │ :NamedEntity     │ entity_id       │ podio:NamedEntity                │
  │ :Concept         │ text            │ podio:AbstractConcept            │
  │ :SocialMediaPost │ post_id         │ podio:SocialMediaPost (future)   │
  └──────────────────────────────────────────────────────────────────────┘

Edge Reference:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Edge Type              │ From          │ To              │ Properties   │
  ├────────────────────────┼───────────────┼─────────────────┼──────────────┤
  │ :MEMBER_OF             │ Politician    │ PoliticalParty  │ since        │
  │ :DELIVERED_SPEECH      │ Politician    │ Speech          │ date         │
  │ :PART_OF_SESSION       │ Speech        │ PlenarySession  │ -            │
  │ :MENTIONS_TOPIC        │ Speech        │ Topic           │ count        │
  │ :REFERENCES_ENTITY     │ Speech        │ NamedEntity     │ score        │
  │ :CONTAINS_TRIPLE       │ Speech        │ Concept         │ -            │
  │ :SEMANTIC_RELATION     │ Concept       │ Concept         │ relation     │
  │ :MADE_POST (future)    │ Politician    │ SocialMediaPost │ platform     │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

# Full Cypher DDL for schema creation (use in migrations or reset scripts)
SCHEMA_CYPHER = """
// ── Uniqueness Constraints ──────────────────────────────────────────────────
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Politician)     REQUIRE n.politician_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:PoliticalParty) REQUIRE n.party_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Speech)         REQUIRE n.speech_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:PlenarySession) REQUIRE n.session_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Topic)          REQUIRE n.label IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:NamedEntity)    REQUIRE n.entity_id IS UNIQUE;

// ── Performance Indexes ─────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.date);
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.year);
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.party);
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.country_code);
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.language);
CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.year, n.party);
CREATE INDEX IF NOT EXISTS FOR (n:Politician) ON (n.country_code);
CREATE INDEX IF NOT EXISTS FOR (n:NamedEntity) ON (n.label);

// ── Vector Index (Neo4j 5.x native, for fallback search) ───────────────────
// Note: dimensionality depends on the embedding model:
//   multilingual-e5-large: 1024 dims
//   multilingual-e5-base:   768 dims
//   xlm-roberta-large:      1024 dims
CREATE VECTOR INDEX IF NOT EXISTS speech_embeddings
FOR (n:Speech) ON (n.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
}};
"""

# Future: SocialMediaPost node extension (wired up when Twitter loader is ready)
SOCIAL_MEDIA_SCHEMA_CYPHER = """
CREATE CONSTRAINT IF NOT EXISTS FOR (n:SocialMediaPost) REQUIRE n.post_id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (n:SocialMediaPost) ON (n.platform);
CREATE INDEX IF NOT EXISTS FOR (n:SocialMediaPost) ON (n.timestamp);
"""

# Property definitions for documentation / validation
NODE_PROPERTIES: dict[str, dict[str, str]] = {
    "Politician": {
        "politician_id": "str — ParlaMint speaker ID, e.g. 'ParlaMint-DE_Anna.Schmidt.1'",
        "name": "str — full name",
        "party": "str — party name",
        "country_code": "str — ISO 3166-1 alpha-2",
        "role": "str — parliamentary role, e.g. 'MP', 'Minister'",
        "social_media_handles": "map — {platform: handle}, e.g. {'twitter': '@example'}",
    },
    "PoliticalParty": {
        "party_id": "str — {country_code}_{party_name_normalised}",
        "name": "str — full party name",
        "country_code": "str",
        "abbreviation": "str? — e.g. 'CDU', 'SPD'",
        "political_family": "str? — e.g. 'EPP', 'S&D', 'ID'",
    },
    "Speech": {
        "speech_id": "str — unique utterance ID from ParlaMint XML",
        "session_id": "str — plenary session ID",
        "date": "date — YYYY-MM-DD",
        "timestamp": "datetime — for time-series ordering",
        "year": "int — extracted for fast range queries",
        "language": "str — BCP-47",
        "subcorpus": "str? — e.g. 'COVID', 'REGULAR'",
        "word_count": "int",
        "raw_text": "str — first 10,000 chars",
        "sentiment_score": "float? — [-1.0, 1.0], set after NLP extraction",
        "ttr": "float? — Type-Token Ratio",
        "flesch_kincaid_grade": "float? — FK grade level",
        "avg_sentence_length": "float? — words per sentence",
        "gunning_fog": "float? — Gunning Fog readability index",
        "embedding": "float[] — 1024-dim embedding vector (set after embedding step)",
        "embedding_model": "str? — name of model used",
        "country_code": "str",
        "party": "str — denormalised for fast aggregation queries",
    },
    "Topic": {
        "label": "str — canonical topic label, e.g. 'Climate Policy'",
        "keywords": "str[] — associated keywords",
    },
    "NamedEntity": {
        "entity_id": "str — {LABEL}_{normalised_text}",
        "text": "str — surface form",
        "label": "str — PERSON | ORGANIZATION | LOCATION | LAW_TREATY | ...",
        "canonical_text": "str? — normalised/linked form",
    },
    "SocialMediaPost": {
        "post_id": "str — platform-specific ID",
        "platform": "str — 'twitter' | 'mastodon' | 'bluesky'",
        "text": "str",
        "timestamp": "datetime",
        "politician_id": "str — FK to Politician.politician_id",
        "language": "str",
        "sentiment_score": "float?",
    },
}
