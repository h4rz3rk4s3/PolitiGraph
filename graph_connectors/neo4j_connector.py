"""
graph_connectors/neo4j_connector.py
=====================================
Neo4j driver wrapper with idempotent MERGE operations for all node
and edge types defined in the PolitiGraph ontology.

Design decisions:
  - Uses the official neo4j Python driver (v5.x) with connection pooling.
  - All writes use MERGE (not CREATE) to ensure idempotency — safe to re-run.
  - Batch operations use explicit transactions for performance.
  - Schema initialisation (constraints + indexes) runs automatically on connect.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Generator

from loguru import logger
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

from config.models import (
    ExtractionResult,
    NamedEntityNode,
    PoliticalPartyNode,
    PoliticianNode,
    RawDocument,
    SpeechNode,
    TopicNode,
)
from config.settings import get_settings


class Neo4jConnector:
    """
    Manages all interactions with the Neo4j knowledge graph.

    Usage
    -----
    >>> conn = Neo4jConnector()
    >>> conn.upsert_speech(speech_node, politician_node, party_node)
    >>> conn.upsert_extraction(speech_id, extraction_result)
    >>> conn.close()

    Or as a context manager:
    >>> with Neo4jConnector() as conn:
    ...     conn.upsert_speech(...)
    """

    def __init__(self) -> None:
        settings = get_settings().neo4j
        self._driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password),
            max_connection_pool_size=50,
        )
        logger.info(f"[Neo4j] Connected to {settings.uri}")
        self._init_schema()

    def __enter__(self) -> "Neo4jConnector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        with self._driver.session() as session:
            yield session

    # ── Schema initialisation ─────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create uniqueness constraints and indexes (idempotent)."""
        constraints = [
            ("Politician", "politician_id"),
            ("PoliticalParty", "party_id"),
            ("Speech", "speech_id"),
            ("Topic", "label"),
            ("NamedEntity", "entity_id"),
        ]
        indexes = [
            ("Speech", "date"),
            ("Speech", "party"),
            ("Speech", "country_code"),
            ("Speech", "language"),
            ("Politician", "country_code"),
        ]
        with self._session() as session:
            for label, prop in constraints:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
            for label, prop in indexes:
                session.run(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.{prop})"
                )
            # Temporal index for longitudinal queries
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:Speech) ON (n.year, n.party)"
            )
            logger.info("[Neo4j] Schema constraints and indexes verified.")

    # ── Node upserts ──────────────────────────────────────────────────────────

    def upsert_politician(self, node: PoliticianNode) -> None:
        query = """
        MERGE (p:Politician {politician_id: $politician_id})
        ON CREATE SET
            p.name = $name,
            p.party = $party,
            p.country_code = $country_code,
            p.role = $role,
            p.social_media_handles = $social_media_handles,
            p.created_at = datetime()
        ON MATCH SET
            p.name = $name,
            p.party = $party,
            p.role = $role
        """
        with self._session() as s:
            s.run(query, **node.model_dump())

    def upsert_party(self, node: PoliticalPartyNode) -> None:
        query = """
        MERGE (p:PoliticalParty {party_id: $party_id})
        ON CREATE SET
            p.name = $name,
            p.country_code = $country_code,
            p.abbreviation = $abbreviation,
            p.political_family = $political_family
        ON MATCH SET
            p.name = $name
        """
        with self._session() as s:
            s.run(query, **node.model_dump())

    def upsert_speech(
        self,
        doc: RawDocument,
        speech_node: SpeechNode,
    ) -> None:
        """
        Upsert a Speech node and create edges to Politician and PoliticalParty.
        Also ensures Politician and PoliticalParty nodes exist.
        """
        query = """
        // Upsert Speech node
        MERGE (s:Speech {speech_id: $speech_id})
        ON CREATE SET
            s.session_id     = $session_id,
            s.date           = date($date_str),
            s.timestamp      = datetime($timestamp_str),
            s.year           = $year,
            s.language       = $language,
            s.subcorpus      = $subcorpus,
            s.word_count     = $word_count,
            s.raw_text       = $raw_text,
            s.country_code   = $country_code,
            s.party          = $party

        // Upsert Politician node
        MERGE (pol:Politician {politician_id: $speaker_id})
        ON CREATE SET
            pol.name         = $speaker_name,
            pol.party        = $party,
            pol.country_code = $country_code,
            pol.role         = $role,
            pol.social_media_handles = {}

        // Upsert PoliticalParty node
        MERGE (pp:PoliticalParty {party_id: $party_id})
        ON CREATE SET
            pp.name          = $party,
            pp.country_code  = $country_code

        // Politician → Party membership
        MERGE (pol)-[:MEMBER_OF]->(pp)

        // Politician → Speech
        MERGE (pol)-[ds:DELIVERED_SPEECH]->(s)
        ON CREATE SET ds.date = date($date_str)

        // Speech → PlenarySession
        MERGE (sess:PlenarySession {session_id: $session_id})
        ON CREATE SET
            sess.date        = date($date_str),
            sess.country_code = $country_code
        MERGE (s)-[:PART_OF_SESSION]->(sess)
        """
        dt = speech_node.date
        params = {
            "speech_id": speech_node.speech_id,
            "session_id": speech_node.session_id,
            "date_str": str(dt),
            "timestamp_str": speech_node.timestamp.isoformat(),
            "year": dt.year,
            "language": speech_node.language,
            "subcorpus": speech_node.subcorpus,
            "word_count": speech_node.word_count,
            "raw_text": speech_node.raw_text[:10_000],  # cap to avoid huge nodes
            "country_code": doc.country_code,
            "party": doc.party,
            "party_id": f"{doc.country_code}_{doc.party.replace(' ', '_')}",
            "speaker_id": doc.speaker_id,
            "speaker_name": doc.speaker_name,
            "role": doc.role,
        }
        with self._session() as s:
            s.run(query, **params)

    def upsert_extraction(
        self, speech_id: str, result: ExtractionResult
    ) -> None:
        """
        Store NLP extraction results: sentiment, complexity, topics,
        named entities, and semantic triples.
        """
        with self._session() as session:
            with session.begin_transaction() as tx:
                # Update Speech node with computed metrics
                tx.run(
                    """
                    MATCH (s:Speech {speech_id: $speech_id})
                    SET s.sentiment_score       = $sentiment,
                        s.ttr                   = $ttr,
                        s.flesch_kincaid_grade  = $fk_grade,
                        s.avg_sentence_length   = $asl,
                        s.gunning_fog           = $fog
                    """,
                    speech_id=speech_id,
                    sentiment=result.sentiment_score,
                    ttr=result.complexity_metrics.get("ttr"),
                    fk_grade=result.complexity_metrics.get("flesch_kincaid_grade"),
                    asl=result.complexity_metrics.get("avg_sentence_length"),
                    fog=result.complexity_metrics.get("gunning_fog"),
                )

                # Topics
                for topic in result.topics:
                    tx.run(
                        """
                        MERGE (t:Topic {label: $label})
                        ON CREATE SET t.keywords = $keywords
                        WITH t
                        MATCH (s:Speech {speech_id: $speech_id})
                        MERGE (s)-[:MENTIONS_TOPIC]->(t)
                        """,
                        label=topic.label,
                        keywords=topic.keywords,
                        speech_id=speech_id,
                    )

                # Named Entities
                for entity in result.named_entities:
                    entity_id = f"{entity.label}_{entity.text.lower().replace(' ', '_')}"
                    tx.run(
                        """
                        MERGE (e:NamedEntity {entity_id: $entity_id})
                        ON CREATE SET
                            e.text  = $text,
                            e.label = $label
                        WITH e
                        MATCH (s:Speech {speech_id: $speech_id})
                        MERGE (s)-[:REFERENCES_ENTITY {score: $score}]->(e)
                        """,
                        entity_id=entity_id,
                        text=entity.text,
                        label=entity.label,
                        score=entity.score,
                        speech_id=speech_id,
                    )

                # Semantic Triples → store as generic Relationship nodes
                for triple in result.triples:
                    tx.run(
                        """
                        MATCH (s:Speech {speech_id: $speech_id})
                        MERGE (subj:Concept {text: $subject})
                        MERGE (obj:Concept  {text: $obj})
                        MERGE (subj)-[r:SEMANTIC_RELATION {
                            relation: $relation,
                            speech_id: $speech_id
                        }]->(obj)
                        MERGE (s)-[:CONTAINS_TRIPLE]->(subj)
                        """,
                        speech_id=speech_id,
                        subject=triple.subject,
                        relation=triple.relation,
                        obj=triple.obj,
                    )

                tx.commit()

    # ── Vector embedding update ───────────────────────────────────────────────

    def store_embedding_in_node(
        self, speech_id: str, embedding: list[float], model_name: str
    ) -> None:
        """
        Store embedding vector directly in the Speech node.
        Enables Neo4j native vector search (Neo4j 5.x+).
        Supplement with Qdrant for more advanced ANN queries.
        """
        with self._session() as s:
            s.run(
                """
                MATCH (s:Speech {speech_id: $speech_id})
                CALL db.create.setNodeVectorProperty(s, 'embedding', $embedding)
                SET s.embedding_model = $model_name
                """,
                speech_id=speech_id,
                embedding=embedding,
                model_name=model_name,
            )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_speeches_for_analysis(
        self,
        country_code: str,
        start_year: int = 2010,
        end_year: int = 2025,
    ) -> list[dict[str, Any]]:
        """
        Fetch speech records with metadata for the analysis layer.
        Returns lightweight dicts (no raw_text) to keep memory usage low.
        """
        query = """
        MATCH (pol:Politician)-[:DELIVERED_SPEECH]->(s:Speech)
        WHERE s.country_code = $country_code
          AND s.year >= $start_year
          AND s.year <= $end_year
        RETURN
            s.speech_id          AS speech_id,
            s.year               AS year,
            s.date               AS date,
            s.party              AS party,
            s.sentiment_score    AS sentiment_score,
            s.ttr                AS ttr,
            s.flesch_kincaid_grade AS flesch_kincaid_grade,
            s.avg_sentence_length  AS avg_sentence_length,
            s.word_count         AS word_count,
            s.language           AS language,
            pol.politician_id    AS politician_id,
            pol.name             AS politician_name
        ORDER BY s.date ASC
        """
        with self._session() as session:
            result = session.run(
                query,
                country_code=country_code,
                start_year=start_year,
                end_year=end_year,
            )
            return [dict(record) for record in result]

    def get_unprocessed_speeches(
        self, country_code: str, batch_size: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch speeches that haven't been through NLP extraction yet."""
        query = """
        MATCH (s:Speech)
        WHERE s.country_code = $country_code
          AND s.sentiment_score IS NULL
        RETURN s.speech_id AS speech_id, s.raw_text AS raw_text,
               s.language AS language
        LIMIT $batch_size
        """
        with self._session() as session:
            result = session.run(
                query, country_code=country_code, batch_size=batch_size
            )
            return [dict(r) for r in result]
