"""
analyzers/longitudinal_analyzer.py
=====================================
Longitudinal analysis of political rhetoric using embeddings and graph data.

Features:
  1. Cosine Drift Analysis:   How the centroid of each party's embedding
                               cloud moves relative to others over time.
  2. Topic Salience:          Frequency and sentiment of specific topics
                               across the 2010–2025 timeline.
  3. Party Centroid Distance: Pairwise cosine distance between party
                               centroids per year — tracks ideological
                               convergence/divergence.
  4. Semantic Velocity:       Rate of change of a party's centroid vector
                               year-over-year (fast drift = rhetorical shift).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import cosine as cosine_distance

from embeddings.vector_store import VectorStore
from graph_connectors.neo4j_connector import Neo4jConnector


class LongitudinalAnalyzer:
    """
    Computes longitudinal rhetoric shift metrics from embeddings and graph data.

    Parameters
    ----------
    connector:
        Neo4j connector for fetching speech metadata.
    vector_store:
        Qdrant vector store for fetching embeddings.
    country_code:
        Focus country for analysis.
    start_year, end_year:
        Analysis time range.
    """

    def __init__(
        self,
        connector: Neo4jConnector,
        vector_store: VectorStore,
        country_code: str,
        start_year: int = 2010,
        end_year: int = 2024,
    ) -> None:
        self.connector = connector
        self.vector_store = vector_store
        self.country_code = country_code
        self.start_year = start_year
        self.end_year = end_year

    # ── Public Analysis Methods ───────────────────────────────────────────────

    def compute_party_centroids(self, year: int) -> dict[str, np.ndarray]:
        """
        Compute the mean embedding vector (centroid) for each party
        in a given year.

        Returns
        -------
        dict[party_name, centroid_vector]
        """
        vectors, payloads = self.vector_store.get_all_vectors_for_country(
            country_code=self.country_code,
            year=year,
        )
        if vectors.shape[0] == 0:
            logger.warning(f"[LongitudinalAnalyzer] No vectors for {self.country_code} in {year}")
            return {}

        # Group by party
        party_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        for vec, payload in zip(vectors, payloads):
            party = payload.get("party", "Unknown")
            party_vectors[party].append(vec)

        return {
            party: np.mean(np.array(vecs), axis=0)
            for party, vecs in party_vectors.items()
            if len(vecs) >= 5  # require at least 5 speeches for a stable centroid
        }

    def cosine_drift_over_time(
        self,
        party_a: str,
        party_b: str,
    ) -> pd.DataFrame:
        """
        Track the cosine distance between two parties' centroids year-by-year.

        This is the primary metric for detecting ideological convergence
        (distance decreases) or divergence/radicalisation (distance increases).

        Returns
        -------
        pd.DataFrame with columns: [year, party_a, party_b, cosine_distance]
        """
        rows = []
        for year in range(self.start_year, self.end_year + 1):
            centroids = self.compute_party_centroids(year)
            if party_a in centroids and party_b in centroids:
                dist = cosine_distance(centroids[party_a], centroids[party_b])
                rows.append({
                    "year": year,
                    "party_a": party_a,
                    "party_b": party_b,
                    "cosine_distance": round(float(dist), 6),
                })
            else:
                missing = [p for p in [party_a, party_b] if p not in centroids]
                logger.debug(f"[Drift] {year}: missing parties {missing}")

        df = pd.DataFrame(rows)
        logger.info(
            f"[Drift] Computed cosine drift {party_a} ↔ {party_b}: {len(df)} years"
        )
        return df

    def semantic_velocity(self, party: str) -> pd.DataFrame:
        """
        Compute the year-over-year rate of change of a party's centroid.

        High semantic velocity = rapid rhetorical shift.
        Tracks how fast a party is moving through semantic space.

        Returns
        -------
        pd.DataFrame with columns: [year, party, velocity]
        """
        centroids: dict[int, np.ndarray] = {}
        for year in range(self.start_year, self.end_year + 1):
            year_centroids = self.compute_party_centroids(year)
            if party in year_centroids:
                centroids[year] = year_centroids[party]

        rows = []
        years = sorted(centroids.keys())
        for i in range(1, len(years)):
            y1, y2 = years[i - 1], years[i]
            velocity = cosine_distance(centroids[y1], centroids[y2])
            rows.append({
                "year": y2,
                "party": party,
                "velocity": round(float(velocity), 6),
                "year_from": y1,
                "year_to": y2,
            })

        df = pd.DataFrame(rows)
        logger.info(f"[Velocity] Computed semantic velocity for {party}: {len(df)} transitions")
        return df

    def pairwise_centroid_distances(self, year: int) -> pd.DataFrame:
        """
        Compute all pairwise cosine distances between party centroids
        for a given year.  Produces an ideological distance matrix.

        Returns
        -------
        pd.DataFrame (wide format): rows = parties, cols = parties, values = distances
        """
        centroids = self.compute_party_centroids(year)
        parties = sorted(centroids.keys())

        matrix = np.zeros((len(parties), len(parties)))
        for i, p1 in enumerate(parties):
            for j, p2 in enumerate(parties):
                if i != j:
                    matrix[i, j] = cosine_distance(centroids[p1], centroids[p2])

        df = pd.DataFrame(matrix, index=parties, columns=parties)
        logger.info(f"[PairwiseDistance] Computed {len(parties)}x{len(parties)} matrix for {year}")
        return df

    def topic_salience_over_time(self, topic_label: str) -> pd.DataFrame:
        """
        Track how often a specific topic appears, and its average sentiment,
        across the full timeline.

        Query is run directly against Neo4j for efficiency.

        Returns
        -------
        pd.DataFrame with columns: [year, party, mention_count, avg_sentiment]
        """
        query = """
        MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t:Topic {label: $topic_label})
        WHERE s.country_code = $country_code
          AND s.year >= $start_year
          AND s.year <= $end_year
        RETURN
            s.year                AS year,
            s.party               AS party,
            count(s)              AS mention_count,
            avg(s.sentiment_score) AS avg_sentiment
        ORDER BY s.year ASC, s.party ASC
        """
        with self.connector._session() as session:
            result = session.run(
                query,
                topic_label=topic_label,
                country_code=self.country_code,
                start_year=self.start_year,
                end_year=self.end_year,
            )
            rows = [dict(r) for r in result]

        df = pd.DataFrame(rows)
        logger.info(
            f"[TopicSalience] '{topic_label}': {len(df)} year-party data points"
        )
        return df

    def complexity_trends(self) -> pd.DataFrame:
        """
        Retrieve and aggregate linguistic complexity metrics per party per year.

        Tracks the "simplification" of rhetoric hypothesis:
        if populist parties show decreasing FK grade and TTR over time,
        this is quantitative evidence of rhetoric simplification.

        Returns
        -------
        pd.DataFrame with columns: [year, party, avg_ttr, avg_fk_grade,
                                     avg_sentence_length, speech_count]
        """
        query = """
        MATCH (pol:Politician)-[:DELIVERED_SPEECH]->(s:Speech)
        WHERE s.country_code = $country_code
          AND s.year >= $start_year
          AND s.year <= $end_year
          AND s.ttr IS NOT NULL
        RETURN
            s.year                          AS year,
            s.party                         AS party,
            avg(s.ttr)                      AS avg_ttr,
            avg(s.flesch_kincaid_grade)     AS avg_fk_grade,
            avg(s.avg_sentence_length)      AS avg_sentence_length,
            avg(s.gunning_fog)              AS avg_gunning_fog,
            count(s)                        AS speech_count
        ORDER BY s.year ASC, s.party ASC
        """
        with self.connector._session() as session:
            result = session.run(
                query,
                country_code=self.country_code,
                start_year=self.start_year,
                end_year=self.end_year,
            )
            rows = [dict(r) for r in result]

        df = pd.DataFrame(rows)
        logger.info(f"[Complexity] Retrieved trends for {len(df)} year-party combinations")
        return df

    def sentiment_trends(self) -> pd.DataFrame:
        """
        Average sentiment score per party per year.

        Returns
        -------
        pd.DataFrame with columns: [year, party, avg_sentiment, speech_count, std_sentiment]
        """
        query = """
        MATCH (s:Speech)
        WHERE s.country_code = $country_code
          AND s.year >= $start_year
          AND s.year <= $end_year
          AND s.sentiment_score IS NOT NULL
        RETURN
            s.year                      AS year,
            s.party                     AS party,
            avg(s.sentiment_score)      AS avg_sentiment,
            stdev(s.sentiment_score)    AS std_sentiment,
            count(s)                    AS speech_count
        ORDER BY s.year ASC, s.party ASC
        """
        with self.connector._session() as session:
            result = session.run(
                query,
                country_code=self.country_code,
                start_year=self.start_year,
                end_year=self.end_year,
            )
            rows = [dict(r) for r in result]

        return pd.DataFrame(rows)
