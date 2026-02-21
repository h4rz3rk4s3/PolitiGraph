"""
embeddings/vector_store.py
===========================
Qdrant vector store adapter.

Architecture decision:
  - Qdrant is the primary vector layer (mature HNSW indexing, filtering, namespaces).
  - Neo4j's built-in vector index serves as a secondary fallback for
    graph-native similarity queries (e.g. "find speeches similar to X that
    were delivered by politicians from party Y in 2018").
  - This class handles ONLY Qdrant operations.
  - Neo4j vector operations are in graph_connectors/neo4j_connector.py.

Collection schema:
  - Collection name: politigraph_speeches (configurable)
  - Vector: 1024-dim float32 (mE5-large) or 768-dim (mE5-base / XLM-R)
  - Payload: {doc_id, party, country_code, year, language, model_name}
"""

from __future__ import annotations

from typing import Any
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from loguru import logger

from config.models import EmbeddingRecord
from config.settings import get_settings


def _stable_uuid(doc_id: str) -> str:
    """Generate a stable UUID from a doc_id string (Qdrant requires UUID point IDs)."""
    return str(uuid5(NAMESPACE_URL, doc_id))


class VectorStore:
    """
    Qdrant vector store for speech embeddings.

    Parameters
    ----------
    collection_name:
        Qdrant collection name. Defaults to settings.
    vector_size:
        Embedding dimensionality. Must match the embedding model.
    recreate_collection:
        If True, drops and recreates the collection on init.
        Useful for development; set to False in production.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        vector_size: int = 1024,
        recreate_collection: bool = False,
    ) -> None:
        settings = get_settings()
        self._collection = collection_name or settings.qdrant.collection
        self._vector_size = vector_size

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            timeout=30,
        )
        self._ensure_collection(recreate=recreate_collection)

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self._client.get_collections().collections]

        if recreate and self._collection in collections:
            self._client.delete_collection(self._collection)
            logger.info(f"[Qdrant] Dropped collection: {self._collection}")

        if self._collection not in collections or recreate:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                    on_disk=True,  # store vectors on disk for large corpora
                ),
            )
            # Payload indexes for fast filtering
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="year",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="party",
                field_schema="keyword",
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="country_code",
                field_schema="keyword",
            )
            logger.info(f"[Qdrant] Created collection: {self._collection} ({self._vector_size}d)")
        else:
            logger.debug(f"[Qdrant] Collection exists: {self._collection}")

    def upsert_batch(self, records: list[EmbeddingRecord]) -> None:
        """
        Upsert a batch of EmbeddingRecord objects into Qdrant.

        Uses Qdrant's native batch upsert for efficiency.
        Point IDs are stable UUIDs derived from doc_id (idempotent).
        """
        if not records:
            return

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_stable_uuid(r.doc_id),
                vector=r.vector,
                payload={
                    "doc_id": r.doc_id,
                    "party": r.party,
                    "country_code": r.country_code,
                    "year": r.year,
                    "language": r.language,
                    "model_name": r.model_name,
                },
            )
            for r in records
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        logger.debug(f"[Qdrant] Upserted {len(points)} vectors.")

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 10,
        filter_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        ANN search for the most similar speeches to a query vector.

        Parameters
        ----------
        query_vector:
            Query embedding (1D array of floats).
        top_k:
            Number of results to return.
        filter_params:
            Optional Qdrant filter, e.g. {"party": "CDU", "year": 2015}.

        Returns
        -------
        list[dict]
            List of {doc_id, score, year, party, country_code} dicts.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filter_params:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_params.items()
            ]
            qdrant_filter = Filter(must=conditions)

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"score": r.score, **r.payload}
            for r in results
        ]

    def get_all_vectors_for_country(
        self,
        country_code: str,
        year: int | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Retrieve all embedding vectors and payloads for a country (optionally filtered by year).
        Used by the analysis layer for centroid computation and clustering.

        Returns
        -------
        (vectors, payloads):
            vectors: np.ndarray of shape (N, dim)
            payloads: list of payload dicts
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must_conditions = [
            FieldCondition(key="country_code", match=MatchValue(value=country_code))
        ]
        if year is not None:
            must_conditions.append(
                FieldCondition(key="year", match=MatchValue(value=year))
            )

        scroll_filter = Filter(must=must_conditions)

        all_points = []
        next_page_offset = None

        while True:
            points, next_page_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=next_page_offset,
                with_vectors=True,
                with_payload=True,
            )
            all_points.extend(points)
            if next_page_offset is None:
                break

        if not all_points:
            return np.empty((0, self._vector_size)), []

        vectors = np.array([p.vector for p in all_points], dtype=np.float32)
        payloads = [p.payload for p in all_points]
        return vectors, payloads
