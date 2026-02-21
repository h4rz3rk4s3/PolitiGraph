"""
analyzers/embedding_evaluator.py
==================================
Embedding model evaluation and tournament.

Evaluation methodology (scientifically neutral):
  1. Generate embeddings for a held-out sample of speeches
  2. Project to 2D via UMAP for qualitative inspection
  3. Cluster with KMeans (k = number of parties)
  4. Compute Silhouette Score vs. unsupervised cluster assignments
  5. Compute Adjusted Rand Index (ARI) between:
       - Unsupervised KMeans clusters
       - Ground-truth party labels from ParlaMint metadata
  6. The model with the highest ARI wins for this corpus

Why ARI over Silhouette:
  Silhouette measures internal cluster coherence.
  ARI measures agreement with an external ground truth.
  For political corpora where party affiliation IS the ground truth,
  ARI is the correct metric.  A model that cleanly separates CDU from
  SPD speeches has learned ideologically meaningful representations.

Scientific neutrality note:
  Party labels are treated as abstract categorical identifiers.
  No ideological ordering or scoring is applied.
  The evaluation makes no claim about which party is "correct" or "extreme".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

from embeddings.embedding_module import EmbeddingModule
from embeddings.vector_store import VectorStore


@dataclass
class EvaluationResult:
    """Results from evaluating a single embedding model."""

    model_name: str
    adjusted_rand_index: float
    silhouette_score: float
    n_samples: int
    n_parties: int
    umap_2d: np.ndarray | None = field(default=None, repr=False)
    party_labels: list[str] | None = field(default=None, repr=False)
    cluster_labels: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> str:
        return (
            f"Model: {self.model_name}\n"
            f"  ARI:              {self.adjusted_rand_index:.4f}\n"
            f"  Silhouette Score: {self.silhouette_score:.4f}\n"
            f"  Samples: {self.n_samples} | Parties: {self.n_parties}"
        )


class EmbeddingEvaluator:
    """
    Evaluates and compares multiple embedding models on a political speech corpus.

    The evaluation uses party affiliation from ParlaMint metadata as the
    ground-truth label — the only label known to be correct without any
    human annotation.

    Parameters
    ----------
    vector_store:
        Qdrant store for fetching existing embeddings.
    max_sample_size:
        Cap total speeches to avoid OOM during UMAP (10k+ is usually fine).
    random_seed:
        For reproducibility of KMeans and UMAP.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        max_sample_size: int = 5000,
        random_seed: int = 42,
    ) -> None:
        self.vector_store = vector_store
        self.max_sample_size = max_sample_size
        self.random_seed = random_seed

    def evaluate_single_model(
        self,
        country_code: str,
        model_name: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate the quality of embeddings currently stored in Qdrant
        for a given country.

        Parameters
        ----------
        country_code:
            Restrict evaluation to speeches from this country.
        model_name:
            For labelling purposes only. If None, read from payload.

        Returns
        -------
        EvaluationResult
        """
        logger.info(f"[Evaluator] Fetching embeddings for {country_code}...")
        vectors, payloads = self.vector_store.get_all_vectors_for_country(country_code)

        if vectors.shape[0] == 0:
            raise ValueError(f"No embeddings found for country: {country_code}")

        # Subsample for tractability
        n = min(len(payloads), self.max_sample_size)
        if n < len(payloads):
            idx = np.random.RandomState(self.random_seed).choice(len(payloads), n, replace=False)
            vectors = vectors[idx]
            payloads = [payloads[i] for i in idx]

        # Encode party labels
        parties = [p.get("party", "Unknown") for p in payloads]
        le = LabelEncoder()
        true_labels = le.fit_transform(parties)
        n_parties = len(le.classes_)

        inferred_model = model_name or payloads[0].get("model_name", "unknown")

        # K-Means clustering (unsupervised; k = number of parties)
        logger.info(f"[Evaluator] Running KMeans (k={n_parties}) on {n} vectors...")
        kmeans = KMeans(
            n_clusters=n_parties,
            random_state=self.random_seed,
            n_init="auto",
            max_iter=300,
        )
        cluster_labels = kmeans.fit_predict(vectors)

        # Metrics
        ari = adjusted_rand_score(true_labels, cluster_labels)
        sil = silhouette_score(vectors, cluster_labels, metric="cosine", sample_size=min(n, 2000))

        # UMAP projection
        umap_2d = self._project_umap(vectors)

        result = EvaluationResult(
            model_name=inferred_model,
            adjusted_rand_index=round(ari, 4),
            silhouette_score=round(sil, 4),
            n_samples=n,
            n_parties=n_parties,
            umap_2d=umap_2d,
            party_labels=parties,
            cluster_labels=cluster_labels,
        )
        logger.info(f"\n{result.summary()}")
        return result

    def tournament(
        self,
        candidate_model_names: list[str],
        sample_docs: list[Any],  # list[RawDocument]
        country_code: str,
    ) -> tuple[str, list[EvaluationResult]]:
        """
        Run a full evaluation tournament across multiple embedding models.

        Parameters
        ----------
        candidate_model_names:
            List of HuggingFace model IDs to compare.
        sample_docs:
            RawDocument objects to embed for each candidate.
        country_code:
            Used for result labelling.

        Returns
        -------
        (winner_model_name, all_results)
        """
        results: list[EvaluationResult] = []

        for model_name in candidate_model_names:
            logger.info(f"[Tournament] Evaluating model: {model_name}")
            try:
                embedder = EmbeddingModule(model_name=model_name)
                records = embedder.embed_documents(sample_docs)

                # Write to a temp collection and evaluate
                from embeddings.vector_store import VectorStore

                temp_collection = f"eval_{model_name.split('/')[-1].lower().replace('-', '_')}"
                temp_store = VectorStore(
                    collection_name=temp_collection,
                    vector_size=embedder.embedding_dim,
                    recreate_collection=True,
                )
                temp_store.upsert_batch(records)

                result = self.evaluate_single_model(
                    country_code=country_code,
                    model_name=model_name,
                )
                result.model_name = model_name
                results.append(result)

            except Exception as e:
                logger.error(f"[Tournament] Failed for {model_name}: {e}")

        if not results:
            raise RuntimeError("All candidate models failed evaluation.")

        # Winner = highest ARI
        winner = max(results, key=lambda r: r.adjusted_rand_index)
        logger.info(
            f"\n{'='*60}\n"
            f"TOURNAMENT WINNER: {winner.model_name}\n"
            f"  ARI: {winner.adjusted_rand_index:.4f}\n"
            f"{'='*60}"
        )
        return winner.model_name, results

    def _project_umap(self, vectors: np.ndarray) -> np.ndarray | None:
        """Project embeddings to 2D using UMAP."""
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                random_state=self.random_seed,
                n_neighbors=15,
                min_dist=0.1,
                verbose=False,
            )
            return reducer.fit_transform(vectors)
        except ImportError:
            logger.warning("[Evaluator] umap-learn not installed. Skipping UMAP projection.")
            return None

    def save_umap_plot(
        self,
        result: EvaluationResult,
        output_path: str = "umap_party_embeddings.html",
    ) -> None:
        """
        Save an interactive UMAP scatter plot coloured by party affiliation.
        Requires plotly.
        """
        if result.umap_2d is None or result.party_labels is None:
            logger.warning("[Evaluator] No UMAP data to plot.")
            return
        try:
            import plotly.express as px

            df = pd.DataFrame(
                {
                    "x": result.umap_2d[:, 0],
                    "y": result.umap_2d[:, 1],
                    "party": result.party_labels,
                }
            )
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="party",
                title=f"UMAP Party Embedding Space — {result.model_name}\n"
                      f"ARI: {result.adjusted_rand_index:.3f} | "
                      f"Silhouette: {result.silhouette_score:.3f}",
                opacity=0.6,
                width=1200,
                height=800,
            )
            fig.write_html(output_path)
            logger.info(f"[Evaluator] Saved UMAP plot: {output_path}")
        except ImportError:
            logger.warning("[Evaluator] plotly not installed. Cannot save UMAP plot.")
