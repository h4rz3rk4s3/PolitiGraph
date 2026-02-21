"""
embeddings/embedding_module.py
================================
Multilingual embedding generation for parliamentary speeches.

Default model: intfloat/multilingual-e5-large
  - 560M params, 1024-dim output
  - Trained on 100+ languages with strong cross-lingual alignment
  - Outperforms XLM-R on BEIR benchmarks for semantic similarity
  - mE5-large was specifically fine-tuned on MS-MARCO and NLI tasks,
    giving it strong domain-transfer properties for formal political text

Alternative: FacebookAI/xlm-roberta-large (768 dims)
  - Faster inference, lower memory
  - Slightly lower quality on semantic similarity vs. mE5-large

The EmbeddingEvaluator in analyzers/ runs a tournament (ARI comparison)
to empirically pick the best model for this corpus.

mE5 prompt prefix:
  intfloat/multilingual-e5-large requires a "query: " or "passage: " prefix.
  For document embedding (not retrieval), use "passage: " prefix.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger

from config.models import EmbeddingRecord, RawDocument
from config.settings import get_settings

# mE5 requires a prompt prefix for the embedding direction
_ME5_PASSAGE_PREFIX = "passage: "
_ME5_MODELS = {"intfloat/multilingual-e5-large", "intfloat/multilingual-e5-base"}


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str):
    """Load and cache a SentenceTransformer model (singleton per model name)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"[Embeddings] Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"[Embeddings] Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


class EmbeddingModule:
    """
    Generates sentence embeddings for parliamentary speeches.

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local path.
    device:
        "cpu", "cuda", or "mps".
    batch_size:
        Number of sentences to embed per forward pass.
    normalize:
        Whether to L2-normalize output vectors (recommended for cosine similarity).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        settings = get_settings().embedding
        self.model_name = model_name or settings.model
        self.device = device or settings.device
        self.batch_size = batch_size or settings.batch_size
        self.normalize = normalize
        self._use_me5_prefix = self.model_name in _ME5_MODELS

    def _model(self):
        return _load_model(self.model_name, self.device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns float32 array of shape (N, dim).

        Parameters
        ----------
        texts:
            Raw speech texts. Do NOT pre-tokenize; the model handles that.

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            return np.empty((0, self._model().get_sentence_embedding_dimension()), dtype=np.float32)

        if self._use_me5_prefix:
            texts = [f"{_ME5_PASSAGE_PREFIX}{t}" for t in texts]

        embeddings = self._model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_documents(self, docs: list[RawDocument]) -> list[EmbeddingRecord]:
        """
        Embed a list of RawDocuments and return EmbeddingRecord objects.
        """
        if not docs:
            return []

        texts = [doc.text for doc in docs]
        vectors = self.embed_texts(texts)

        records = []
        for doc, vec in zip(docs, vectors):
            records.append(
                EmbeddingRecord(
                    doc_id=doc.doc_id,
                    vector=vec.tolist(),
                    model_name=self.model_name,
                    party=doc.party,
                    country_code=doc.country_code,
                    year=doc.date.year,
                    language=doc.language,
                )
            )
        return records

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 1D float32 array."""
        return self.embed_texts([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._model().get_sentence_embedding_dimension()
