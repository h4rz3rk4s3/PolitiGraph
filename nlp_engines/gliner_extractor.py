"""
nlp_engines/gliner_extractor.py
================================
Track A: High-Precision Named Entity Recognition using GLiNER.

GLiNER (Generalist Model for NER) uses a BERT-like encoder with a
bipartite matching head. It supports arbitrary entity types defined
at inference time — no fine-tuning required. This makes it ideal for
multilingual political corpora where entity types vary.

Model: urchade/gliner_mediumv2.1
  - ~90M params, multilingual via mDeBERTa base
  - Supports 100+ languages including all EU official languages
  - Runs comfortably on CPU for batch sizes ≤ 32

Reference: https://github.com/urchade/GLiNER
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

from config.models import ExtractionResult, NamedEntity, RawDocument
from config.settings import get_settings
from nlp_engines.base_extractor import BaseExtractor

if TYPE_CHECKING:
    from gliner import GLiNER as GLiNERModel


# Entity labels we care about for political discourse analysis
_ENTITY_LABELS: list[str] = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "LAW_TREATY",
    "POLITICAL_PARTY",
    "EVENT",
]


@lru_cache(maxsize=1)
def _load_gliner_model(model_name: str) -> "GLiNERModel":
    """Load and cache the GLiNER model (singleton per process)."""
    from gliner import GLiNER  # lazy import to avoid cold-start overhead

    logger.info(f"[GLiNER] Loading model: {model_name}")
    model = GLiNER.from_pretrained(model_name)
    logger.info("[GLiNER] Model loaded.")
    return model


class GLiNERExtractor(BaseExtractor):
    """
    Extracts rigid named entities from speech text using GLiNER.

    Produces:
        ExtractionResult.named_entities — list[NamedEntity]

    Parameters
    ----------
    model_name:
        HuggingFace model ID, default from settings.
    confidence_threshold:
        Minimum GLiNER score to keep an entity prediction.
    entity_labels:
        Entity types to extract. Overrides defaults if provided.
    """

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        entity_labels: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.pipeline.gliner_model
        self._threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.pipeline.gliner_confidence_threshold
        )
        self._entity_labels = entity_labels or _ENTITY_LABELS

    @property
    def extractor_name(self) -> str:
        return "gliner"

    def _model(self) -> "GLiNERModel":
        return _load_gliner_model(self._model_name)

    def extract(self, doc: RawDocument) -> ExtractionResult:
        return ExtractionResult(
            doc_id=doc.doc_id,
            named_entities=self._run_ner(doc.text),
        )

    def extract_batch(self, docs: list[RawDocument]) -> list[ExtractionResult]:
        """
        Use GLiNER's native batch_predict for throughput improvement
        vs looping over single-document predict calls.
        """
        if not docs:
            return []

        texts = [doc.text for doc in docs]
        model = self._model()

        try:
            # GLiNER batch predict: returns list[list[dict]]
            batch_predictions = model.batch_predict_entities(
                texts=texts,
                labels=self._entity_labels,
                threshold=self._threshold,
            )
        except Exception as e:
            logger.error(f"[GLiNER] Batch prediction failed: {e}. Falling back to sequential.")
            return [self.extract(doc) for doc in docs]

        results = []
        for doc, preds in zip(docs, batch_predictions):
            entities = [
                NamedEntity(
                    text=p["text"],
                    label=p["label"].upper(),
                    start=p["start"],
                    end=p["end"],
                    score=round(float(p["score"]), 4),
                )
                for p in preds
                if float(p.get("score", 0)) >= self._threshold
            ]
            results.append(ExtractionResult(doc_id=doc.doc_id, named_entities=entities))

        return results
