"""
nlp_engines/pipeline.py
========================
Hybrid NLP Extraction Pipeline using LangGraph as the state machine.

Graph structure:
  load_doc → track_a_ner → gate → track_b_llm → compute_complexity → merge → done
                                     ↑
                              (conditional: only if doc passes quality gate)

The "gate" node decides whether a document is rich enough to warrant LLM
processing (Track B is ~100x slower than GLiNER).  The gate uses a simple
heuristic: word_count >= MIN_WORDS_FOR_LLM.

The pipeline is instrumented for LangSmith tracing when
LANGCHAIN_TRACING_V2=true is set in the environment.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel

from config.models import ExtractionResult, RawDocument
from nlp_engines.base_extractor import BaseExtractor
from nlp_engines.complexity_calculator import ComplexityCalculator
from nlp_engines.gliner_extractor import GLiNERExtractor
from nlp_engines.llm_extractor import LLMExtractor

# Minimum word count to warrant a (slow) LLM Track B extraction
_MIN_WORDS_FOR_LLM = 50


# ── LangGraph state schema ────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """Typed state passed between LangGraph nodes."""

    doc: RawDocument
    result_a: ExtractionResult | None = None   # GLiNER output
    result_b: ExtractionResult | None = None   # LLM output
    result_complexity: dict[str, float] | None = None
    merged: ExtractionResult | None = None
    run_llm: bool = True

    model_config = {"arbitrary_types_allowed": True}


# ── Node functions ────────────────────────────────────────────────────────────

def _node_quality_gate(state: PipelineState) -> PipelineState:
    """Decide whether this document is rich enough for LLM extraction."""
    word_count = len(state.doc.text.split())
    state.run_llm = word_count >= _MIN_WORDS_FOR_LLM
    if not state.run_llm:
        logger.debug(
            f"[Pipeline] Doc {state.doc.doc_id} skipped LLM "
            f"(word_count={word_count} < {_MIN_WORDS_FOR_LLM})"
        )
    return state


def _make_track_a_node(extractor: GLiNERExtractor):
    def _node(state: PipelineState) -> PipelineState:
        state.result_a = extractor.extract(state.doc)
        return state
    return _node


def _make_track_b_node(extractor: LLMExtractor):
    def _node(state: PipelineState) -> PipelineState:
        if state.run_llm:
            state.result_b = extractor.extract(state.doc)
        else:
            state.result_b = ExtractionResult(doc_id=state.doc.doc_id)
        return state
    return _node


def _make_complexity_node(calc: ComplexityCalculator):
    def _node(state: PipelineState) -> PipelineState:
        state.result_complexity = calc.compute(state.doc.text, state.doc.language)
        return state
    return _node


def _node_merge(state: PipelineState) -> PipelineState:
    """Merge Track A + Track B + complexity into a single ExtractionResult."""
    a = state.result_a or ExtractionResult(doc_id=state.doc.doc_id)
    b = state.result_b or ExtractionResult(doc_id=state.doc.doc_id)

    state.merged = ExtractionResult(
        doc_id=state.doc.doc_id,
        named_entities=a.named_entities,
        topics=b.topics,
        triples=b.triples,
        sentiment_score=b.sentiment_score,
        complexity_metrics=state.result_complexity or {},
    )
    return state


# ── LangGraph conditional edge ────────────────────────────────────────────────

def _route_llm(state: PipelineState) -> str:
    return "track_b_llm" if state.run_llm else "skip_llm"


# ── Pipeline class ────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Orchestrates the hybrid GLiNER + LLM extraction pipeline.

    Internally builds a LangGraph StateGraph for clean observability
    and conditional routing.  Falls back to a simple sequential
    implementation if langgraph is not installed.

    Parameters
    ----------
    gliner_extractor:
        Track A extractor. Uses defaults if not provided.
    llm_extractor:
        Track B extractor. Uses defaults if not provided.
    complexity_calculator:
        Complexity metrics module.
    """

    def __init__(
        self,
        gliner_extractor: GLiNERExtractor | None = None,
        llm_extractor: LLMExtractor | None = None,
        complexity_calculator: ComplexityCalculator | None = None,
    ) -> None:
        self._gliner = gliner_extractor or GLiNERExtractor()
        self._llm = llm_extractor or LLMExtractor()
        self._complexity = complexity_calculator or ComplexityCalculator()
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph StateGraph."""
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(PipelineState)

            # Add nodes
            graph.add_node("quality_gate", _node_quality_gate)
            graph.add_node("track_a_ner", _make_track_a_node(self._gliner))
            graph.add_node("track_b_llm", _make_track_b_node(self._llm))
            graph.add_node("skip_llm", lambda s: s)  # no-op passthrough
            graph.add_node("compute_complexity", _make_complexity_node(self._complexity))
            graph.add_node("merge", _node_merge)

            # Edges
            graph.set_entry_point("quality_gate")
            graph.add_edge("quality_gate", "track_a_ner")
            graph.add_conditional_edges(
                "track_a_ner",
                _route_llm,
                {"track_b_llm": "track_b_llm", "skip_llm": "skip_llm"},
            )
            graph.add_edge("track_b_llm", "compute_complexity")
            graph.add_edge("skip_llm", "compute_complexity")
            graph.add_edge("compute_complexity", "merge")
            graph.add_edge("merge", END)

            compiled = graph.compile()
            logger.info("[Pipeline] LangGraph state machine compiled successfully.")
            return compiled

        except ImportError:
            logger.warning(
                "[Pipeline] langgraph not available. "
                "Falling back to sequential execution."
            )
            return None

    def run(self, doc: RawDocument) -> ExtractionResult:
        """Process a single document through the full pipeline."""
        if self._graph is not None:
            initial_state = PipelineState(doc=doc)
            final_state = self._graph.invoke(initial_state)
            # LangGraph returns a dict-like; coerce back
            if isinstance(final_state, dict):
                final_state = PipelineState(**final_state)
            return final_state.merged or ExtractionResult(doc_id=doc.doc_id)
        else:
            return self._run_sequential(doc)

    def _run_sequential(self, doc: RawDocument) -> ExtractionResult:
        """Fallback sequential implementation (no LangGraph)."""
        state = PipelineState(doc=doc)
        state = _node_quality_gate(state)
        state = _make_track_a_node(self._gliner)(state)
        state = _make_track_b_node(self._llm)(state)
        state = _make_complexity_node(self._complexity)(state)
        state = _node_merge(state)
        return state.merged or ExtractionResult(doc_id=doc.doc_id)

    def run_batch(
        self, docs: list[RawDocument], gliner_batch_size: int = 16
    ) -> list[ExtractionResult]:
        """
        Process a batch of documents.

        GLiNER track uses native batch inference for efficiency.
        LLM track is still sequential (one at a time).

        Parameters
        ----------
        docs:
            List of RawDocument objects.
        gliner_batch_size:
            Batch size for GLiNER's batch_predict (tune to GPU/CPU memory).
        """
        if not docs:
            return []

        logger.info(f"[Pipeline] Processing batch of {len(docs)} documents.")

        # Track A: native GLiNER batch
        gliner_results: dict[str, ExtractionResult] = {}
        for i in range(0, len(docs), gliner_batch_size):
            batch = docs[i : i + gliner_batch_size]
            for doc, result in zip(batch, self._gliner.extract_batch(batch)):
                gliner_results[doc.doc_id] = result

        # Track B + complexity: sequential (LLM is the bottleneck)
        final_results = []
        for doc in docs:
            state = PipelineState(doc=doc)
            state.result_a = gliner_results.get(doc.doc_id)
            state = _node_quality_gate(state)
            state = _make_track_b_node(self._llm)(state)
            state = _make_complexity_node(self._complexity)(state)
            state = _node_merge(state)
            final_results.append(state.merged or ExtractionResult(doc_id=doc.doc_id))

        logger.info(f"[Pipeline] Batch complete. {len(final_results)} results produced.")
        return final_results
