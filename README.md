# PolitiGraph PoC
## Longitudinal Semantic Analysis of European Parliamentary Discourse

A production-grade pipeline for ingesting ParlaMint XML data, extracting semantic entities and relationships via a hybrid NER/LLM approach, storing them in a Neo4j Knowledge Graph, and enabling longitudinal analysis of political language.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PolitiGraph Pipeline                         │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────────┐  │
│  │ Data Loaders │───▶│  NLP Engine        │───▶│ Graph Connector │  │
│  │              │    │  Track A: GLiNER   │    │  Neo4j + PODIO  │  │
│  │  ParlaMint   │    │  Track B: LLM      │    │  Ontology       │  │
│  │  (TEI XML)   │    │  (Ollama/vLLM)     │    │                 │  │
│  │  [Twitter*]  │    │  Pydantic schemas  │    │  Vector Index   │  │
│  └──────────────┘    └────────────────────┘    └────────-────────┘  │
│                                                          │          │
│  ┌───────────────────────────────────────────────────────▼────────┐ │
│  │                    Analysis Layer                              │ │
│  │   Embeddings (mE5-large / XLM-R) │ UMAP │ ARI │ Silhouette     │ │
│  │   Cosine Drift │ Topic Salience  │ Complexity Metrics          │ │
│  └───────────────────────────────────────────────────────────────-┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your Neo4j credentials and Ollama/vLLM URL
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
docker-compose up -d
# Starts: Neo4j (bolt + HTTP), Ollama (LLM inference), Qdrant (vector DB)
```

### 3. Pull LLM Model (via Ollama)

```bash
docker exec -it politigraph-ollama ollama pull mistral:7b-instruct-q4_K_M
```

### 4. Download ParlaMint Data

```bash
# Download from: https://www.clarin.eu/parlamint
# Place XML files in: data/raw/ParlaMint-DE/
```

### 5. Run the Pipeline

```bash
# Step 1: Ingest & parse XML → Neo4j Speech nodes
python scripts/ingest.py --country DE --data-dir data/raw/ParlaMint-DE

# Step 2: Run hybrid NLP extraction → populate Topics, Entities, Triples
python scripts/extract.py --country DE --batch-size 50

# Step 3: Generate embeddings & run longitudinal analysis
python scripts/analyze.py --country DE --start-year 2010 --end-year 2024
```

---

## Embedding Model Evaluation

The `embedding_evaluator.py` runs a tournament:
1. Generates embeddings with each candidate model
2. Projects to 2D via UMAP
3. Computes Silhouette Score (clustering coherence vs. party labels)
4. Computes Adjusted Rand Index (ARI) between unsupervised clusters and ground-truth party labels
5. **The model with the highest ARI wins** for this corpus

---

## Design Decisions & Deviations from Spec

| Decision | Rationale |
|---|---|
| **Pydantic v2** for all schemas | Replaces ad-hoc dicts; gives runtime validation + serialization for free |
| **LangGraph** for the extraction pipeline | Enables retries, conditional routing (GLiNER confidence gates LLM usage), and full observability as a state machine |
| **Qdrant** as primary vector store | More mature ANN indexing than Neo4j's vector search; Neo4j stays the graph layer |
| **`mistral:7b-instruct-q4_K_M`** as default LLM | Best quality/speed/VRAM tradeoff for structured JSON extraction on consumer hardware |
| **`intfloat/multilingual-e5-large`** as default embedder | Outperforms XLM-R on semantic similarity benchmarks across EU languages |
| **LangSmith tracing** (optional) | Injected via env var; zero overhead when disabled |

---

## Extending to New Data Sources

Implement `BaseLoader` and register it:

```python
class MyNewLoader(BaseLoader):
    def load(self, source_path: str) -> Iterator[RawDocument]:
        ...
    def get_metadata_schema(self) -> Dict[str, type]:
        ...
```
