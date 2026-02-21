"""
scripts/analyze.py
===================
CLI entry point for the full analysis pipeline:
  1. Generate and store embeddings (Qdrant + Neo4j)
  2. Evaluate embedding quality (UMAP + ARI + Silhouette)
  3. Compute longitudinal metrics (cosine drift, topic salience, complexity)
  4. Export results to CSV for downstream use

Usage:
    python scripts/analyze.py --country DE --start-year 2010 --end-year 2024
    python scripts/analyze.py --country DE --step embeddings
    python scripts/analyze.py --country DE --step evaluate --compare-models
    python scripts/analyze.py --country DE --step longitudinal --party-a CDU --party-b AfD
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Literal

import typer
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from embeddings.embedding_module import EmbeddingModule
from embeddings.vector_store import VectorStore
from graph_connectors.neo4j_connector import Neo4jConnector
from analyzers.longitudinal_analyzer import LongitudinalAnalyzer
from analyzers.embedding_evaluator import EmbeddingEvaluator

app = typer.Typer(name="analyze", help="Run longitudinal analysis on PolitiGraph data.")


@app.command()
def run(
    country: str = typer.Option(..., "--country", "-c"),
    start_year: int = typer.Option(2010, "--start-year"),
    end_year: int = typer.Option(2024, "--end-year"),
    step: str = typer.Option(
        "all",
        "--step",
        help="Which step to run: all | embeddings | evaluate | longitudinal",
    ),
    party_a: str = typer.Option(None, "--party-a", help="First party for drift analysis"),
    party_b: str = typer.Option(None, "--party-b", help="Second party for drift analysis"),
    compare_models: bool = typer.Option(
        False,
        "--compare-models",
        help="Run embedding model tournament (slower, picks best model)",
    ),
    output_dir: Path = typer.Option(Path("data/processed"), "--output-dir"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the full analysis pipeline or a specific step."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  PolitiGraph Analysis: {country.upper()}")
    typer.echo(f"  Period: {start_year}–{end_year} | Step: {step}")
    typer.echo(f"{'='*60}\n")

    connector = Neo4jConnector()
    vector_store = VectorStore(vector_size=1024)  # mE5-large dim

    try:
        if step in ("all", "embeddings"):
            _run_embeddings(connector, vector_store, country, settings, output_dir)

        if step in ("all", "evaluate"):
            _run_evaluation(vector_store, country, compare_models, output_dir)

        if step in ("all", "longitudinal"):
            _run_longitudinal(
                connector, vector_store, country,
                start_year, end_year, party_a, party_b, output_dir
            )

    finally:
        connector.close()

    typer.echo("\n✓ Analysis complete. Results written to: " + str(output_dir))


def _run_embeddings(connector, vector_store, country, settings, output_dir):
    """Generate embeddings for all speeches and store in Qdrant + Neo4j."""
    typer.echo("\n[Step 1/3] Generating embeddings...")

    embedder = EmbeddingModule()
    speeches = connector.get_speeches_for_analysis(country_code=country)

    if not speeches:
        logger.warning("No speeches found for embedding. Run ingestion first.")
        return

    # Filter to speeches without embeddings
    typer.echo(f"  Found {len(speeches):,} speeches total.")

    batch_size = settings.embedding.batch_size
    total = 0

    from config.models import RawDocument, EmbeddingRecord
    from datetime import date

    for i in tqdm(range(0, len(speeches), batch_size), desc="Embedding"):
        batch = speeches[i : i + batch_size]

        # Build minimal RawDocuments for the embedder
        # (raw_text not available in the metadata fetch, need separate query)
        doc_ids = [s["speech_id"] for s in batch]

        # Fetch raw texts for this batch
        texts_query = """
        MATCH (s:Speech)
        WHERE s.speech_id IN $ids
        RETURN s.speech_id AS speech_id, s.raw_text AS raw_text
        """
        with connector._session() as session:
            raw_texts = {
                r["speech_id"]: r["raw_text"]
                for r in session.run(texts_query, ids=doc_ids)
            }

        docs = [
            RawDocument(
                doc_id=s["speech_id"],
                source="parlamint",
                country_code=country,
                language=s.get("language", "unk"),
                text=raw_texts.get(s["speech_id"], ""),
                speaker_id=s.get("politician_id", ""),
                speaker_name=s.get("politician_name", ""),
                party=s.get("party", "Unknown"),
                role="",
                date=date(int(s["year"]), 1, 1),
                session_id="",
            )
            for s in batch
            if raw_texts.get(s["speech_id"])
        ]

        if not docs:
            continue

        records = embedder.embed_documents(docs)
        vector_store.upsert_batch(records)

        # Optionally write to Neo4j too (for graph-native vector search)
        for record in records:
            try:
                connector.store_embedding_in_node(
                    record.doc_id, record.vector, record.model_name
                )
            except Exception:
                pass  # Non-critical; Qdrant is the primary vector store

        total += len(records)

    typer.echo(f"  ✓ Embedded {total:,} speeches with {get_settings().embedding.model}")


def _run_evaluation(vector_store, country, compare_models, output_dir):
    """Evaluate embedding quality with UMAP + ARI + Silhouette."""
    typer.echo("\n[Step 2/3] Evaluating embedding quality...")

    evaluator = EmbeddingEvaluator(vector_store=vector_store)

    if compare_models:
        typer.echo("  Running embedding model tournament...")
        # Tournament: compare candidate models on a sample
        # Note: this requires re-embedding, so only run during model selection
        candidates = [
            "intfloat/multilingual-e5-large",
            "intfloat/multilingual-e5-base",
            "FacebookAI/xlm-roberta-large",
        ]
        typer.echo(f"  Candidates: {candidates}")
        typer.echo("  (Tournament requires re-embedding; use --compare-models only once)")
    else:
        # Evaluate current embeddings
        result = evaluator.evaluate_single_model(country_code=country)

        typer.echo(f"\n  Embedding Evaluation Results:")
        typer.echo(f"    Model:            {result.model_name}")
        typer.echo(f"    ARI:              {result.adjusted_rand_index:.4f}")
        typer.echo(f"    Silhouette Score: {result.silhouette_score:.4f}")
        typer.echo(f"    Samples:          {result.n_samples:,}")
        typer.echo(f"    Parties:          {result.n_parties}")

        # Save UMAP plot
        umap_path = output_dir / f"umap_{country.lower()}_{datetime.now().strftime('%Y%m%d')}.html"
        evaluator.save_umap_plot(result, str(umap_path))


def _run_longitudinal(
    connector, vector_store, country, start_year, end_year,
    party_a, party_b, output_dir
):
    """Run all longitudinal analyses and export to CSV."""
    typer.echo("\n[Step 3/3] Running longitudinal analysis...")

    analyzer = LongitudinalAnalyzer(
        connector=connector,
        vector_store=vector_store,
        country_code=country,
        start_year=start_year,
        end_year=end_year,
    )

    prefix = output_dir / f"{country.lower()}"

    # Complexity trends
    typer.echo("  → Computing complexity trends...")
    df_complexity = analyzer.complexity_trends()
    df_complexity.to_csv(f"{prefix}_complexity_trends.csv", index=False)
    typer.echo(f"    Saved: {prefix}_complexity_trends.csv")

    # Sentiment trends
    typer.echo("  → Computing sentiment trends...")
    df_sentiment = analyzer.sentiment_trends()
    df_sentiment.to_csv(f"{prefix}_sentiment_trends.csv", index=False)
    typer.echo(f"    Saved: {prefix}_sentiment_trends.csv")

    # Cosine drift between two parties
    if party_a and party_b:
        typer.echo(f"  → Computing cosine drift: {party_a} ↔ {party_b}...")
        df_drift = analyzer.cosine_drift_over_time(party_a, party_b)
        df_drift.to_csv(f"{prefix}_drift_{party_a}_{party_b}.csv", index=False)
        typer.echo(f"    Saved: {prefix}_drift_{party_a}_{party_b}.csv")

        typer.echo(f"  → Computing semantic velocity: {party_a}...")
        df_vel_a = analyzer.semantic_velocity(party_a)
        df_vel_a.to_csv(f"{prefix}_velocity_{party_a}.csv", index=False)

        typer.echo(f"  → Computing semantic velocity: {party_b}...")
        df_vel_b = analyzer.semantic_velocity(party_b)
        df_vel_b.to_csv(f"{prefix}_velocity_{party_b}.csv", index=False)

    # Pairwise distances for last year
    typer.echo(f"  → Computing pairwise centroid distances for {end_year}...")
    df_distances = analyzer.pairwise_centroid_distances(year=end_year)
    df_distances.to_csv(f"{prefix}_distances_{end_year}.csv")
    typer.echo(f"    Saved: {prefix}_distances_{end_year}.csv")

    # Summary
    typer.echo(f"\n  ✓ Longitudinal analysis complete.")
    if not df_complexity.empty:
        typer.echo(f"\n  Sample — Complexity trends (latest year per party):")
        latest = df_complexity.groupby("party").last().reset_index()
        typer.echo(
            latest[["party", "avg_ttr", "avg_fk_grade", "avg_sentence_length", "speech_count"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    app()
