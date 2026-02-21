"""
scripts/extract.py
===================
CLI entry point for running the hybrid NLP extraction pipeline
on speeches already stored in Neo4j.

Workflow:
  1. Fetch unprocessed Speech nodes from Neo4j (no sentiment_score yet)
  2. Run GLiNER (Track A) + LLM (Track B) + complexity metrics
  3. Write extracted entities, topics, triples, and metrics back to Neo4j

Usage:
    python scripts/extract.py --country DE
    python scripts/extract.py --country DE --batch-size 50 --max-batches 10
    python scripts/extract.py --country DE --no-llm   # GLiNER + complexity only (faster)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import RawDocument
from config.settings import get_settings
from graph_connectors.neo4j_connector import Neo4jConnector
from nlp_engines.gliner_extractor import GLiNERExtractor
from nlp_engines.llm_extractor import LLMExtractor
from nlp_engines.pipeline import ExtractionPipeline
from datetime import date as date_type

app = typer.Typer(
    name="extract",
    help="Run NLP extraction on ParlaMint speeches stored in Neo4j.",
)


@app.command()
def run(
    country: str = typer.Option(..., "--country", "-c"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Speeches per batch"),
    max_batches: int = typer.Option(-1, "--max-batches", help="Stop after N batches (-1 = unlimited)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM Track B (faster, no topics/triples)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Run hybrid NLP extraction on all unprocessed speeches for a country.
    Writes results (topics, entities, triples, sentiment, complexity) back to Neo4j.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  PolitiGraph NLP Extraction: {country.upper()}")
    typer.echo(f"  Batch size: {batch_size} | LLM enabled: {not no_llm}")
    typer.echo(f"{'='*60}\n")

    connector = Neo4jConnector()
    pipeline = ExtractionPipeline(
        gliner_extractor=GLiNERExtractor(),
        llm_extractor=LLMExtractor() if not no_llm else None,
    )

    total_processed = 0
    total_errors = 0
    batch_count = 0
    start_time = datetime.now()

    try:
        while True:
            if max_batches > 0 and batch_count >= max_batches:
                break

            # Fetch next batch of unprocessed speeches
            raw_batch = connector.get_unprocessed_speeches(
                country_code=country.upper(), batch_size=batch_size
            )
            if not raw_batch:
                typer.echo("No more unprocessed speeches found.")
                break

            # Convert Neo4j records → minimal RawDocuments for the pipeline
            docs = [
                RawDocument(
                    doc_id=r["speech_id"],
                    source="parlamint",
                    country_code=country.upper(),
                    language=r.get("language", "unk"),
                    text=r["raw_text"],
                    # These fields aren't needed for extraction, just pipeline schema compliance
                    speaker_id="",
                    speaker_name="",
                    party="",
                    role="",
                    date=date_type.today(),
                    session_id="",
                )
                for r in raw_batch
                if r.get("raw_text")
            ]

            if not docs:
                break

            # Run extraction pipeline
            results = pipeline.run_batch(docs, gliner_batch_size=16)

            # Write results back to Neo4j
            for result in results:
                try:
                    connector.upsert_extraction(result.doc_id, result)
                    total_processed += 1
                except Exception as e:
                    logger.warning(f"Failed to store extraction for {result.doc_id}: {e}")
                    total_errors += 1

            batch_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_processed / max(elapsed, 1)
            logger.info(
                f"Batch {batch_count}: {len(docs)} processed | "
                f"Total: {total_processed} | Rate: {rate:.1f}/s"
            )

    except KeyboardInterrupt:
        typer.echo("\n\nInterrupted by user.")
    finally:
        connector.close()
        elapsed = (datetime.now() - start_time).total_seconds()
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  Extraction Complete")
        typer.echo(f"  Speeches processed: {total_processed:,}")
        typer.echo(f"  Errors:             {total_errors:,}")
        typer.echo(f"  Elapsed:            {elapsed:.1f}s")
        typer.echo(f"{'='*60}\n")


if __name__ == "__main__":
    app()
