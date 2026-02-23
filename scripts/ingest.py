"""
scripts/ingest.py
==================
CLI entry point for ingesting ParlaMint XML data into Neo4j.

Usage:
    python scripts/ingest.py --country DE --data-dir parlamint_data/ParlaMint-DE
    python scripts/ingest.py --country FR --data-dir parlamint_data/ParlaMint-FR --batch-size 200
    python scripts/ingest.py --country DE --data-dir parlamint_data/ParlaMint-AT --dry-run
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import SpeechNode
from config.settings import get_settings
from data_loaders.parlamint_loader import ParlaMintLoader
from graph_connectors.neo4j_connector import Neo4jConnector

app = typer.Typer(
    name="ingest",
    help="Ingest ParlaMint TEI XML data into the PolitiGraph Neo4j knowledge graph.",
)


@app.command()
def run(
    country: str = typer.Option(..., "--country", "-c", help="ISO 3166-1 alpha-2 country code (e.g. DE, FR)"),
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Path to ParlaMint XML directory"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Documents per Neo4j write batch"),
    min_tokens: int = typer.Option(20, "--min-tokens", help="Minimum token count per utterance"),
    subcorpus: list[str] = typer.Option(None, "--subcorpus", help="Filter to specific subcorpora (e.g. COVID)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and count without writing to Neo4j"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Parse ParlaMint XML and load speeches into Neo4j.

    For Germany, expects files in:  data/raw/ParlaMint-DE/*.xml
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  PolitiGraph Ingestion: {country.upper()}")
    typer.echo(f"  Source:    {data_dir}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Dry run:   {dry_run}")
    typer.echo(f"{'='*60}\n")

    # Initialise loader
    loader = ParlaMintLoader(
        country_code=country,
        subcorpus_filter=list(subcorpus) if subcorpus else None,
        min_token_count=min_tokens,
    )

    if dry_run:
        # Dry run: just count documents
        total = sum(1 for _ in loader.load(data_dir))
        typer.echo(f"[DRY RUN] Would ingest {total} utterances from {data_dir}")
        return

    # Connect to Neo4j
    connector = Neo4jConnector()
    total_ingested = 0
    total_errors = 0
    start_time = datetime.now()

    try:
        for batch in tqdm(loader.batch(data_dir, batch_size=batch_size), desc="Ingesting batches"):
            batch_errors = 0
            for doc in batch:
                try:
                    # Build SpeechNode from RawDocument
                    speech_node = SpeechNode(
                        speech_id=doc.doc_id,
                        session_id=doc.session_id,
                        date=doc.date,
                        timestamp=datetime(doc.date.year, doc.date.month, doc.date.day),
                        language=doc.language,
                        subcorpus=doc.subcorpus,
                        word_count=doc.metadata.get("token_count", len(doc.text.split())),
                        raw_text=doc.text,
                    )
                    connector.upsert_speech(doc, speech_node)
                    total_ingested += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest {doc.doc_id}: {e}")
                    batch_errors += 1
                    total_errors += 1

            if batch_errors > 0:
                logger.warning(f"Batch had {batch_errors} errors.")

    except KeyboardInterrupt:
        typer.echo("\n\nInterrupted by user.")
    finally:
        connector.close()
        elapsed = (datetime.now() - start_time).total_seconds()
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  Ingestion Complete")
        typer.echo(f"  Total ingested: {total_ingested:,}")
        typer.echo(f"  Total errors:   {total_errors:,}")
        typer.echo(f"  Elapsed:        {elapsed:.1f}s")
        typer.echo(f"  Rate:           {total_ingested / max(elapsed, 1):.0f} speeches/sec")
        typer.echo(f"{'='*60}\n")


if __name__ == "__main__":
    app()
