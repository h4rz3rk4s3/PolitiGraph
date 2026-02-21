"""
data_loaders/twitter_loader.py
==============================
STUB – Twitter/X JSON export loader.

This stub implements the BaseLoader interface so that the Twitter source
can be wired into the pipeline by simply completing the TODO sections below.
The `Politician` node already has a `social_media_handles` property, and the
graph schema accommodates a `SocialMediaPost` node — no schema changes needed.

To activate:
    1. Obtain tweets via the Academic Research API or an export tool.
    2. Store as newline-delimited JSON (NDJSON) in data/raw/twitter/<country>/
    3. Implement the TODO sections below.
    4. Register in scripts/ingest.py with:
           loader = TwitterLoader(country_code="DE")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from loguru import logger

from config.models import RawDocument
from data_loaders.base_loader import BaseLoader


class TwitterLoader(BaseLoader):
    """
    Loads politician tweets from a directory of NDJSON files.

    Expected file format (Twitter API v2 export):
    {
        "id": "...",
        "text": "...",
        "author_id": "...",
        "created_at": "2023-01-15T14:32:00.000Z",
        "lang": "de",
        "public_metrics": {...},
        "referenced_tweets": [...]
    }

    Parameters
    ----------
    country_code: str
        ISO 3166-1 alpha-2 country code.
    politician_handle_map: dict[str, dict]
        Maps Twitter author_id → {speaker_id, name, party, role}.
        Build this once from your Neo4j Politician nodes.
    """

    def __init__(
        self,
        country_code: str,
        politician_handle_map: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.country_code = country_code.upper()
        self.politician_handle_map = politician_handle_map or {}

    @property
    def source_name(self) -> str:
        return "twitter"

    def get_metadata_schema(self) -> Dict[str, type]:
        return {
            "platform": str,
            "tweet_id": str,
            "retweet_count": int,
            "like_count": int,
            "reply_count": int,
        }

    def load(self, source_path: str | Path) -> Iterator[RawDocument]:
        """
        TODO: Implement NDJSON loading.

        Steps:
            1. Iterate .ndjson files in source_path
            2. Parse each line as JSON
            3. Look up author_id in self.politician_handle_map
            4. Yield a RawDocument with source="twitter"
        """
        root_dir = self.validate_source(source_path)
        ndjson_files = list(root_dir.rglob("*.ndjson"))
        logger.info(
            f"[TwitterLoader] STUB – found {len(ndjson_files)} files. "
            "Implement load() to activate Twitter ingestion."
        )

        # ── STUB: no-op ──────────────────────────────────────────────────────
        # Replace this with actual implementation:
        #
        # import json
        # for ndjson_file in ndjson_files:
        #     with open(ndjson_file) as f:
        #         for line in f:
        #             tweet = json.loads(line)
        #             author = self.politician_handle_map.get(tweet["author_id"], {})
        #             if not author:
        #                 continue
        #             yield RawDocument(
        #                 doc_id=f"tw_{tweet['id']}",
        #                 source="twitter",
        #                 country_code=self.country_code,
        #                 language=tweet.get("lang", "unk"),
        #                 text=tweet["text"],
        #                 speaker_id=author["speaker_id"],
        #                 speaker_name=author["name"],
        #                 party=author["party"],
        #                 role=author["role"],
        #                 date=datetime.fromisoformat(tweet["created_at"]).date(),
        #                 session_id="twitter",
        #                 subcorpus=None,
        #                 metadata={
        #                     "platform": "twitter",
        #                     "tweet_id": tweet["id"],
        #                     **tweet.get("public_metrics", {}),
        #                 },
        #             )
        #
        return iter([])  # Remove when implemented
