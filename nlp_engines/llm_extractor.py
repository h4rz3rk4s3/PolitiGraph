"""
nlp_engines/llm_extractor.py
==============================
Track B: Abstract Topic & Relationship Extraction via a quantized local LLM.

Uses the Ollama Python client to call a locally-running Mistral-7B-Instruct
(q4_K_M quantization).  The LLM is prompted to return structured JSON, which
is validated with Pydantic before being stored.

Design: 
  - Prompt engineering uses few-shot examples tuned for political speech.
  - JSON output is validated with Pydantic v2; malformed responses trigger
    a structured retry with an explicit correction prompt.
  - Sentiment is extracted in the same pass to avoid a second LLM call.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from config.models import (
    AbstractTopic,
    ExtractionResult,
    RawDocument,
    Triple,
)
from config.settings import get_settings
from nlp_engines.base_extractor import BaseExtractor


# ── Pydantic schema for LLM JSON output ──────────────────────────────────────

class _LLMOutput(BaseModel):
    """Validated schema for the LLM's JSON response."""

    topics: list[dict[str, Any]] = Field(default_factory=list)
    triples: list[dict[str, Any]] = Field(default_factory=list)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a political discourse analyst and knowledge extraction engine.
Your task is to analyse a parliamentary speech and extract structured information.
You MUST respond with valid JSON only — no prose, no markdown fences, no explanation.

Output schema (strict JSON):
{
  "topics": [
    {"label": "Short topic label (3-5 words)", "keywords": ["keyword1", "keyword2"]}
  ],
  "triples": [
    {"subject": "Subject entity", "relation": "RELATION_TYPE", "obj": "Object entity",
     "subject_type": "PERSON|ORG|CONCEPT|LOCATION", "obj_type": "PERSON|ORG|CONCEPT|LOCATION|POLICY"}
  ],
  "sentiment_score": 0.0
}

Rules:
- Extract 2-5 abstract topics that the speech is primarily about (e.g. "Climate Policy", "Pension Reform", "Border Security").
- Extract 3-10 semantic triples that capture key claims, positions, or relationships.
- Relation types should be uppercase snake_case (e.g. SUPPORTS, OPPOSES, PROPOSES, BLAMES, CALLS_FOR, CRITICISES).
- sentiment_score: float in [-1.0, 1.0] where -1 = very negative/hostile, 0 = neutral, 1 = very positive/constructive.
- Be language-agnostic: output English labels regardless of input language.
- Normalise entity names (e.g. "The Chancellor" → actual name if determinable from context).
"""

_FEW_SHOT_EXAMPLES = """
Example input:
"Die Bundesregierung muss endlich handeln. Die Energiepreise explodieren und die Bürger können sich das nicht mehr leisten. Wir fordern sofortige Steuersenkungen auf Energie und ein Ende der Abhängigkeit von russischem Gas."

Example output:
{
  "topics": [
    {"label": "Energy Price Crisis", "keywords": ["energy prices", "tax cuts", "affordability"]},
    {"label": "Energy Independence", "keywords": ["Russian gas", "dependency", "energy security"]}
  ],
  "triples": [
    {"subject": "Federal Government", "relation": "FAILS_TO", "obj": "Address Energy Crisis", "subject_type": "ORG", "obj_type": "CONCEPT"},
    {"subject": "Speaker", "relation": "DEMANDS", "obj": "Energy Tax Cuts", "subject_type": "PERSON", "obj_type": "POLICY"},
    {"subject": "Germany", "relation": "DEPENDS_ON", "obj": "Russian Gas", "subject_type": "LOCATION", "obj_type": "CONCEPT"}
  ],
  "sentiment_score": -0.6
}
"""

_USER_TEMPLATE = """\
Analyse the following parliamentary speech excerpt and extract topics, triples, and sentiment.

Speech text:
\"\"\"
{text}
\"\"\"

{examples}

Respond with valid JSON only:"""


class LLMExtractor(BaseExtractor):
    """
    Extracts abstract topics, semantic triples, and sentiment from speech text
    using a locally-hosted quantized LLM (via Ollama).

    Produces:
        ExtractionResult.topics       — list[AbstractTopic]
        ExtractionResult.triples      — list[Triple]
        ExtractionResult.sentiment_score — float | None

    Parameters
    ----------
    max_text_chars:
        Truncate speech text to this length before sending to LLM.
        Prevents context window overflows (Mistral-7B: 32k token context).
    use_few_shot:
        Whether to include few-shot examples in the prompt.
    max_retries:
        Number of times to retry on JSON parse / validation failure.
    """

    def __init__(
        self,
        max_text_chars: int = 3000,
        use_few_shot: bool = True,
        max_retries: int = 2,
    ) -> None:
        settings = get_settings()
        self._settings = settings.llm
        self._max_text_chars = max_text_chars
        self._use_few_shot = use_few_shot
        self._max_retries = max_retries
        self._client = None  # lazy init

    @property
    def extractor_name(self) -> str:
        return "llm_mistral"

    def _get_client(self):
        """Lazily initialise the Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self._settings.base_url)
        return self._client

    def extract(self, doc: RawDocument) -> ExtractionResult:
        text = doc.text[: self._max_text_chars]
        examples = _FEW_SHOT_EXAMPLES if self._use_few_shot else ""
        prompt = _USER_TEMPLATE.format(text=text, examples=examples)

        raw_json: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw_json = self._call_ollama(prompt)
                parsed = self._parse_response(raw_json)
                return self._to_extraction_result(doc.doc_id, parsed)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < self._max_retries:
                    logger.debug(
                        f"[LLMExtractor] Retry {attempt + 1}/{self._max_retries} "
                        f"for doc {doc.doc_id}: {e}"
                    )
                    # Correction prompt
                    prompt = self._correction_prompt(raw_json, str(e))
                else:
                    logger.warning(
                        f"[LLMExtractor] Failed to parse LLM output for {doc.doc_id} "
                        f"after {self._max_retries} retries. Returning empty extraction."
                    )
            except Exception as e:
                logger.error(f"[LLMExtractor] LLM call failed for {doc.doc_id}: {e}")
                break

        return ExtractionResult(doc_id=doc.doc_id)

    def _call_ollama(self, prompt: str) -> str:
        """Make a single synchronous call to the Ollama API."""
        client = self._get_client()
        response = client.chat(
            model=self._settings.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self._settings.temperature,
                "num_predict": self._settings.max_tokens,
            },
        )
        return response["message"]["content"]

    def _parse_response(self, raw: str) -> _LLMOutput:
        """Extract and validate JSON from the LLM response string."""
        # Strip any markdown fences the model may still emit
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        # Find the outermost JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", raw, 0)

        data = json.loads(match.group(0))
        return _LLMOutput(**data)

    def _to_extraction_result(self, doc_id: str, parsed: _LLMOutput) -> ExtractionResult:
        topics = [
            AbstractTopic(
                label=t.get("label", "Unknown"),
                keywords=t.get("keywords", []),
            )
            for t in parsed.topics
            if t.get("label")
        ]

        triples = [
            Triple(
                subject=tr.get("subject", ""),
                relation=tr.get("relation", "RELATED_TO").upper().replace(" ", "_"),
                obj=tr.get("obj", ""),
                subject_type=tr.get("subject_type"),
                obj_type=tr.get("obj_type"),
            )
            for tr in parsed.triples
            if tr.get("subject") and tr.get("obj")
        ]

        return ExtractionResult(
            doc_id=doc_id,
            topics=topics,
            triples=triples,
            sentiment_score=parsed.sentiment_score,
        )

    @staticmethod
    def _correction_prompt(bad_json: str | None, error: str) -> str:
        return f"""\
Your previous response could not be parsed as valid JSON. Error: {error}

Previous response:
{bad_json or '(empty)'}

Please output ONLY a valid JSON object matching this schema exactly:
{{"topics": [...], "triples": [...], "sentiment_score": 0.0}}

No markdown, no explanation — pure JSON only:"""
