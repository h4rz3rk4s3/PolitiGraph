"""
data_loaders/parlamint_loader.py
================================
TEI XML parser for the ParlaMint corpus.

ParlaMint uses the TEI P5 standard with custom extensions:
  - <teiCorpus> or <TEI> root
  - <teiHeader> for corpus/text-level metadata
  - <body> → <div type="debateSection"> → <u who="#speakerID"> utterances
  - Speaker metadata in <teiHeader><particDesc><listPerson>
  - Party/org metadata in <teiHeader><particDesc><listOrg>

Reference: https://github.com/clarin-eric/ParlaMint
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator
from xml.etree import ElementTree as ET

from loguru import logger

from config.models import RawDocument
from data_loaders.base_loader import BaseLoader

# TEI / XML namespace map
_NS: dict[str, str] = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


def _ns(tag: str) -> str:
    """Expand a prefixed tag name using the namespace map."""
    prefix, local = tag.split(":")
    return f"{{{_NS[prefix]}}}{local}"


class ParlaMintLoader(BaseLoader):
    """
    Loads ParlaMint TEI XML files for a given country.

    Parameters
    ----------
    country_code:
        ISO 3166-1 alpha-2 code, e.g. "DE", "FR", "PL".
    subcorpus_filter:
        Optional list of subcorpus tags to include, e.g. ["COVID"].
        If None, all subcorpora are included.
    min_token_count:
        Skip utterances with fewer tokens than this threshold.
        Avoids ingesting procedure-only noise ("Order!", "Agreed.").
    """

    def __init__(
        self,
        country_code: str,
        subcorpus_filter: list[str] | None = None,
        min_token_count: int = 20,
    ) -> None:
        self.country_code = country_code.upper()
        self.subcorpus_filter = subcorpus_filter
        self.min_token_count = min_token_count

    @property
    def source_name(self) -> str:
        return "parlamint"

    def get_metadata_schema(self) -> Dict[str, type]:
        return {
            "session_id": str,
            "subcorpus": str | None,
            "role": str,
            "party": str,
            "speaker_name": str,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def load(self, source_path: str | Path) -> Iterator[RawDocument]:
        """
        Iterate over all .xml files in source_path, yielding one RawDocument
        per valid utterance (<u> element in TEI).
        """
        root_dir = self.validate_source(source_path)

        xml_files = sorted(root_dir.rglob("*.xml"))
        logger.info(f"[ParlaMintLoader] Found {len(xml_files)} XML files in {root_dir}")

        for xml_file in xml_files:
            # ParlaMint distributes a *-root.xml that includes others — skip it
            if xml_file.name.endswith("-root.xml") or xml_file.name.endswith("_root.xml"):
                continue
            try:
                yield from self._parse_file(xml_file)
            except ET.ParseError as e:
                logger.warning(f"[ParlaMintLoader] XML parse error in {xml_file.name}: {e}")
            except Exception as e:
                logger.error(f"[ParlaMintLoader] Unexpected error in {xml_file.name}: {e}")

    # ── Private parsing helpers ───────────────────────────────────────────────

    def _parse_file(self, path: Path) -> Iterator[RawDocument]:
        tree = ET.parse(path)
        root = tree.getroot()

        # Determine root element (teiCorpus or TEI)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag not in ("TEI", "teiCorpus"):
            return  # Not a ParlaMint document

        # ── Extract corpus-level metadata ──
        header = root.find(_ns("tei:teiHeader"))
        if header is None:
            logger.debug(f"No teiHeader in {path.name}, skipping")
            return

        session_id, session_date, language, subcorpus = self._extract_header_meta(header, path)

        # Apply subcorpus filter
        if self.subcorpus_filter and subcorpus not in self.subcorpus_filter:
            return

        # ── Build speaker lookup {speaker_id → {name, party, role}} ──
        speakers = self._extract_speakers(header)
        parties = self._extract_parties(header)

        # ── Iterate utterances ──
        body = root.find(f".//{_ns('tei:body')}")
        if body is None:
            return

        for u in body.iter(_ns("tei:u")):
            try:
                doc = self._parse_utterance(
                    u, session_id, session_date, language,
                    subcorpus, speakers, parties
                )
                if doc is not None:
                    yield doc
            except Exception as e:
                logger.debug(f"Skipping utterance in {path.name}: {e}")

    def _extract_header_meta(
        self, header: ET.Element, path: Path
    ) -> tuple[str, date, str, str | None]:
        """Parse session ID, date, language, and subcorpus from teiHeader."""
        # Session ID from fileDesc/titleStmt
        title_stmt = header.find(f".//{_ns('tei:titleStmt')}")
        session_id = path.stem  # fallback

        if title_stmt is not None:
            meeting = title_stmt.find(f".//{_ns('tei:meeting')}")
            if meeting is not None and meeting.get("n"):
                session_id = meeting.get("n", path.stem)

        # Date from profileDesc/settingDesc/setting/date
        session_date = date.today()
        date_el = header.find(f".//{_ns('tei:settingDesc')}/{_ns('tei:setting')}/{_ns('tei:date')}")
        if date_el is not None:
            when = date_el.get("when") or date_el.get(f"{{{_NS['xml']}}}id", "")
            try:
                session_date = self._parse_date(when)
            except ValueError:
                pass

        # Language
        lang_el = header.find(f".//{_ns('tei:langUsage')}/{_ns('tei:language')}")
        language = "unk"
        if lang_el is not None:
            language = lang_el.get("ident", "unk")

        # Subcorpus from textClass/catRef
        subcorpus = None
        cat_ref = header.find(f".//{_ns('tei:textClass')}/{_ns('tei:catRef')}")
        if cat_ref is not None:
            target = cat_ref.get("target", "")
            # e.g. "#parlamint.corpus.covid"  → "COVID"
            subcorpus = target.split(".")[-1].upper() if target else None

        return session_id, session_date, language, subcorpus

    def _extract_speakers(self, header: ET.Element) -> dict[str, dict[str, str]]:
        """Build {speaker_id → {name, role}} from particDesc/listPerson."""
        result: dict[str, dict[str, str]] = {}
        list_person = header.find(f".//{_ns('tei:particDesc')}/{_ns('tei:listPerson')}")
        if list_person is None:
            return result

        for person in list_person.iter(_ns("tei:person")):
            pid = person.get(f"{{{_NS['xml']}}}id", "")
            if not pid:
                continue

            # Name: prefer <persName full="yes">, fall back to concatenation
            name = "Unknown"
            pers_name = person.find(_ns("tei:persName"))
            if pers_name is not None:
                forename = pers_name.findtext(_ns("tei:forename"), "")
                surname = pers_name.findtext(_ns("tei:surname"), "")
                name = f"{forename} {surname}".strip() or "Unknown"

            # Role from <roleName>
            role = person.findtext(f".//{_ns('tei:roleName')}", "MP")

            # Party affiliation: <affiliation role="member" ref="#PartyID">
            party = "Unknown"
            for aff in person.iter(_ns("tei:affiliation")):
                if aff.get("role") in ("member", "memberOf"):
                    ref = aff.get("ref", "").lstrip("#")
                    if ref:
                        party = ref
                        break

            result[pid] = {"name": name, "role": role, "party": party}

        return result

    def _extract_parties(self, header: ET.Element) -> dict[str, str]:
        """Build {party_id → party_name} from particDesc/listOrg."""
        result: dict[str, str] = {}
        list_org = header.find(f".//{_ns('tei:particDesc')}/{_ns('tei:listOrg')}")
        if list_org is None:
            return result

        for org in list_org.iter(_ns("tei:org")):
            oid = org.get(f"{{{_NS['xml']}}}id", "")
            if not oid:
                continue
            name_el = org.find(_ns("tei:orgName"))
            name = name_el.text.strip() if name_el is not None and name_el.text else oid
            result[oid] = name

        return result

    def _parse_utterance(
        self,
        u: ET.Element,
        session_id: str,
        session_date: date,
        language: str,
        subcorpus: str | None,
        speakers: dict[str, dict[str, str]],
        parties: dict[str, str],
    ) -> RawDocument | None:
        # Speaker reference: who="#SpeakerID"
        raw_who = u.get("who", "").lstrip("#")
        if not raw_who:
            return None

        speaker_info = speakers.get(raw_who, {
            "name": raw_who, "role": "MP", "party": "Unknown"
        })

        # Extract text: concatenate all <seg> children
        segments = u.findall(f".//{_ns('tei:seg')}")
        if segments:
            text = " ".join(
                "".join(seg.itertext()).strip()
                for seg in segments
            ).strip()
        else:
            text = "".join(u.itertext()).strip()

        # Filter noise
        token_count = len(text.split())
        if token_count < self.min_token_count:
            return None

        # Resolve party ID → party name
        party_id = speaker_info.get("party", "Unknown")
        party_name = parties.get(party_id, party_id)

        # Build the speech ID
        speech_id = u.get(f"{{{_NS['xml']}}}id") or f"{session_id}_{raw_who}_{id(u)}"

        return RawDocument(
            doc_id=speech_id,
            source="parlamint",
            country_code=self.country_code,
            language=language,
            text=text,
            speaker_id=raw_who,
            speaker_name=speaker_info.get("name", "Unknown"),
            party=party_name,
            role=speaker_info.get("role", "MP"),
            date=session_date,
            session_id=session_id,
            subcorpus=subcorpus,
            metadata={"token_count": token_count},
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(s: str) -> date:
        """Parse various date formats found in ParlaMint XML."""
        s = s.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        # Try extracting from a filename-like string: e.g. "ParlaMint-DE_2020-03-13"
        m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        raise ValueError(f"Cannot parse date: '{s}'")
