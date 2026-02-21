"""
tests/test_parlamint_loader.py
================================
Unit tests for the ParlaMint TEI XML loader.
No network or database access required.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path
import tempfile
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.parlamint_loader import ParlaMintLoader


MINIMAL_TEI_XML = textwrap.dedent("""
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"
     xml:id="ParlaMint-DE_2021-09-15">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">ParlaMint-DE, Regular Session [ParlaMint]</title>
        <meeting n="DE_2021-09-15">Bundestag Plenary 2021-09-15</meeting>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <settingDesc>
        <setting>
          <date when="2021-09-15">September 15, 2021</date>
        </setting>
      </settingDesc>
      <langUsage>
        <language ident="de">German</language>
      </langUsage>
      <textClass>
        <catRef target="#parlamint.corpus.regular"/>
      </textClass>
      <particDesc>
        <listPerson>
          <person xml:id="AnnaSchmidt">
            <persName>
              <forename>Anna</forename>
              <surname>Schmidt</surname>
            </persName>
            <roleName>MP</roleName>
            <affiliation role="member" ref="#CDU"/>
          </person>
          <person xml:id="MaxMüller">
            <persName>
              <forename>Max</forename>
              <surname>Müller</surname>
            </persName>
            <roleName>MP</roleName>
            <affiliation role="member" ref="#SPD"/>
          </person>
        </listPerson>
        <listOrg>
          <org xml:id="CDU">
            <orgName>Christlich Demokratische Union</orgName>
          </org>
          <org xml:id="SPD">
            <orgName>Sozialdemokratische Partei Deutschlands</orgName>
          </org>
        </listOrg>
      </particDesc>
    </profileDesc>
  </teiHeader>
  <body>
    <div type="debateSection">
      <u who="#AnnaSchmidt" xml:id="speech001">
        <seg>Die Bundesregierung hat in den vergangenen Jahren erhebliche Fortschritte bei der
        Energiewende erzielt. Wir müssen jedoch noch mehr tun, um unsere Klimaziele zu erreichen
        und gleichzeitig die wirtschaftliche Wettbewerbsfähigkeit Deutschlands zu erhalten.</seg>
      </u>
      <u who="#MaxMüller" xml:id="speech002">
        <seg>Die Energiepreise steigen dramatisch und belasten die Haushalte der Bürgerinnen und
        Bürger. Die Regierung muss endlich handeln und die Menschen vor diesen massiven
        Kostensteigerungen schützen. Das ist eine Frage der sozialen Gerechtigkeit.</seg>
      </u>
      <u who="#AnnaSchmidt" xml:id="speech003">
        <seg>Kurz.</seg>
      </u>
    </div>
  </body>
</TEI>
""").strip()


@pytest.fixture
def tei_file(tmp_path: Path) -> Path:
    """Write a minimal valid TEI XML file to a temp directory."""
    f = tmp_path / "ParlaMint-DE_2021-09-15.xml"
    f.write_text(MINIMAL_TEI_XML, encoding="utf-8")
    return tmp_path


class TestParlaMintLoader:

    def test_loads_correct_number_of_speeches(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        # speech003 "Kurz." should be filtered out (< 5 tokens)
        assert len(docs) == 2

    def test_speaker_metadata_extracted(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        speakers = {d.speaker_name for d in docs}
        assert "Anna Schmidt" in speakers
        assert "Max Müller" in speakers

    def test_party_resolved_correctly(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        party_map = {d.speaker_name: d.party for d in docs}
        assert party_map["Anna Schmidt"] == "Christlich Demokratische Union"
        assert party_map["Max Müller"] == "Sozialdemokratische Partei Deutschlands"

    def test_date_parsed(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        for doc in docs:
            assert doc.date == date(2021, 9, 15)

    def test_language_extracted(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        for doc in docs:
            assert doc.language == "de"

    def test_country_code_set(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="de", min_token_count=5)
        docs = list(loader.load(tei_file))
        for doc in docs:
            assert doc.country_code == "DE"  # should be uppercased

    def test_source_is_parlamint(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        docs = list(loader.load(tei_file))
        for doc in docs:
            assert doc.source == "parlamint"

    def test_min_token_filter(self, tei_file: Path):
        """With a high min_token threshold, all speeches should be filtered."""
        loader = ParlaMintLoader(country_code="DE", min_token_count=1000)
        docs = list(loader.load(tei_file))
        assert len(docs) == 0

    def test_batch_yields_lists(self, tei_file: Path):
        loader = ParlaMintLoader(country_code="DE", min_token_count=5)
        batches = list(loader.batch(tei_file, batch_size=1))
        assert all(isinstance(b, list) for b in batches)
        flat = [d for b in batches for d in b]
        assert len(flat) == 2

    def test_nonexistent_path_raises(self):
        loader = ParlaMintLoader(country_code="DE")
        with pytest.raises(FileNotFoundError):
            list(loader.load("/nonexistent/path/to/data"))


class TestComplexityCalculator:

    def test_basic_metrics(self):
        from nlp_engines.complexity_calculator import ComplexityCalculator

        calc = ComplexityCalculator()
        text = (
            "Die Bundesregierung muss endlich handeln. "
            "Die Energiepreise explodieren und die Bürger können sich das nicht mehr leisten. "
            "Wir fordern sofortige Steuersenkungen auf Energie und ein Ende der Abhängigkeit."
        )
        metrics = calc.compute(text, language="de")

        assert "ttr" in metrics
        assert "cttr" in metrics
        assert "avg_sentence_length" in metrics
        assert 0.0 < metrics["ttr"] <= 1.0
        assert metrics["avg_sentence_length"] > 0

    def test_empty_text_returns_empty(self):
        from nlp_engines.complexity_calculator import ComplexityCalculator

        calc = ComplexityCalculator()
        result = calc.compute("", language="de")
        assert result == {}

    def test_ttr_decreases_with_repetition(self):
        from nlp_engines.complexity_calculator import ComplexityCalculator

        calc = ComplexityCalculator()
        rich = "The cat sat on the mat under the tree near the house by the river"
        poor = "The the the the the the the the the the the the the the"
        assert calc._ttr(rich) > calc._ttr(poor)
