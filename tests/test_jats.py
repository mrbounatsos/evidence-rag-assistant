"""Tests for JATS text extraction, based on patterns seen in the corpus."""

import xml.etree.ElementTree as ET

import pytest

from evidence_rag_assistant.ingestion.jats import block_text


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        pytest.param(
            "<p><bold>Dy</bold>ad models.</p>",
            "Dyad models.",
            id="inline-formatting-does-not-split-words",
        ),
        pytest.param(
            '<p>Widely used drugs,<xref ref-type="bibr" rid="b1">1</xref>,'
            '<xref ref-type="bibr" rid="b2">2</xref> as shown before.</p>',
            "Widely used drugs, as shown before.",
            id="superscript-style-citations",
        ),
        pytest.param(
            '<p>Shown before [<xref ref-type="bibr" rid="b1">1</xref>, '
            '<xref ref-type="bibr" rid="b2">2</xref>].</p>',
            "Shown before.",
            id="bracketed-citations",
        ),
        pytest.param(
            '<p>Earlier work [<xref ref-type="bibr" rid="b3">3</xref>–'
            '<xref ref-type="bibr" rid="b5">5</xref>] used this.</p>',
            "Earlier work used this.",
            id="citation-range",
        ),
        pytest.param(
            '<p>Prior models (<xref ref-type="bibr" rid="b1">1</xref>; '
            '<xref ref-type="bibr" rid="b2">2</xref>) failed.</p>',
            "Prior models failed.",
            id="parenthesized-citations",
        ),
        pytest.param(
            '<p>Antibodies<sup><xref ref-type="bibr" rid="b1">1</xref></sup> bind.</p>',
            "Antibodies bind.",
            id="citation-inside-superscript",
        ),
        pytest.param(
            '<p>As shown by <xref ref-type="bibr" rid="b1">Smith et al., 2020</xref>,'
            " this works.</p>",
            "As shown by Smith et al., 2020, this works.",
            id="author-year-citation-kept",
        ),
        pytest.param(
            '<p>See <xref ref-type="fig" rid="f1">Figure 1</xref>.</p>',
            "See Figure 1.",
            id="figure-reference-kept",
        ),
        pytest.param(
            "<p>Results are shown.<fig><caption><p>A caption.</p></caption></fig> More text.</p>",
            "Results are shown. More text.",
            id="nested-figure-skipped",
        ),
        pytest.param(
            "<p>  Spread\n   over\tlines.  </p>",
            "Spread over lines.",
            id="whitespace-collapsed",
        ),
    ],
)
def test_block_text(xml: str, expected: str) -> None:
    assert block_text(ET.fromstring(xml)) == expected
