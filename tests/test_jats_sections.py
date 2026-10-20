"""Tests for article-level extraction: which parts of a JATS article are kept."""

import xml.etree.ElementTree as ET

import pytest

from evidence_rag_assistant.ingestion.jats import (
    Block,
    extract_article,
    normalize_title,
    skip_reason,
)

ARTICLE = """
<article>
  <front><article-meta>
    <title-group><article-title>A <italic>test</italic> article</article-title></title-group>
    <abstract><title>Abstract</title><p>Abstract text.</p></abstract>
    <abstract abstract-type="graphical"><p>Graphical summary.</p></abstract>
  </article-meta></front>
  <body>
    <p>Untitled opening.</p>
    <sec><title>1. Introduction</title>
      <p>Intro text.</p>
      <list><list-item><p>A list item.</p></list-item></list>
    </sec>
    <sec><title>Results</title>
      <sec><title>Binding</title>
        <p>Binding text.<fig><label>Figure 1</label>
          <caption><title>Overview.</title><p>Details.</p></caption></fig></p>
      </sec>
      <table-wrap><label>Table 1</label></table-wrap>
    </sec>
    <sec sec-type="ack"><title>Thanks to colleagues</title><p>Thanks.</p></sec>
    <sec><title>5 Funding</title><p>Funded by a grant.</p></sec>
    <sec><title>References</title><sec><ref-list><ref/></ref-list></sec></sec>
    <sec><title>Associated Data</title>
      <sec><title>Supplementary Materials</title><supplementary-material/></sec>
    </sec>
  </body>
  <back><ref-list/></back>
</article>
"""


def test_extract_article_keeps_content() -> None:
    blocks, _ = extract_article(ET.fromstring(ARTICLE))
    assert blocks == [
        Block("title", (), "A test article"),
        Block("abstract", ("Abstract",), "Abstract text."),
        Block("paragraph", (), "Untitled opening."),
        Block("paragraph", ("1. Introduction",), "Intro text."),
        Block("paragraph", ("1. Introduction",), "A list item."),
        Block("paragraph", ("Results", "Binding"), "Binding text."),
        Block("caption", ("Results", "Binding"), "Figure 1: Overview. Details."),
    ]


def test_extract_article_records_skips() -> None:
    _, skips = extract_article(ET.fromstring(ARTICLE))
    assert [skip.reason for skip in skips] == [
        "typed_abstract",
        "table_not_extracted",
        "sec_type:ack",
        "title:funding",
        "contains_ref_list",
        "title:associated data",
        "in_back",
    ]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("1. Introduction", "introduction"),
        ("2.3 Results", "results"),
        ("Acknowledgements:", "acknowledgements"),
        ("  Data   Availability ", "data availability"),
        ("3D structure prediction", "3d structure prediction"),
        ("V. cholerae infection", "v. cholerae infection"),
    ],
)
def test_normalize_title(title: str, expected: str) -> None:
    assert normalize_title(title) == expected


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        pytest.param(
            "<sec><title>Methods</title><p>We did things.</p></sec>",
            None,
            id="content-section-kept",
        ),
        pytest.param(
            '<sec sec-type="ack"><title>Methods</title><p>x</p></sec>',
            "sec_type:ack",
            id="sec-type-beats-title",
        ),
        pytest.param(
            "<sec><title>Extras</title><supplementary-material/></sec>",
            "only_supplementary_material",
            id="only-supplementary-material",
        ),
    ],
)
def test_skip_reason(xml: str, expected: str | None) -> None:
    assert skip_reason(ET.fromstring(xml)) == expected
