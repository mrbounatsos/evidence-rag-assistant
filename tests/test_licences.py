"""Tests for licence detection, based on publisher patterns seen in real data."""

import xml.etree.ElementTree as ET

import pytest

from evidence_rag_assistant.ingestion.licences import (
    licence_from_url,
    licence_from_xml,
    normalize_licence,
)

NAMESPACES = (
    'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
)


def make_article(permissions: str) -> ET.Element:
    """Wrap a <permissions> block in a minimal JATS article."""
    return ET.fromstring(
        f"<article {NAMESPACES}><front><article-meta>{permissions}</article-meta></front></article>"
    )


@pytest.mark.parametrize(
    ("permissions", "expected"),
    [
        pytest.param(
            "<permissions><license><license-p>Distributed under the terms of "
            "the Creative Commons Attribution-NonCommercial License "
            "(https://creativecommons.org/licenses/by-nc/4.0/).</license-p>"
            "</license></permissions>",
            "cc by-nc",
            id="url-in-prose",
        ),
        pytest.param(
            "<permissions><license><ali:license_ref>"
            "https://creativecommons.org/licenses/by/4.0/</ali:license_ref>"
            "<license-p>This article is licensed under CC-BY 4.0</license-p>"
            "</license></permissions>",
            "cc by",
            id="ali-license-ref",
        ),
        pytest.param(
            "<permissions><license><license-p>Distributed under the terms of the "
            '<ext-link xlink:href="https://creativecommons.org/licenses/by/4.0/">'
            "Creative Commons Attribution License</ext-link>.</license-p>"
            "</license></permissions>",
            "cc by",
            id="nested-ext-link",
        ),
        pytest.param(
            '<permissions><license xlink:href="'
            'https://creativecommons.org/licenses/by-nc-nd/4.0/">'
            "<license-p>Open access.</license-p></license></permissions>",
            "cc by-nc-nd",
            id="href-on-license",
        ),
        pytest.param(
            "<permissions><license><license-p>"
            "This article is licensed under CC-BY-NC-ND 4.0"
            "</license-p></license></permissions>",
            None,
            id="abbreviation-only",
        ),
        pytest.param(
            "<permissions><copyright-statement>© 2026 The Authors"
            "</copyright-statement></permissions>",
            None,
            id="no-licence",
        ),
    ],
)
def test_licence_from_xml(permissions: str, expected: str | None) -> None:
    licence, evidence = licence_from_xml(make_article(permissions))
    assert licence == expected
    assert (evidence is None) == (expected is None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "cc by"),
        ("HTTPS://CreativeCommons.org/licenses/BY-SA/4.0", "cc by-sa"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("https://example.com/terms-of-use", None),
    ],
)
def test_licence_from_url(text: str, expected: str | None) -> None:
    assert licence_from_url(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" CC  BY-NC ", "cc by-nc"),
        ("cc by", "cc by"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_licence(raw: object, expected: str | None) -> None:
    assert normalize_licence(raw) == expected
