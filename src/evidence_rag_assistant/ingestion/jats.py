"""Text extraction from JATS full-text XML.

Text within a block element (paragraph, title, caption) is joined without
separators, so inline formatting such as <bold>Dy</bold>ad cannot split
words. Numbered citation markers are dropped; author-year citations are kept
because they are part of the sentence.
"""

import re
import xml.etree.ElementTree as ET

# Elements nested inside a block that are not part of its prose.
# Figure and table captions are extracted separately.
_SKIP_INSIDE_TEXT = frozenset({"fig", "table-wrap"})

_NUMERIC_CITATION = re.compile(r"^[\s\d,;\-–—\[\]()]+$")
_EMPTY_BRACKETS = re.compile(r"[\[(][\s,;\-–—]*[\])]")
_REPEATED_SEPARATORS = re.compile(r"([,;])(?:\s*[,;])+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:)\]])")


def local_name(tag: str) -> str:
    """Strip an XML namespace: '{http://...}license' -> 'license'."""
    return tag.rsplit("}", 1)[-1]


def _is_numbered_citation(element: ET.Element) -> bool:
    if local_name(element.tag) != "xref" or element.get("ref-type") != "bibr":
        return False
    return bool(_NUMERIC_CITATION.match("".join(element.itertext())))


def _collect(element: ET.Element, parts: list[str]) -> None:
    if element.text:
        parts.append(element.text)
    for child in element:
        skip = local_name(child.tag) in _SKIP_INSIDE_TEXT or _is_numbered_citation(child)
        if not skip:
            _collect(child, parts)
        # Text after a child is stored as the child's .tail, so it is kept
        # even when the child itself is dropped.
        if child.tail:
            parts.append(child.tail)


def clean_text(text: str) -> str:
    """Collapse whitespace and tidy punctuation left behind by removed citations."""
    text = " ".join(text.split())
    text = _EMPTY_BRACKETS.sub("", text)
    text = _REPEATED_SEPARATORS.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return " ".join(text.split())


def block_text(element: ET.Element) -> str:
    """Return the clean text of a block element such as <p>, <title> or <caption>."""
    parts: list[str] = []
    _collect(element, parts)
    return clean_text("".join(parts))
