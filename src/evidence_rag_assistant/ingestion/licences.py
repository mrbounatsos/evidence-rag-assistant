"""Licence policy and licence detection for JATS full-text XML.

The licence declared in an article's own XML is authoritative. Only
machine-readable Creative Commons URLs count as evidence; licences named
only in prose (e.g. "CC-BY-NC-ND 4.0") are deliberately not recognized.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any

from evidence_rag_assistant.ingestion.jats import local_name

# Licences acceptable for local, non-redistributed use.
# Article text never enters the public repository.
ALLOWED_LICENCES = frozenset(
    {
        "cc0",
        "cc by",
        "cc by-sa",
        "cc by-nd",
        "cc by-nc",
        "cc by-nc-sa",
        "cc by-nc-nd",
    }
)

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

_CC_URL_PATTERN = re.compile(
    r"creativecommons\.org/(?:licenses/(?P<terms>[a-z-]+)|publicdomain/(?P<pd>zero))",
    re.IGNORECASE,
)


def normalize_licence(raw: Any) -> str | None:
    """Normalize a metadata licence string, e.g. ' CC  BY-NC ' -> 'cc by-nc'."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return " ".join(raw.strip().lower().split())


def licence_from_url(text: str) -> str | None:
    """Return the normalized licence for the first Creative Commons URL in text."""
    match = _CC_URL_PATTERN.search(text)
    if match is None:
        return None
    if match.group("pd"):
        return "cc0"
    return "cc " + match.group("terms").lower()


def licence_from_xml(root: ET.Element) -> tuple[str | None, str | None]:
    """Return (normalized licence, evidence string) from JATS licence elements."""
    for element in root.iter():
        if local_name(element.tag) not in {"license", "license_ref"}:
            continue
        # A licence URL can sit on the element itself, in its text, or on a
        # nested <ext-link xlink:href="...">, so check the whole subtree.
        hrefs = [el.get(XLINK_HREF) for el in element.iter()]
        text = " ".join(" ".join(element.itertext()).split())
        for candidate in [*filter(None, hrefs), text]:
            licence = licence_from_url(candidate)
            if licence:
                evidence = text if candidate == text else f"{candidate} | {text}"
                return licence, evidence[:300]
    return None, None
