"""Minimal client for the Europe PMC REST search API."""

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = "EvidenceRAG/0.1 (research portfolio)"
DOI_PREFIX = "https://doi.org/"


def normalize_doi(url: str) -> str:
    """'https://doi.org/10.1128/MSphere.00109-24' -> '10.1128/msphere.00109-24'"""
    doi = url.strip()
    if doi.lower().startswith(DOI_PREFIX):
        doi = doi[len(DOI_PREFIX) :]
    return doi.lower()


def search(query: str, page_size: int = 25, timeout: float = 30) -> list[dict[str, Any]]:
    """Return Europe PMC 'core' search results for a query."""
    params = urlencode(
        {"query": query, "format": "json", "resultType": "core", "pageSize": page_size}
    )
    request = Request(f"{SEARCH_URL}?{params}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    results: list[dict[str, Any]] = payload.get("resultList", {}).get("result", [])
    return results


def journal_title(article: dict[str, Any]) -> str | None:
    title = article.get("journalInfo", {}).get("journal", {}).get("title")
    return title if isinstance(title, str) else None
