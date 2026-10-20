"""Discover candidate open-access biomedical articles from Europe PMC.

This step only records candidates and their licence metadata.
It does not download full text or decide what enters the corpus.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OUTPUT_PATH = Path("data/raw/europe_pmc_candidates.jsonl")
PAGE_SIZE = 50

QUERY = (
    "OPEN_ACCESS:Y AND ("
    "TITLE:antibody OR "
    'TITLE:"protein structure prediction" OR '
    'TITLE:"antibody design")'
)

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


def normalize_licence(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return " ".join(raw.strip().lower().split())


def journal_title(article: dict[str, Any]) -> str | None:
    title = article.get("journalInfo", {}).get("journal", {}).get("title")
    return title if isinstance(title, str) else None


def main() -> None:
    params = urlencode(
        {
            "query": QUERY,
            "format": "json",
            "resultType": "core",
            "pageSize": PAGE_SIZE,
        }
    )
    request = Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "EvidenceRAG/0.1 (research portfolio)"},
    )

    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    retrieved_at = datetime.now(UTC).isoformat()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    licence_counts: Counter[str] = Counter()
    count = 0
    eligible = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for article in payload.get("resultList", {}).get("result", []):
            pmcid = article.get("pmcid")
            if not pmcid:
                continue

            licence = normalize_licence(article.get("license"))
            licence_ok = licence in ALLOWED_LICENCES
            licence_counts[licence or "<missing>"] += 1

            record = {
                "source": "Europe PMC",
                "pmcid": pmcid,
                "pmid": article.get("pmid"),
                "doi": article.get("doi"),
                "title": article.get("title"),
                "authors": article.get("authorString"),
                "journal": journal_title(article),
                "publication_year": article.get("pubYear"),
                "is_open_access": article.get("isOpenAccess"),
                "in_epmc": article.get("inEPMC"),
                "licence": licence,
                "licence_ok": licence_ok,
                "source_url": f"https://europepmc.org/articles/{pmcid}",
                "query": QUERY,
                "retrieved_at": retrieved_at,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            eligible += licence_ok

    print(f"Total hits for query: {payload.get('hitCount')}")
    print(f"Candidate records saved: {count}")
    print(f"Licence-eligible: {eligible}")
    print("Licence breakdown:")
    for name, n in licence_counts.most_common():
        print(f"  {name}: {n}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
