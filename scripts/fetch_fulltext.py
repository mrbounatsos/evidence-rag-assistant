"""Download full-text JATS XML for licence-eligible Europe PMC candidates.

Each article's licence is re-read from its own XML (the authoritative source).
XML is written to disk only when that licence is allowed. Every attempt is
recorded in a manifest. All output stays under data/raw, which is gitignored.
"""

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CANDIDATE_PATHS = (
    Path("data/raw/europe_pmc_candidates.jsonl"),
    Path("data/raw/litqa2_candidates.jsonl"),
)
FULLTEXT_DIR = Path("data/raw/fulltext")
MANIFEST_PATH = Path("data/raw/fulltext_manifest.jsonl")
API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
USER_AGENT = "EvidenceRAG/0.1 (research portfolio)"
REQUEST_DELAY_SECONDS = 0.5

# Duplicated from discover_europe_pmc.py; moves into src/ in the next step.
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
CC_URL_PATTERN = re.compile(
    r"creativecommons\.org/(?:licenses/(?P<terms>[a-z-]+)|publicdomain/(?P<pd>zero))",
    re.IGNORECASE,
)


def licence_from_url(text: str) -> str | None:
    match = CC_URL_PATTERN.search(text)
    if match is None:
        return None
    if match.group("pd"):
        return "cc0"
    return "cc " + match.group("terms").lower()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def fetch_xml(pmcid: str) -> bytes:
    request = Request(API_URL.format(pmcid=pmcid), headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        data: bytes = response.read()
    return data


def process(candidate: dict[str, Any]) -> dict[str, Any]:
    pmcid = candidate["pmcid"]
    xml_path = FULLTEXT_DIR / f"{pmcid}.xml"
    entry: dict[str, Any] = {
        "pmcid": pmcid,
        "source": candidate.get("source"),
        "metadata_licence": candidate.get("licence"),
        "xml_licence": None,
        "xml_licence_evidence": None,
        "licence_match": False,
        "xml_path": None,
        "sha256": None,
        "status": "",
        "error": None,
        "checked_at": datetime.now(UTC).isoformat(),
    }

    cached = xml_path.exists()
    if cached:
        data = xml_path.read_bytes()
    else:
        try:
            data = fetch_xml(pmcid)
        except (HTTPError, URLError, TimeoutError) as error:
            entry["status"] = "fetch_error"
            entry["error"] = str(error)
            return entry
        finally:
            time.sleep(REQUEST_DELAY_SECONDS)

    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        entry["status"] = "parse_error"
        entry["error"] = str(error)
        return entry

    licence, evidence = licence_from_xml(root)
    entry["xml_licence"] = licence
    entry["xml_licence_evidence"] = evidence
    entry["licence_match"] = licence == candidate.get("licence")
    entry["sha256"] = hashlib.sha256(data).hexdigest()

    if licence is None:
        entry["status"] = "licence_unverified"
    elif licence not in ALLOWED_LICENCES:
        entry["status"] = "licence_not_allowed"
    else:
        if not cached:
            xml_path.write_bytes(data)
        entry["xml_path"] = str(xml_path)
        entry["status"] = "cached" if cached else "stored"
    return entry


def load_eligible_candidates() -> list[dict[str, Any]]:
    """Candidates with a PMCID, full text in Europe PMC and an allowed index licence.

    The index licence is only a first filter; the article's own XML decides.
    A paper listed in several candidate files is fetched once.
    """
    eligible: dict[str, dict[str, Any]] = {}
    for path in CANDIDATE_PATHS:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                candidate = json.loads(line)
                if (
                    candidate.get("licence_ok")
                    and candidate.get("pmcid")
                    and candidate.get("in_epmc") == "Y"
                ):
                    eligible.setdefault(candidate["pmcid"], candidate)
    return list(eligible.values())


def main() -> None:
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = load_eligible_candidates()

    statuses: Counter[str] = Counter()
    mismatches: list[str] = []

    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest:
        for index, candidate in enumerate(eligible, start=1):
            entry = process(candidate)
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            statuses[entry["status"]] += 1
            if entry["xml_licence"] and not entry["licence_match"]:
                mismatches.append(
                    f"{entry['pmcid']}: metadata={entry['metadata_licence']} "
                    f"xml={entry['xml_licence']}"
                )
            print(f"[{index}/{len(eligible)}] {entry['pmcid']}: {entry['status']}")

    print(f"\nStatus summary: {dict(statuses)}")
    print(f"Licence mismatches: {len(mismatches)}")
    for line in mismatches:
        print(f"  {line}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
