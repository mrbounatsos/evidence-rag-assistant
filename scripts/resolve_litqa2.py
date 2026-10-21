"""Resolve LitQA2 source DOIs to Europe PMC records with licence metadata.

Writes one candidate record per source DOI, in the same shape as the antibody
discovery candidates, so the existing licence-verified fetch can process them.
"""

import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from evidence_rag_assistant.ingestion.europe_pmc import journal_title, search
from evidence_rag_assistant.ingestion.licences import ALLOWED_LICENCES, normalize_licence

LITQA2_DIR = Path("data/raw/lab-bench/LitQA2")
OUTPUT_PATH = Path("data/raw/litqa2_candidates.jsonl")
DOI_PREFIX = "https://doi.org/"
REQUEST_DELAY_SECONDS = 0.3


def normalize_doi(url: str) -> str:
    """'https://doi.org/10.1128/MSphere.00109-24' -> '10.1128/msphere.00109-24'"""
    doi = url.strip()
    if doi.lower().startswith(DOI_PREFIX):
        doi = doi[len(DOI_PREFIX) :]
    return doi.lower()


def main() -> None:
    rows = pq.read_table(LITQA2_DIR).to_pylist()
    questions_by_doi: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for source in row["sources"] or []:
            questions_by_doi[normalize_doi(source)].append(row["id"])

    retrieved_at = datetime.now(UTC).isoformat()
    counts: Counter[str] = Counter()
    covered_questions: set[str] = set()

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for index, (doi, question_ids) in enumerate(sorted(questions_by_doi.items()), 1):
            query = f'DOI:"{doi}"'
            results = search(query, page_size=1)
            time.sleep(REQUEST_DELAY_SECONDS)
            article = results[0] if results else {}
            licence = normalize_licence(article.get("license"))
            record = {
                "source": "LitQA2",
                "doi": doi,
                "litqa2_question_ids": question_ids,
                "pmcid": article.get("pmcid"),
                "pmid": article.get("pmid"),
                "title": article.get("title"),
                "journal": journal_title(article),
                "publication_year": article.get("pubYear"),
                "is_open_access": article.get("isOpenAccess"),
                "in_epmc": article.get("inEPMC"),
                "licence": licence,
                "licence_ok": licence in ALLOWED_LICENCES,
                "query": query,
                "retrieved_at": retrieved_at,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            counts["found"] += bool(article)
            counts["has_pmcid"] += bool(record["pmcid"])
            counts["full_text_in_epmc"] += record["in_epmc"] == "Y"
            eligible = bool(record["pmcid"]) and record["in_epmc"] == "Y" and licence_ok(record)
            counts["eligible"] += eligible
            if eligible:
                covered_questions.update(question_ids)
            print(f"[{index}/{len(questions_by_doi)}] {doi}: {record['pmcid'] or '-'}")

    print(f"\nQuestions: {len(rows)} | distinct source DOIs: {len(questions_by_doi)}")
    print(f"Found in Europe PMC: {counts['found']}")
    print(f"With PMCID: {counts['has_pmcid']}")
    print(f"Full text in Europe PMC: {counts['full_text_in_epmc']}")
    print(f"Eligible (PMCID + full text + allowed licence): {counts['eligible']}")
    print(f"Questions whose source paper is eligible: {len(covered_questions)}")
    print(f"Output: {OUTPUT_PATH}")


def licence_ok(record: dict[str, object]) -> bool:
    return bool(record["licence_ok"])


if __name__ == "__main__":
    main()
