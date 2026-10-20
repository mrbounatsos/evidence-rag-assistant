"""Extract clean text blocks from verified JATS full-text XML.

Reads the fetch manifest, re-checks each file's SHA-256 against it, and writes
one record per extracted block plus an audit of everything skipped.
All output stays under data/processed, which is gitignored.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evidence_rag_assistant.ingestion.jats import extract_article

MANIFEST_PATH = Path("data/raw/fulltext_manifest.jsonl")
BLOCKS_PATH = Path("data/processed/blocks.jsonl")
SKIPS_PATH = Path("data/processed/extraction_skips.jsonl")
VERIFIED_STATUSES = frozenset({"stored", "cached"})


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    verified = [e for e in manifest if e["status"] in VERIFIED_STATUSES]

    BLOCKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    unhandled: Counter[str] = Counter()
    words = 0
    extracted = 0
    hash_failures: list[str] = []
    no_paragraphs: list[str] = []

    with (
        BLOCKS_PATH.open("w", encoding="utf-8") as blocks_out,
        SKIPS_PATH.open("w", encoding="utf-8") as skips_out,
    ):
        for entry in verified:
            pmcid = entry["pmcid"]
            data = Path(entry["xml_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                hash_failures.append(pmcid)
                continue

            blocks, skips = extract_article(ET.fromstring(data))
            extracted += 1

            for index, block in enumerate(blocks):
                record = {
                    "pmcid": pmcid,
                    "block_index": index,
                    **asdict(block),
                    "licence": entry["xml_licence"],
                    "source_sha256": entry["sha256"],
                }
                blocks_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kinds[block.kind] += 1
                words += len(block.text.split())

            for skip in skips:
                record = {"pmcid": pmcid, **asdict(skip)}
                skips_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                reasons[skip.reason] += 1
                if skip.reason == "unhandled_element":
                    unhandled[skip.element] += 1

            if not any(block.kind == "paragraph" for block in blocks):
                no_paragraphs.append(pmcid)

    print(f"Articles extracted: {extracted} / {len(verified)}")
    print(f"Hash mismatches (not extracted): {hash_failures or 'none'}")
    print(f"Blocks by kind: {dict(kinds)}")
    print(f"Total words: {words:,}")
    print(f"Articles with no body paragraphs: {no_paragraphs or 'none'}")
    print("\nSkip reasons:")
    for reason, n in reasons.most_common():
        print(f"  {n:4d}  {reason}")
    print("\nUnhandled elements:")
    for element, n in unhandled.most_common():
        print(f"  {n:4d}  {element}")
    print(f"\nOutput: {BLOCKS_PATH}, {SKIPS_PATH}")


if __name__ == "__main__":
    main()
