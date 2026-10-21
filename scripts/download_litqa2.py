"""Download the LitQA2 subset of LAB-Bench (FutureHouse) from Hugging Face.

Dataset: https://huggingface.co/datasets/futurehouse/lab-bench (CC BY-SA 4.0).
The download is pinned to a specific dataset revision, recorded alongside it.
Data stays under data/raw (gitignored); only this code is committed.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "futurehouse/lab-bench"
OUTPUT_DIR = Path("data/raw/lab-bench")


def main() -> None:
    revision = HfApi().dataset_info(REPO_ID).sha
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["LitQA2/*", "LICENSE", "README.md"],
        local_dir=OUTPUT_DIR,
    )

    provenance = {
        "repo_id": REPO_ID,
        "revision": revision,
        "licence": "CC BY-SA 4.0",
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    files = sorted(
        p.relative_to(OUTPUT_DIR)
        for p in OUTPUT_DIR.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    )
    print(f"Revision: {revision}")
    for file in files:
        print(f"  {file}")


if __name__ == "__main__":
    main()
