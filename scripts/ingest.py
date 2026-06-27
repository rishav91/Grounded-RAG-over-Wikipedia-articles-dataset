#!/usr/bin/env python3
"""M0 batch ingestion entrypoint.

Usage: python scripts/ingest.py [--limit N]
"""

from __future__ import annotations

import argparse
import logging

from grounded_rag.config import MVP_SLICE_SIZE
from grounded_rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=MVP_SLICE_SIZE, help="number of documents to ingest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run_ingestion(limit=args.limit)
    print(f"ingested {stats.num_docs} documents -> {stats.num_chunks} chunks")


if __name__ == "__main__":
    main()
