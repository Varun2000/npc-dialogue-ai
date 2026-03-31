"""
init_db.py
----------
Run once to ingest game lore into ChromaDB.
Safe to re-run — upserts are idempotent.

Usage:
    python -m backend.init_db
"""

import os
from backend.services.rag_service import rag_service

LORE_DIR = os.path.join(os.path.dirname(__file__), "data", "lore")


def main():
    print("Ingesting lore documents into ChromaDB...")
    count = rag_service.ingest_lore_directory(LORE_DIR, collection_name="world_lore")
    total = rag_service.collection_size("world_lore")
    print(f"  Ingested {count} new chunks. Collection now has {total} total chunks.")
    print("Done! ChromaDB is ready.")


if __name__ == "__main__":
    main()
