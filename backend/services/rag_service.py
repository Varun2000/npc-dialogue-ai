"""
RAG Service (ChromaDB + Sentence Transformers)
----------------------------------------------
Loads game lore documents into a persistent ChromaDB collection,
then retrieves the most relevant chunks at query time to inject
into the NPC's system prompt as grounded context.

This gives NPCs accurate, consistent knowledge of the game world
without hallucinating facts.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from backend.config import get_settings

settings = get_settings()

# Use a local sentence-transformer model so there are no extra API calls.
# all-MiniLM-L6-v2 is fast (80ms/query on CPU) and ~80MB download.
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 600      # characters per chunk
_CHUNK_OVERLAP = 100   # overlap between chunks


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split a long document into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny trailing chunks


class RAGService:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL
        )
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def ingest_lore_directory(self, lore_dir: str, collection_name: str = "world_lore") -> int:
        """
        Read all .txt files in lore_dir, chunk them, and upsert into ChromaDB.
        Safe to run multiple times — uses file path + chunk index as document ID.
        Returns number of chunks ingested.
        """
        collection = self._get_collection(collection_name)
        total = 0

        for filepath in glob.glob(os.path.join(lore_dir, "*.txt")):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = _chunk_text(content)
            filename = os.path.basename(filepath)

            ids       = [f"{filename}::chunk{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

            # Upsert so re-running init_db is idempotent
            collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
            total += len(chunks)

        return total

    def query(
        self,
        query_text: str,
        collection_name: str = "world_lore",
        top_k: int | None = None,
    ) -> list[str]:
        """
        Retrieve the top-k most relevant lore chunks for a player query.
        Returns a list of document strings (empty list if collection is empty).
        """
        k = top_k or settings.rag_top_k
        collection = self._get_collection(collection_name)

        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=min(k, collection.count()),
        )
        return results["documents"][0] if results["documents"] else []

    def collection_size(self, collection_name: str = "world_lore") -> int:
        return self._get_collection(collection_name).count()


rag_service = RAGService()
