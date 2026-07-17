import os
import logging
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSIONS = 768  # gemini-embedding-001, balanced quality/storage default


class ChromaService:
    """Wraps ChromaDB connection and collection access for storing/querying
    document chunk embeddings.

    Note: Chroma's collection is created once here (get_or_create), matching
    the chosen embedding dimensionality. If you ever change the embedding
    model or dimensionality, you must create a new collection — vectors of
    different dimensions are not comparable within the same collection.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "document_chunks",
    ):
        host = host or os.environ.get("CHROMA_HOST", "chroma")
        port = int(port or os.environ.get("CHROMA_PORT", 8000))

        self._client = chromadb.HttpClient(host=host, port=port)

        # metadata hint is informational only — Chroma infers actual dimensionality
        # from the first vectors added, so ensure your embedding calls are
        # consistent with DEFAULT_EMBEDDING_DIMENSIONS from the start.
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_dimensions": DEFAULT_EMBEDDING_DIMENSIONS},
        )
        logger.info(f"Connected to Chroma collection '{collection_name}' at {host}:{port}")

    # --- Insert ---------------------------------------------------------

    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """Store a batch of chunk embeddings. `ids` should be the same UUIDs
        used as _id in the corresponding Mongo chunk documents, so a Chroma
        hit can be joined back to its full text/metadata in Mongo."""
        if not ids:
            return
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    # --- Query (for later, once search/retrieval is built) --------------

    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )