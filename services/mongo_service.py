import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)


class MongoService:
    """Wraps MongoDB connection, dedup checks, and chunk/tracking-record storage.

    Retries the initial connection since Mongo can take a moment to accept
    connections after container start (compose `depends_on` only waits for
    the container to start, not for the service inside it to be ready).
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = "rag_db",
        max_retries: int = 8,
        retry_delay_seconds: float = 2.0,
    ):
        uri = uri or os.environ["MONGO_URI"]

        self._client = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                client.admin.command("ping")  # force connection check
                self._client = client
                logger.info(f"Connected to MongoDB on attempt {attempt}")
                break
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                last_error = e
                logger.warning(
                    f"MongoDB connection attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {retry_delay_seconds}s..."
                )
                time.sleep(retry_delay_seconds)

        if self._client is None:
            raise ConnectionError(
                f"Could not connect to MongoDB after {max_retries} attempts"
            ) from last_error

        self._db = self._client[db_name]
        self.ingested_files = self._db["ingested_files"]
        self.chunks = self._db["chunks"]

        # Ensure fast lookups on the field we dedup against
        self.ingested_files.create_index("file_hash", unique=True)

    # --- Dedup ---------------------------------------------------------

    def is_file_ingested(self, file_hash: str) -> bool:
        return self.ingested_files.find_one({"file_hash": file_hash}) is not None

    def mark_file_ingested(
        self,
        file_hash: str,
        filename: str,
        mime_type: str,
        num_chunks: int,
    ) -> None:
        """Write the tracking record. Call this LAST, only after all chunks
        for the file have been successfully embedded and stored — writing it
        earlier risks marking a partially-ingested file as complete if the
        process fails mid-batch."""
        self.ingested_files.insert_one(
            {
                "file_hash": file_hash,
                "filename": filename,
                "mime_type": mime_type,
                "num_chunks": num_chunks,
                "ingested_at": datetime.now(timezone.utc),
            }
        )

    # --- Chunk storage ---------------------------------------------------

    def insert_chunks(self, chunk_docs: list[dict]) -> None:
        """Bulk-insert chunk documents. Each dict should include at least:
        _id (UUID string), text, file_hash, filename, mime_type, chunk_index,
        created_at."""
        if not chunk_docs:
            return
        self.chunks.insert_many(chunk_docs)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunk documents by their UUIDs (e.g. after a Chroma similarity
        search returns matching IDs)."""
        return list(self.chunks.find({"_id": {"$in": chunk_ids}}))