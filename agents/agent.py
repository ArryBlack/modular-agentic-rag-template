import hashlib
import base64
import io
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Callable

import magic
import zipfile

from google import genai
from google.genai import types as genai_types

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from services.mongo_service import MongoService
from services.chroma_service import ChromaService

logger = logging.getLogger(__name__)

# --- Service clients (initialized once at module import) -------------------

mongo_service = MongoService()
chroma_service = ChromaService()
genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # must match ChromaService's DEFAULT_EMBEDDING_DIMENSIONS
EMBEDDING_BATCH_SIZE = 20   # conservative default — check current API limits before raising

# --- Deterministic type detection ---------------------------------------

TEXT_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _normalize_mime(raw_bytes: bytes, sniffed: str, declared: Optional[str]) -> str:
    if sniffed == "application/zip":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                names = z.namelist()
                if any(n.startswith("word/") for n in names):
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except zipfile.BadZipFile:
            pass

    if sniffed.startswith("text/") and sniffed not in {"text/markdown", "text/html"}:
        return "text/plain"

    return sniffed or declared or "application/octet-stream"


def _detect_file(ctx: InvocationContext) -> Optional[tuple[bytes, str, str]]:
    user_content = ctx.user_content
    if not user_content or not user_content.parts:
        return None
    for part in user_content.parts:
        if part.inline_data and part.inline_data.data:
            raw = part.inline_data.data
            sniffed = magic.from_buffer(raw, mime=True)
            mime = _normalize_mime(raw, sniffed, part.inline_data.mime_type)
            filename = getattr(part.inline_data, "display_name", None) or "unknown"
            return raw, mime, filename
    return None


def _text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
    )


# --- Extraction registry ---------------------------------------------------

def _extract_text_plain(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="replace")


EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    "text/plain": _extract_text_plain,
}


# --- Chunking ---------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# --- Embedding ---------------------------------------------------------------

def _batch(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Batch-call gemini-embedding-001 with RETRIEVAL_DOCUMENT task type.
    Returns one embedding vector per input chunk, in order."""
    all_embeddings: list[list[float]] = []

    for batch in _batch(chunks, EMBEDDING_BATCH_SIZE):
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        for embedding_obj in response.embeddings:
            all_embeddings.append(embedding_obj.values)

    return all_embeddings


# --- Sub-agents -----------------------------------------------------------

class EmbeddingAgent(BaseAgent):
    """Embeds text_chunks from session state via gemini-embedding-001,
    writes vectors to Chroma and chunk text/metadata to Mongo, then marks
    the file as ingested (last, only on full success)."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        chunks: list[str] = ctx.session.state.get("text_chunks", [])
        file_hash = ctx.session.state.get("file_hash")
        filename = ctx.session.state.get("filename")
        mime_type = ctx.session.state.get("detected_mime_type")

        if not chunks:
            yield _text_event(self.name, "No text content was found to embed.")
            return

        logger.info(f"[EmbeddingAgent] embedding {len(chunks)} chunks for '{filename}'")

        try:
            embeddings = _embed_chunks(chunks)
        except Exception:
            logger.exception("[EmbeddingAgent] embedding call failed")
            yield _text_event(
                self.name,
                "Something went wrong while embedding this file. Please try again.",
            )
            return

        if len(embeddings) != len(chunks):
            logger.error(
                f"[EmbeddingAgent] mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
            yield _text_event(
                self.name,
                "Something went wrong while embedding this file. Please try again.",
            )
            return

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        now = datetime.now(timezone.utc)

        mongo_docs = [
            {
                "_id": chunk_id,
                "text": chunk_text,
                "file_hash": file_hash,
                "filename": filename,
                "mime_type": mime_type,
                "chunk_index": idx,
                "created_at": now,
            }
            for idx, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunks))
        ]
        chroma_metadatas = [
            {"file_hash": file_hash, "filename": filename, "chunk_index": idx}
            for idx in range(len(chunks))
        ]

        try:
            chroma_service.add_chunks(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=chroma_metadatas,
                documents=chunks,
            )
            mongo_service.insert_chunks(mongo_docs)
            # Written last, only after both stores succeed — avoids marking a
            # partially-ingested file as complete if something fails mid-way.
            mongo_service.mark_file_ingested(
                file_hash=file_hash,
                filename=filename,
                mime_type=mime_type,
                num_chunks=len(chunks),
            )
        except Exception:
            logger.exception("[EmbeddingAgent] storage write failed")
            yield _text_event(
                self.name,
                "Something went wrong while storing this file's embeddings. Please try again.",
            )
            return

        logger.info(f"[EmbeddingAgent] stored {len(chunks)} chunks for '{filename}'")

        yield _text_event(
            self.name,
            f"Done! '{filename}' was split into {len(chunks)} chunks and embedded successfully.",
        )


embedding_agent = EmbeddingAgent(name="EmbeddingAgent")

unsupported_file_agent = LlmAgent(
    name="UnsupportedFileAgent",
    model="gemini-2.5-flash",
    instruction=(
        "A file was uploaded whose detected mime type is exactly: {detected_mime_type}. "
        "Tell the user this specific file type isn't supported yet by this workflow, "
        "and end the conversation. Do not guess or alter the mime type — state it "
        "exactly as given."
    ),
)


# --- Extraction + chunking agent (deterministic, dispatches by mime type) --

class TextExtractionAgent(BaseAgent):
    embedding_agent: EmbeddingAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, embedding_agent: EmbeddingAgent):
        super().__init__(name=name, embedding_agent=embedding_agent, sub_agents=[embedding_agent])

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        mime_type = ctx.session.state.get("detected_mime_type")
        raw_b64 = ctx.session.state.get("file_bytes_b64")
        file_hash = ctx.session.state.get("file_hash")

        extractor = EXTRACTORS.get(mime_type)
        if extractor is None or raw_b64 is None:
            yield _text_event(self.name, "Something went wrong reading this file. Please try again.")
            return

        if mongo_service.is_file_ingested(file_hash):
            yield _text_event(
                self.name,
                "This file has already been processed. No changes were made.",
            )
            return

        raw_bytes = base64.b64decode(raw_b64)
        text = extractor(raw_bytes)
        chunks = _chunk_text(text)

        ctx.session.state["text_chunks"] = chunks
        ctx.session.state["num_chunks"] = len(chunks)

        logger.info(f"[TextExtractionAgent] extracted {len(text)} chars, {len(chunks)} chunks")

        async for event in self.embedding_agent.run_async(ctx):
            yield event


# --- Router: within text-based files, route to extraction if supported ----

class TextTypeRouterAgent(BaseAgent):
    extraction_agent: TextExtractionAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, extraction_agent: TextExtractionAgent):
        super().__init__(name=name, extraction_agent=extraction_agent, sub_agents=[extraction_agent])

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        mime_type = ctx.session.state.get("detected_mime_type")
        if mime_type in EXTRACTORS:
            async for event in self.extraction_agent.run_async(ctx):
                yield event
        else:
            yield _text_event(self.name, "Support for this file type will be added later.")
            return


# --- Top-level router: file present? text-based vs other -------------------

class FileRouterAgent(BaseAgent):
    text_type_router: TextTypeRouterAgent
    unsupported_agent: LlmAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, text_type_router: TextTypeRouterAgent, unsupported_agent: LlmAgent):
        super().__init__(
            name=name,
            text_type_router=text_type_router,
            unsupported_agent=unsupported_agent,
            sub_agents=[text_type_router, unsupported_agent],
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        detected = _detect_file(ctx)

        if detected is None:
            yield _text_event(self.name, "Please upload a file so I can process it.")
            return

        raw_bytes, mime_type, filename = detected
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        logger.info(f"[FileRouterAgent] sniffed mime_type = {mime_type!r}, file_hash = {file_hash[:12]}...")

        ctx.session.state["detected_mime_type"] = mime_type
        ctx.session.state["file_bytes_b64"] = base64.b64encode(raw_bytes).decode("ascii")
        ctx.session.state["file_hash"] = file_hash
        ctx.session.state["filename"] = filename

        if mime_type in TEXT_MIME_TYPES:
            async for event in self.text_type_router.run_async(ctx):
                yield event
        else:
            async for event in self.unsupported_agent.run_async(ctx):
                yield event


root_agent = FileRouterAgent(
    name="File_Router",
    text_type_router=TextTypeRouterAgent(
        name="Text_Type_Router",
        extraction_agent=TextExtractionAgent(name="Text_Extraction_Agent", embedding_agent=embedding_agent),
    ),
    unsupported_agent=unsupported_file_agent,
)