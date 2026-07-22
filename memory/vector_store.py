"""ChromaDB-backed long-term semantic memory for EnergyForge Agent.

Stores two knowledge classes in one collection:

* **equipment manuals** — operating envelopes, alarm thresholds, service notes
* **historical RCAs** — root-cause analyses of past incidents

Agents (diagnostics, reporting) and the orchestrator retrieve semantically
similar documents to ground their reasoning — e.g. "vibration at 3× normal
with bearing temperature rising 2 °C/hr" retrieves RCA-WT-001.

Implementation notes
--------------------
* Uses ``chromadb.AsyncHttpClient`` against the ``chroma`` compose service.
* Embeddings are computed client-side with Chroma's default ONNX model
  (all-MiniLM-L6-v2 via onnxruntime). The ~23 MB model downloads once on
  first use and is cached under ``~/.cache``.
* All operations raise :class:`~exceptions.VectorStoreError` — the EnergyForge
  hierarchy, never raw chroma exceptions.
* ``reset()`` is intentionally dev/test-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings, get_settings
from exceptions import VectorStoreError
from logging_config import get_logger

if TYPE_CHECKING:  # heavy generics only needed for annotations
    from chromadb.api.models.AsyncCollection import AsyncCollection
    from chromadb.api.types import EmbeddingFunction

logger = get_logger(__name__)

MetadataValue = str | int | float | bool


class DocumentRecord(BaseModel):
    """One document to store in the vector collection."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,63}$")
    text: str = Field(min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """One retrieval result from the collection."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    distance: float = Field(ge=0.0)

    @property
    def similarity(self) -> float:
        """Cosine-space similarity derived from distance, clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.distance))


class VectorStore:
    """Async wrapper around a single Chroma collection.

    Args:
        settings: Application settings (defaults to the global singleton).
        collection_name: Override for the configured collection name —
            tests use this for isolation.
        embedding_function: Custom chroma embedding function; when ``None``
            the default ONNX all-MiniLM-L6-v2 model is used.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        collection_name: str | None = None,
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._collection_name = collection_name or self._settings.chroma_collection
        self._ef: EmbeddingFunction | None = embedding_function
        self._client: chromadb.AsyncHttpClient | None = None
        self._collection: AsyncCollection | None = None

    # ── internals ─────────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.AsyncHttpClient:
        if self._client is None:
            self._client = chromadb.AsyncHttpClient(
                host=self._settings.chroma_host,
                port=self._settings.chroma_port,
                ssl=self._settings.chroma_ssl,
            )
        return self._client

    async def _get_collection(self) -> AsyncCollection:
        if self._collection is None:
            try:
                self._collection = await self._get_client().get_or_create_collection(
                    name=self._collection_name,
                    embedding_function=self._ef or embedding_functions.DefaultEmbeddingFunction(),
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # broad: chroma raises several exception types
                raise VectorStoreError(
                    "failed to open chroma collection",
                    context={
                        "collection": self._collection_name,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:300],
                    },
                ) from exc
        return self._collection

    # ── public API ────────────────────────────────────────────────────────

    async def add_documents(self, records: Sequence[DocumentRecord]) -> int:
        """Upsert documents (idempotent — safe to re-seed). Returns count added."""
        if not records:
            return 0
        collection = await self._get_collection()
        try:
            await collection.upsert(
                ids=[r.id for r in records],
                documents=[r.text for r in records],
                metadatas=[r.metadata for r in records],
            )
        except Exception as exc:
            raise VectorStoreError(
                "failed to upsert documents",
                context={"count": len(records), "error": type(exc).__name__, "detail": str(exc)[:300]},
            ) from exc
        logger.info("vector_store.upserted", count=len(records), collection=self._collection_name)
        return len(records)

    async def query(
        self,
        text: str,
        *,
        n_results: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """Semantic search. Filter optionally on exact metadata equality."""
        collection = await self._get_collection()
        try:
            result = await collection.query(
                query_texts=[text],
                n_results=n_results,
                where=where,
            )
        except Exception as exc:
            raise VectorStoreError(
                "chroma query failed",
                context={"error": type(exc).__name__, "detail": str(exc)[:300]},
            ) from exc

        hits: list[SearchHit] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        for doc_id, doc, meta, dist in zip(ids, documents[0], metadatas[0], distances[0], strict=True):
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=doc or "",
                    metadata=dict(meta or {}),
                    distance=float(dist),
                )
            )
        logger.info("vector_store.query", results=len(hits), collection=self._collection_name)
        return hits

    async def count(self) -> int:
        """Number of documents in the collection."""
        collection = await self._get_collection()
        try:
            return await collection.count()
        except Exception as exc:
            raise VectorStoreError(
                "chroma count failed",
                context={"error": type(exc).__name__, "detail": str(exc)[:300]},
            ) from exc

    async def reset(self) -> None:
        """Delete and recreate the collection. **Dev/test only.**"""
        client = self._get_client()
        try:
            await client.delete_collection(self._collection_name)
        except Exception:  # collection may not exist — fine either way
            logger.debug("vector_store.reset_delete_skipped", collection=self._collection_name)
        self._collection = None
        logger.info("vector_store.reset", collection=self._collection_name)

    async def seed_default_corpus(self) -> int:
        """Upsert the built-in equipment manuals + historical RCAs (idempotent)."""
        from memory.knowledge_corpus import HISTORICAL_RCAS, EQUIPMENT_MANUALS

        records = [*EQUIPMENT_MANUALS, *HISTORICAL_RCAS]
        added = await self.add_documents(records)
        logger.info("vector_store.corpus_seeded", records=added)
        return added
