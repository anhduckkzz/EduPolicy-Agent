"""Retrieval augmented generation (RAG) tool backed by Milvus."""

from __future__ import annotations

import logging
from typing import List, Tuple

from ...config import settings
from ...db.milvus_client import MilvusVectorStore
from ...utils import chunk_text, embed_texts, load_pdf_text

LOGGER = logging.getLogger(__name__)


class RAGTool:
    """Encapsulates RAG ingestion and retrieval logic."""

    def __init__(self) -> None:
        self.vector_store: MilvusVectorStore | None = None
        try:
            self.vector_store = MilvusVectorStore()
        except Exception:  # pragma: no cover - startup guard
            LOGGER.exception(
                "Unable to initialise Milvus vector store; RAG tool will run in degraded mode."
            )
            return
        if self.vector_store.is_empty:
            LOGGER.info("Milvus collection empty; starting ingestion pipeline")
            self._ingest_corpus()

    # ------------------------------------------------------------------
    def _ingest_corpus(self) -> None:
        if not self.vector_store:
            LOGGER.warning("Skipping ingestion because no vector store is available.")
            return
        pdf_path = settings.data_dir / "regulations.pdf"
        if not pdf_path.exists():
            LOGGER.warning("Regulations PDF not found at %s. Skipping ingestion.", pdf_path)
            return
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            LOGGER.warning("No text chunks produced from %s", pdf_path)
            return
        embeddings = embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add_embeddings(
            embeddings=embeddings,
            chunks=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        LOGGER.info("Ingestion complete: %s chunks", len(chunks))

    # ------------------------------------------------------------------
    def query_rag(self, query: str, *, top_k: int | None = None) -> Tuple[str, List[str]]:
        """Perform semantic search and return concatenated context."""

        if not self.vector_store:
            return (
                "Chuc nang RAG tam thoi khong kha dung vi khong ket noi duoc toi Milvus.",
                [],
            )
        top_k = top_k or settings.top_k
        embedding = embed_texts([query])[0]
        documents = self.vector_store.query(embedding, top_k=top_k)
        if not documents:
            return "Khong tim thay thong tin phu hop trong co so quy dinh.", []
        snippets = [doc.text for doc in documents]
        combined = "\n\n".join(snippets)
        return combined, snippets
