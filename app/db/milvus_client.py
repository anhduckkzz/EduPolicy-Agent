"""Milvus client wrapper used by the RAG tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections, utility

from ..config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class MilvusDocument:
    """Structure representing a document stored in Milvus."""

    text: str
    metadata: dict


class MilvusVectorStore:
    """Thin wrapper around Milvus for storing and querying embeddings."""

    def __init__(self) -> None:
        try:
            connections.connect(alias="default", uri=settings.milvus_uri)
        except MilvusException as exc:  # pragma: no cover - defensive guard
            LOGGER.error("Unable to connect to Milvus at %s: %s", settings.milvus_uri, exc)
            raise
        self.collection_name = settings.milvus_collection
        if not utility.has_collection(self.collection_name):
            LOGGER.info("Creating Milvus collection %s", self.collection_name)
            self._create_collection()
        self.collection = Collection(self.collection_name)
        if not self.collection.has_index():
            self._create_index()
        else:
            self.collection.load()
        if not self.collection.is_empty:
            LOGGER.debug("Milvus collection %s already populated", self.collection_name)

    # ------------------------------------------------------------------
    def _create_collection(self) -> None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="University regulation embeddings")
        Collection(name=self.collection_name, schema=schema)

    def _create_index(self) -> None:
        LOGGER.info("Creating HNSW index on collection %s", self.collection_name)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    # ------------------------------------------------------------------
    def add_embeddings(self, embeddings: Sequence[Sequence[float]], chunks: Sequence[str], metadatas: Sequence[dict]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")
        LOGGER.info("Inserting %s vectors into Milvus", len(embeddings))
        try:
            self.collection.insert([
                list(embeddings),
                list(chunks),
                list(metadatas),
            ])
            self.collection.flush()
            self.collection.load()
        except MilvusException as exc:
            LOGGER.error("Failed to insert into Milvus: %s", exc)
            raise

    def query(self, embedding: Sequence[float], top_k: int) -> List[MilvusDocument]:
        try:
            results = self.collection.search(
                data=[list(embedding)],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 32}},
                limit=top_k,
                output_fields=["text", "metadata"],
            )
        except MilvusException as exc:
            LOGGER.error("Milvus search failed: %s", exc)
            raise
        documents: List[MilvusDocument] = []
        for hits in results:
            for hit in hits:
                documents.append(MilvusDocument(text=hit.entity.get("text"), metadata=hit.entity.get("metadata", {})))
        return documents

    @property
    def is_empty(self) -> bool:
        return self.collection.is_empty
