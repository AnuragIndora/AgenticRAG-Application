from __future__ import annotations

import logging
from typing import Iterable

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

import settings
from models import ChunkRecord, RetrievalResult
from exceptions import MilvusConnectionError

logger = logging.getLogger("MilvusVectorStore")


class MilvusVectorStore:
    def __init__(self):
        self.settings = settings.get_settings()
        self.collection_name = self.settings.milvus_collection_name
        self._collection: Collection | None = None
        self.connect()
        self.ensure_collection()

    def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
                user=self.settings.milvus_username,
                password=self.settings.milvus_password,
            )
        except Exception as exc:
            raise MilvusConnectionError(f"Failed connecting to Milvus: {exc}") from exc

    def ensure_collection(self) -> None:
        """Ensure the collection exists and is loaded."""
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=256),    
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=256),
            ]
            schema = CollectionSchema(fields=fields, description="Agentic RAG chunks")
            collection = Collection(name=self.collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            logger.info("Created Milvus collection: %s", self.collection_name)

        self._collection = Collection(self.collection_name)
        self._collection.load()

    @property
    def collection(self) -> Collection:
        """Get the collection, ensuring it's initialized."""
        if self._collection is None:
            self.ensure_collection()
        assert self._collection is not None
        return self._collection

    def insert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert chunks and embeddings into the collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        if not chunks:
            return

        data = [
            embeddings,
            [c.text for c in chunks],
            [c.doc_id for c in chunks],
            [c.file_name for c in chunks],
            [c.file_type for c in chunks],
            [c.page_number for c in chunks],
            [c.chunk_id for c in chunks],
            [c.chunk_type for c in chunks],
        ]

        self.collection.insert(data)
        self.collection.flush()

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Search the collection for the top_k most similar embeddings."""
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        output_fields = ["text", "doc_id", "file_name", "page_number", "chunk_id", "chunk_type"]

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        formatted: list[RetrievalResult] = []
        if not results:
            return formatted

        for hit in results[0]:
            e = hit.entity
            formatted.append(
                RetrievalResult(
                    id=int(hit.id),
                    text=getattr(e, "text", ""),
                    doc_id=getattr(e, "doc_id", ""),
                    file_name=getattr(e, "file_name", ""),
                    page_number=int(getattr(e, "page_number", 0)),
                    chunk_id=getattr(e, "chunk_id", ""),
                    chunk_type=getattr(e, "chunk_type", "text"),
                    score=float(hit.score),
                )
            )

        return formatted

    @staticmethod
    def keyword_filter(candidates: Iterable[RetrievalResult], query: str) -> list[RetrievalResult]:
        """Rerank search results based on keyword overlap."""
        query_terms = {t.lower() for t in query.split() if len(t) > 2}
        reranked = []

        for item in candidates:
            text_terms = {t.lower() for t in item.text.split()}
            overlap = len(query_terms.intersection(text_terms))
            hybrid_score = (item.score * 0.8) + (overlap * 0.2)
            reranked.append((hybrid_score, item))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in reranked]
    

    def delete_collection(self):
        """
        Deletes the entire Milvus collection safely.
        WARNING: All ingested vectors will be lost!
        """
        if self._collection:
            collection_name = self._collection.name
            self._collection.release()
            from pymilvus import utility
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            self._collection = None