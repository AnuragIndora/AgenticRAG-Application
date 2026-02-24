from __future__ import annotations

import logging
from pathlib import Path
import hashlib
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import SimpleDirectoryReader, Document
from settings import get_settings
from models import ChunkRecord
from text_preprocess import clean_text, tokenize_words, overlap_chunks
from milvus_client import MilvusVectorStore
from ollama_client import OllamaClient

logger = logging.getLogger("IngestionPipeline")
logging.basicConfig(level=logging.INFO)


class DocumentLoader:
    """Loads raw documents from disk using LlamaIndex's SimpleDirectoryReader."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}

    def load_data(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info(f"Loading file: {file_path}")
        return SimpleDirectoryReader(input_files=[file_path]).load_data()


class TextCleaner:
    """Thin wrapper around the text_preprocess module for consistency."""

    def clean(self, text: str) -> str:
        return clean_text(text=text)


class Chunker:
    """Splits document text into overlapping token windows and wraps them as ChunkRecords."""

    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size_tokens
        self.chunk_overlap = settings.chunk_overlap_tokens

    def chunk_records(self, records: List[dict], doc_id: str, file_name: str) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        chunk_counter = 0

        for rec in records:
            text = rec.get("text", "").strip()
            if not text:
                continue

            tokens = tokenize_words(text)

            for token_slice in overlap_chunks(tokens, self.chunk_size, self.chunk_overlap):
                chunk_text = " ".join(token_slice).strip()
                if not chunk_text:
                    continue

                chunk_counter += 1
                chunks.append(
                    ChunkRecord(
                        text=chunk_text,
                        doc_id=doc_id,
                        file_name=file_name,
                        file_type=rec.get("file_type", None),
                        page_number=int(rec.get("page_number", 0)),
                        chunk_id=f"{doc_id}-chunk-{chunk_counter}",
                        chunk_type=str(rec.get("chunk_type", "text")),
                    )
                )

        return chunks


class IngestionPipeline:
    """
    End-to-end pipeline: load → clean → chunk → embed → insert into Milvus.
    Uses a thread pool to parallelize embedding calls for faster throughput.
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = Chunker()
        self.milvus = MilvusVectorStore()
        self.ollama_client = OllamaClient()
        self.setting = get_settings()

    def process_file(self, file_path: str) -> List[ChunkRecord]:
        """Load, clean, and chunk a single file. Returns a list of ChunkRecord objects."""
        path = Path(file_path)
        file_name = path.name
        raw_docs = self.loader.load_data(file_path)

        # Clean each page/section and keep the metadata from LlamaIndex
        normalized_records = []
        for doc in raw_docs:
            text = self.cleaner.clean(doc.text)
            if text:
                record = dict(doc.metadata)
                record["text"] = text
                normalized_records.append(record)

        # Generate a stable doc ID using the first 12 chars of the file's SHA-1 hash
        file_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        doc_id = f"{path.stem}-{file_hash}"

        chunks = self.chunker.chunk_records(normalized_records, doc_id=doc_id, file_name=file_name)
        logger.info(f"Processed file '{file_name}': {len(chunks)} chunks generated")
        return chunks

    def ingest_to_milvus(self, chunks: List[ChunkRecord], batch_size: int = 1000, max_workers: int = 8):
        """
        Embed all chunks in parallel and insert them into Milvus in batches.
        Raises immediately if any embedding call fails — partial ingestion is avoided.
        """
        total_chunks = len(chunks)
        logger.info(f"Starting ingestion of {total_chunks} chunks to Milvus")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = [None] * len(batch)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map each future back to its position so we can preserve order
                future_to_index = {
                    executor.submit(self.ollama_client.embed, chunk.text): idx
                    for idx, chunk in enumerate(batch)
                }

                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        emb = future.result()
                        if len(emb) != self.setting.embedding_dim:
                            raise ValueError(
                                f"Embedding length mismatch for chunk {batch[idx].chunk_id}: "
                                f"{len(emb)} != {self.setting.embedding_dim}"
                            )
                        embeddings[idx] = emb
                    except Exception as e:
                        logger.error(f"Failed to embed chunk {batch[idx].chunk_id}: {e}")
                        raise

            self.milvus.insert(batch, embeddings)
            logger.info(f"Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")

        logger.info(f"Ingestion complete. Total entities: {self.milvus.collection.num_entities}")
        print("Total entities:", self.milvus.collection.num_entities)