
# ===== File: ./models.py =====

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# Supported high-level intent categories for the routing system
IntentType = Literal[
    "document_qa",
    "summarize",
    "structured_data",
    "comparison",
    "multi_hop",
]


class ChunkRecord(BaseModel):
    """A single text chunk produced by the ingestion pipeline."""
    text: str
    doc_id: str
    file_name: str
    file_type: Optional[str] = None
    page_number: int = 0
    chunk_id: str
    chunk_type: str = "text"


class RetrievalResult(BaseModel):
    """A chunk returned by a Milvus search, enriched with its similarity score."""
    id: int
    text: str
    doc_id: str
    file_name: str
    page_number: int
    chunk_id: str
    chunk_type: str
    score: float


class QueryAnalysis(BaseModel):
    """Structured representation of a parsed user query before routing."""
    original_query: str
    intent: IntentType
    expanded_queries: list[str] = Field(default_factory=list)
    requires_sql: bool = False


class SQLExecutionResult(BaseModel):
    """Holds the raw output of a Postgres query."""
    query: str
    rows: list
    row_count: int


class ReasoningResponse(BaseModel):
    """Intermediate response model used before final output assembly."""
    answer: str
    confidence_score: float
    citations: list[str] = Field(default_factory=list)
    sql_query: Optional[str] = None


class AgentRunOutput(BaseModel):
    """Final output returned by any agent to the orchestrator or API layer."""
    answer: str
    intent: IntentType
    confidence_score: float
    reasoning_steps: list[str] = Field(default_factory=list)
    sources: list[RetrievalResult] = Field(default_factory=list)
    sql_query: Optional[str] = None


class IngestResponse(BaseModel):
    """Summary returned after a batch ingestion operation."""
    ingested_files: int
    ingested_chunks: int
    failed_files: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Incoming query payload for the /query API endpoint."""
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    """Response shape returned by the /query API endpoint."""
    answer: str
    intent: IntentType
    confidence_score: float
    sources: list[dict] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    sql_query: Optional[str] = None
# ===== File: ./logger.py =====

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up a single stdout handler on the root logger.
    Calling this multiple times is safe — it bails out if handlers already exist.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured, don't add duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)

# ===== File: ./milvuous_client.py =====

from __future__ import annotations

import logging
from typing import Iterable

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

import settings
from models import ChunkRecord, RetrievalResult
from exceptions import MilvusConnectionError

logger = logging.getLogger("MilvusVectorStore")


class MilvusVectorStore:
    """Manages the Milvus collection lifecycle and exposes insert/search helpers."""

    def __init__(self):
        self.settings = settings.get_settings()
        self.collection_name = self.settings.milvus_collection_name
        self._collection: Collection | None = None
        self.connect()
        self.ensure_collection()

    def connect(self) -> None:
        """Open a connection to Milvus using credentials from settings."""
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
        """
        Create the collection and HNSW index if they don't exist yet, then load into memory.
        Idempotent — safe to call multiple times.
        """
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

            # HNSW gives a good balance of recall and speed; M=16, efConstruction=200 is a sensible default
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
        """Lazy accessor — ensures the collection is initialized before returning it."""
        if self._collection is None:
            self.ensure_collection()
        assert self._collection is not None
        return self._collection

    def insert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert a batch of chunks and their embeddings into the collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        if not chunks:
            return

        # Milvus expects a list-of-column-lists, ordered to match the schema fields
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
        self.collection.flush()  # make results immediately searchable

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Run an ANN search and return the top_k most similar chunks."""
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
        """
        Re-rank candidates by a hybrid score: 80% vector similarity + 20% keyword overlap.
        Filters out very short query words (len <= 2) to avoid noise from stop words.
        """
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
        Drop the entire Milvus collection and clear the local reference.
        WARNING: This permanently removes all ingested vectors.
        """
        if self._collection:
            collection_name = self._collection.name
            self._collection.release()
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            self._collection = None
# ===== File: ./streamlit-app.py =====

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

# -- Session state ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Page title ---------------------------------------------------------------
st.title("🤖 Agentic RAG Assistant")

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📂 Upload File")

    uploaded_file = st.file_uploader(
        "Upload document or CSV",
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"],
    )

    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files)
        if resp.status_code == 200:
            st.success(resp.json()["message"])
        else:
            st.error(resp.text)

    st.divider()

    if st.button("⚠ Reset Database"):
        resp = requests.delete(f"{BACKEND_URL}/reset")
        if resp.status_code == 200:
            st.success("Database cleared")
        else:
            st.error(resp.text)

# -- Chat history -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -- Chat input ---------------------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(f"{BACKEND_URL}/query", json={"query": user_input})

            if resp.status_code == 200:
                result = resp.json()
                answer = result["answer"]
                st.markdown(answer)

                # Expandable debug panel — hidden by default to keep the UI clean
                with st.expander("🔍 Details"):
                    st.write("**Intent:**", result.get("intent"))
                    st.write("**Confidence:**", result.get("confidence_score"))
                    st.write("**Reasoning Steps:**", result.get("reasoning_steps"))

                    if result.get("sql_query"):
                        st.code(result["sql_query"], language="sql")

                    if result.get("sources"):
                        st.write("**Sources:**")
                        for s in result["sources"]:
                            st.write(
                                f"- {s.get('file_name')} | page {s.get('page_number')} | score {s.get('score')}"
                            )

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(resp.text)
# ===== File: ./main.py =====

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil
import logging
import pandas as pd

from real_time_agent import RealTimeAgentAssistant

app = FastAPI(title="Agentic RAG System API")
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = Path("uploaded_docs")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# The assistant starts a background thread to watch UPLOAD_FOLDER for new files
assistant = RealTimeAgentAssistant(watch_folder=str(UPLOAD_FOLDER))


class QueryRequest(BaseModel):
    query: str
    task_type: str | None = None


@app.post("/query")
def query_agent(request: QueryRequest):
    try:
        result = assistant.query(request.query, task_type=request.task_type)
        return {
            "answer": result.answer,
            "intent": result.intent,
            "confidence_score": result.confidence_score,
            "sources": [s.model_dump() for s in result.sources],
            "reasoning_steps": result.reasoning_steps,
            "sql_query": result.sql_query,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Return the list of files that have already been ingested."""
    return {"ingested_files": list(assistant.ingested_files)}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".pptx", ".txt", ".md", ".xlsx", ".csv"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dest = UPLOAD_FOLDER / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Structured files (CSV / Excel) go straight into Postgres, not Milvus
    if suffix in {".xlsx", ".csv"}:
        from postgres_client import PostgresClient

        db = PostgresClient()
        table_name = file.filename.split(".")[0].lower()

        if suffix == ".csv":
            df = pd.read_csv(dest)
        else:
            df = pd.read_excel(dest)

        db.create_table_from_dataframe(table_name, df)
        dest.unlink()  # remove the temp file — the data now lives in Postgres

        return {"message": f"Structured data loaded into table '{table_name}'"}

    # Unstructured files are picked up automatically by the folder watcher
    return {"message": "File uploaded for vector ingestion."}


@app.delete("/reset")
def reset_database(delete_sql: bool = False):
    """
    Wipe all ingested data from Milvus.
    Pass `?delete_sql=true` to also drop all Postgres tables.
    """
    try:
        assistant.orchestrator.ingestion_pipeline.milvus.delete_collection()
        assistant.ingested_files.clear()

        if delete_sql:
            assistant.orchestrator.structured_agent.sql_agent.delete_all_tables()

        return {"message": "Database and ingested files cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# ===== File: ./sql_agent.py =====

from ollama_client import OllamaClient
from postgres_client import PostgresClient


class SQLAgent:
    """Translates natural language questions into safe SELECT queries and executes them."""

    def __init__(self):
        self.llm = OllamaClient()
        self.db = PostgresClient()

    def generate_sql(self, user_query: str) -> str:
        """
        Ask the LLM to produce a safe SQL query, then validate it before returning.
        Only SELECT and WITH-SELECT (CTE) queries are allowed through.
        Also verifies that at least one referenced table actually exists in the DB.
        """
        prompt = f"""
You are a PostgreSQL expert.
Generate ONLY a valid SQL query for this question.
- MUST be a SELECT query or WITH/SELECT (CTE).
- Never use DELETE, UPDATE, INSERT, or DROP.
- Only reference tables in the public schema.
- Return query only, no explanations.

Question:
{user_query}
"""
        response = self.llm.generate([{"role": "user", "content": prompt}])
        sql_query = response.strip()

        # Strip Markdown code fences if the LLM wrapped the query in them
        if sql_query.startswith("```"):
            sql_query = "\n".join(sql_query.split("\n")[1:-1]).strip()

        # Reject anything that isn't a read-only query
        sql_lower = sql_query.lower()
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            raise ValueError(f"Only SELECT queries are allowed. Generated SQL: {sql_query}")

        # Cross-check referenced identifiers against actual tables in the DB
        existing_tables_rows = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )
        existing_tables = [t[0].lower() for t in existing_tables_rows]

        # Crude but fast: tokenize the SQL and see if any token matches a real table name
        referenced_tables = [
            word.strip('"') for word in sql_lower.replace(",", " ").split() if word.isidentifier()
        ]
        if not any(table in existing_tables for table in referenced_tables):
            raise ValueError(f"No referenced table exists in DB. Generated SQL: {sql_query}")

        return sql_query

    def handle_query(self, query: str) -> str:
        """Generate SQL, execute it, and use the LLM to summarize the result in plain English."""
        sql_query = self.generate_sql(query)
        result = self.db.execute_query(sql_query)

        formatted_prompt = f"""
User Question: {query}

SQL Query Used:
{sql_query}

SQL Result:
{result}

Provide a concise and clear answer to the user.
"""
        return self.llm.generate([{"role": "user", "content": formatted_prompt}])

    def delete_all_tables(self):
        """
        Drop every table in the public schema.
        WARNING: This permanently deletes all structured data!
        """
        tables = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )
        for (table_name,) in tables:
            self.db.execute_query(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
# ===== File: ./text_preprocess.py =====

import re
from typing import Iterable

# Pre-compiled patterns — only compiled once at import time for performance
_WHITESPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # control chars except \t, \n, \r


def clean_text(text: str) -> str:
    """
    Normalize raw document text:
    - Strip control characters
    - Remove soft-hyphen (\u00ad) and BOM (\ufeff) characters
    - Collapse all whitespace runs into a single space
    """
    text = _NOISE_RE.sub(" ", text)
    text = text.replace("\u00ad", "")   # soft hyphen — invisible but breaks tokenization
    text = text.replace("\ufeff", "")   # byte-order mark sometimes appears at file start
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize_words(text: str) -> list[str]:
    """
    Split text into word tokens + standalone punctuation.
    Unicode word characters are preserved (handles non-ASCII properly).
    """
    return re.findall(r"\b\w+\b|[^\w\s]", text, flags=re.UNICODE)


def token_count(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(tokenize_words(text))


def overlap_chunks(tokens: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    """
    Yield sliding windows of `size` tokens with `overlap` tokens carried over
    from the previous window. The final window may be shorter than `size`.

    Raises ValueError if size <= 0 or overlap >= size.
    """
    if size <= 0:
        raise ValueError("Chunk size must be > 0")
    if overlap >= size:
        raise ValueError("Chunk overlap must be < chunk size")

    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start = end - overlap

# ===== File: ./exceptions.py =====

class AgenticRAGError(Exception):
    """Base class for all application-specific exceptions."""


class UnsupportedFileFormatError(AgenticRAGError):
    """Raised when an unsupported file type is passed to the ingestion pipeline."""


class MilvusConnectionError(AgenticRAGError):
    """Raised when the app can't connect to or communicate with Milvus."""


class EmptyRetrievalError(AgenticRAGError):
    """Raised when a vector search returns no usable results."""


class OllamaTimeoutError(AgenticRAGError):
    """Raised when a request to the Ollama API times out or returns an empty response."""

# ===== File: ./ingestion_pipeline.py =====

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
# ===== File: ./intent_router_agent.py =====

from ollama_client import OllamaClient


class IntentRouterAgent:
    """
    Classifies a user query into one of three intents using the LLM.
    Falls back to a simple keyword check if the LLM call fails.
    """

    INTENTS = {"structured_data", "document_qa", "summarize"}

    def __init__(self):
        self.llm = OllamaClient()

    def detect_intent(self, query: str) -> str:
        """
        Returns one of: 'structured_data', 'document_qa', 'summarize'.
        """
        prompt = f"""
Classify the user query into one category:
1. structured_data
2. document_qa
3. summarize

Return only one label in lowercase.

Query: {query}
"""
        try:
            result = self.llm.generate([{"role": "user", "content": prompt}])
            return result.strip().lower()
        except Exception:
            # LLM unavailable — fall back to keyword heuristic
            lower_q = query.lower()
            if any(kw in lower_q for kw in ["select", "from", "where", "join"]):
                return "structured_data"
            return "document_qa"
# ===== File: ./settings.py =====

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    App-wide configuration loaded from the .env file.
    Extra keys in .env are silently ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # FastAPI server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Milvus — vector store for unstructured document chunks
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection_name: str = Field(default="agentic_rag_chunks")
    milvus_username: str | None = Field(default=None)
    milvus_password: str | None = Field(default=None)

    embedding_dim: int = Field(default=768)          # must match the embedding model output
    chunk_size_tokens: int = Field(default=1000)
    chunk_overlap_tokens: int = Field(default=150)

    retrieval_top_k: int = Field(default=8)
    retrieval_similarity_threshold: float = Field(default=0.35)
    max_context_tokens: int = Field(default=3500)

    # PostgreSQL — structured data store
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_database: str = Field(default="agentic_rag")
    pg_user: str = Field(default="raguser")
    pg_password: str = Field(default="ragpass")

    sql_max_rows: int = Field(default=200)
    sql_timeout_seconds: int = Field(default=30)

    # Ollama — local LLM and embedding server
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    llm_model: str = Field(default="gemma3n:latest")
    embedding_model: str = Field(default="embeddinggemma:latest")

    request_timeout_seconds: int = Field(default=90)
    max_retries: int = Field(default=2)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance (parsed once per process)."""
    return Settings()
# ===== File: ./real_time_agent.py =====

import time
from pathlib import Path
import threading
import logging

from agents_pipeline import AgentsOrchestrator

logger = logging.getLogger("RealTimeAgent")
logging.basicConfig(level=logging.INFO)


class RealTimeAgentAssistant:
    """
    Watches a folder for new files and ingests them automatically in the background.
    Also provides a `query()` interface to the agent orchestrator.
    """

    def __init__(self, watch_folder: str, poll_interval: int = 5):
        self.watch_folder = Path(watch_folder)
        if not self.watch_folder.is_dir():
            raise ValueError(f"Folder does not exist: {watch_folder}")

        self.poll_interval = poll_interval  # seconds between directory scans
        self.orchestrator = AgentsOrchestrator()

        # Snapshot existing files so we don't re-ingest them on startup
        self.ingested_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())

        # Run the watcher as a daemon so it dies when the main process exits
        self._watcher_thread = threading.Thread(target=self._watch_folder, daemon=True)
        self._watcher_thread.start()

    def _watch_folder(self):
        """Background loop: detect new files and trigger ingestion."""
        logger.info(f"Started watching folder: {self.watch_folder}")
        while True:
            try:
                current_files = set(f.name for f in self.watch_folder.iterdir() if f.is_file())
                new_files = current_files - self.ingested_files

                for file_name in new_files:
                    file_path = self.watch_folder / file_name
                    logger.info(f"New file detected: {file_name} — ingesting...")
                    try:
                        self.orchestrator.ingest_file(str(file_path))
                        self.ingested_files.add(file_name)
                        logger.info(f"Successfully ingested: {file_name}")
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_name}: {e}")

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in folder watcher: {e}")
                time.sleep(self.poll_interval)

    def query(self, text: str, task_type=None) -> str:
        """Forward a query to the orchestrator and return the result."""
        return self.orchestrator.handle_query(text, task_type)
# ===== File: ./rag_pipeline.py =====

from typing import List

from milvus_client import MilvusVectorStore
from ollama_client import OllamaClient
from models import RetrievalResult


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Fetches a wide candidate pool from Milvus, re-ranks by keyword overlap, and
    returns only the top-k most relevant chunks.
    """

    def __init__(self):
        self.milvus = MilvusVectorStore()
        self.ollama = OllamaClient()

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[RetrievalResult]:
        """
        Two-stage retrieval:
        1. ANN search for `candidate_k` approximate neighbors.
        2. Re-rank by keyword overlap and keep only the top `top_k`.
        """
        query_emb = self.ollama.embed(query)
        candidate_results = self.milvus.search(query_emb, top_k=candidate_k)
        reranked_results = self.milvus.keyword_filter(candidate_results, query)
        return reranked_results[:top_k]

    def generate_answer(self, query: str, top_k: int = 5, candidate_k: int = 20) -> str:
        """Retrieve relevant context and generate a grounded answer using the LLM."""
        retrieved_chunks = self.retrieve(query, top_k=top_k, candidate_k=candidate_k)
        context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
        prompt = (
            f"Answer the following question using the context below:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        return self.ollama.generate([{"role": "user", "content": prompt}])
# ===== File: ./agents_pipeline.py =====

import logging
import time
from typing import Tuple, List

from ingestion_pipeline import IngestionPipeline
from agents import DocumentAgent, SummarizationAgent, StructuredDataAgent
from models import AgentRunOutput
from intent_router_agent import IntentRouterAgent

logger = logging.getLogger("AgentsPipeline")


class AgentsOrchestrator:
    """
    Central router that delegates incoming queries to the right agent:
    - StructuredDataAgent for SQL / database questions
    - DocumentAgent for general document Q&A
    - SummarizationAgent for summarization tasks
    """

    def __init__(self, metadata_ttl: int = 300):
        self.ingestion_pipeline = IngestionPipeline()
        self.router = IntentRouterAgent()

        self.document_agent = DocumentAgent()
        self.summary_agent = SummarizationAgent()
        self.structured_agent = StructuredDataAgent()

        # Cache for SQL table/column metadata so we don't query Postgres on every request
        self._cached_tables: List[str] = []
        self._cached_columns: List[str] = []
        self._metadata_last_refresh: float = 0
        self._metadata_ttl = metadata_ttl  # seconds before the cache is considered stale

        # Pre-warm the cache so the first request isn't slow
        self.get_sql_metadata()

    def get_sql_metadata(self) -> Tuple[List[str], List[str]]:
        """
        Returns (tables, columns) from Postgres, refreshing the cache only when TTL has expired.
        """
        now = time.time()

        # Serve from cache if it's still fresh
        if self._cached_tables and now - self._metadata_last_refresh < self._metadata_ttl:
            return self._cached_tables, self._cached_columns

        try:
            logger.info("Refreshing SQL metadata cache")

            tables_rows = self.structured_agent.sql_agent.db.execute_query(
                "SELECT tablename FROM pg_tables WHERE schemaname='public';"
            )
            tables = [t[0] for t in tables_rows]

            # Collect all column names across every table
            columns = []
            for table in tables:
                cols = self.structured_agent.sql_agent.db.execute_query(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s;
                    """,
                    (table,),
                )
                columns.extend([c[0] for c in cols])

            self._cached_tables = tables
            self._cached_columns = columns
            self._metadata_last_refresh = now

            return tables, columns

        except Exception as e:
            logger.error(f"Failed to fetch SQL metadata: {e}")
            return [], []

    def looks_like_sql_query(self, query: str, tables: list, columns: list) -> bool:
        """
        Score-based heuristic to decide if a query is targeting structured data.
        Returns True when the score crosses the threshold (avoids false positives).
        """
        q = query.lower()
        score = 0

        sql_keywords = ["select", "from", "where", "group by", "order by", "join", "having"]

        if any(keyword in q for keyword in sql_keywords):
            score += 3  # generic SQL keyword hit

        if any(table.lower() in q for table in tables):
            score += 2  # the user mentioned a real table name

        if any(col.lower() in q for col in columns):
            score += 1  # the user mentioned a real column name

        # "select ... from ..." is a very strong structural signal
        if "select" in q and "from" in q:
            score += 5

        return score >= 3

    def ingest_file(self, file_path: str):
        """Load a document file, chunk it, embed it, and store in Milvus."""
        try:
            chunks = self.ingestion_pipeline.process_file(file_path)
            self.ingestion_pipeline.ingest_to_milvus(chunks)
            logger.info(f"Successfully ingested {file_path}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise

    def handle_query(self, query: str, task_type=None) -> AgentRunOutput:
        """
        Main entry point for all queries. Steps:
        1. Ask the LLM to classify the intent.
        2. Run a SQL heuristic as a sanity check / override.
        3. Route to the right agent.
        4. Fall back to DocumentAgent if the structured agent explodes.
        """
        start_time = time.time()

        try:
            # Step 1: LLM-based intent classification
            intent = self.router.detect_intent(query)
            logger.info(f"Detected intent: {intent}")

            # Step 2: Double-check with keyword heuristic (catches obvious SQL cases the LLM may miss)
            tables, columns = self.get_sql_metadata()
            sql_like = self.looks_like_sql_query(query, tables, columns)

            # Step 3: Decide the final routing target
            if intent == "structured_data":
                route = "structured_data"
            elif sql_like:
                route = "structured_data"
                logger.info("SQL heuristic override triggered")
            elif intent == "summarize":
                route = "summarize"
            else:
                route = "document_qa"

            logger.info(f"Routing decision: {route} | latency={time.time() - start_time:.2f}s")

            # Step 4: Execute the chosen agent
            if route == "structured_data":
                try:
                    return self.structured_agent.handle_query(query, task_type or route)
                except Exception:
                    # SQL agent failed (bad query, missing table, etc.) — fall back gracefully
                    logger.warning("Structured agent failed. Falling back to DocumentAgent.")
                    return self.document_agent.handle_query(query, task_type or "document_qa")

            if route == "summarize":
                return self.summary_agent.handle_query(query, task_type or route)

            return self.document_agent.handle_query(query, task_type or "document_qa")

        except Exception as e:
            logger.exception("Orchestrator failure")
            return AgentRunOutput(
                answer="System error occurred while processing the query.",
                intent="error",
                confidence_score=0.0,
                reasoning_steps=["Exception in orchestrator"],
                sources=[],
            )
# ===== File: ./agents.py =====

from typing import List

from models import AgentRunOutput, RetrievalResult
from rag_pipeline import RAGPipeline
from sql_agent import SQLAgent


class DocumentAgent:
    """Handles document Q&A by retrieving relevant chunks from Milvus and generating answers via LLM."""

    def __init__(self):
        self.rag = RAGPipeline()

    def handle_query(self, query: str, task_type=None) -> AgentRunOutput:
        # Retrieve the most relevant chunks for the query
        retrieved_chunks: List[RetrievalResult] = self.rag.retrieve(query)

        # Build context from the retrieved chunk texts
        context = "\n\n".join([chunk.text for chunk in retrieved_chunks])

        prompt = f"""
You are a precise AI assistant.

Use ONLY the provided context.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{query}

Provide a clear answer.
"""
        answer = self.rag.ollama.generate([{"role": "user", "content": prompt}])

        return AgentRunOutput(
            answer=answer,
            intent="document_qa",
            confidence_score=0.85,
            reasoning_steps=["Retrieved relevant document chunks", "Generated answer using context"],
            sources=retrieved_chunks,
        )


class SummarizationAgent:
    """Retrieves relevant document chunks and condenses them into a concise summary."""

    def __init__(self):
        self.rag = RAGPipeline()

    def handle_query(self, query: str, task_type=None) -> AgentRunOutput:
        retrieved_chunks = self.rag.retrieve(query)

        combined_text = "\n\n".join([c.text for c in retrieved_chunks])

        prompt = f"""
Summarize the following content concisely:

{combined_text}
"""
        summary = self.rag.ollama.generate([{"role": "user", "content": prompt}])

        return AgentRunOutput(
            answer=summary,
            intent="summarize",
            confidence_score=0.8,
            reasoning_steps=["Retrieved relevant chunks", "Generated summary"],
            sources=retrieved_chunks,
        )


class StructuredDataAgent:
    """Handles SQL-based queries against structured data stored in PostgreSQL."""

    def __init__(self):
        self.sql_agent = SQLAgent()

    def handle_query(self, query: str, task_type=None) -> AgentRunOutput:
        sql_query = self.sql_agent.generate_sql(query)

        # Block any non-SELECT queries as a safety measure
        if not sql_query.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")

        result = self.sql_agent.db.execute_query(sql_query)

        # Ask the LLM to turn the raw SQL result into a readable answer
        format_prompt = f"""
User Question: {query}

SQL Result:
{result}

Provide a clear explanation of the result.
"""
        answer = self.sql_agent.llm.generate([{"role": "user", "content": format_prompt}])

        return AgentRunOutput(
            answer=answer,
            intent="structured_data",
            confidence_score=0.9,
            reasoning_steps=["Generated SQL query", "Executed SQL", "Formatted result"],
            sql_query=sql_query,
        )
# ===== File: ./ollama_client.py =====

import logging
import requests

import settings
from exceptions import OllamaTimeoutError


class OllamaClient:
    """HTTP client for the local Ollama server — handles both embeddings and text generation."""

    def __init__(self) -> None:
        self.settings = settings.get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.timeout = self.settings.request_timeout_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def embed(self, text: str) -> list[float]:
        """Send text to the Ollama embeddings API and return the float vector."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.settings.embedding_model, "prompt": text}

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaTimeoutError("Embedding request timed out") from exc
        except requests.RequestException as exc:
            raise OllamaTimeoutError(f"Embedding request failed: {exc}") from exc

        data = response.json()
        embeddings = data.get("embedding")
        if not embeddings:
            raise OllamaTimeoutError("Ollama embedding API returned empty embedding")
        return embeddings

    def generate(self, messages: list[dict], temperature: float = 0.5) -> str:
        """
        Send a chat message list to Ollama and return the assistant's reply.
        `stream=False` ensures we get the full response in one shot.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaTimeoutError("Generate request timed out") from exc
        except requests.RequestException as exc:
            raise OllamaTimeoutError(f"Generate request failed: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaTimeoutError("Ollama chat API returned empty response")
        return content.strip()

# ===== File: ./postgres_client.py =====

import logging
import psycopg2
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_batch
from settings import get_settings

logger = logging.getLogger("PostgresClient")


class PostgresClient:
    """
    Handles Postgres connectivity and convenience methods for structured data.
    Auto-creates the target database on first run if it doesn't exist.
    """

    def __init__(self):
        self.settings = get_settings()
        self.dbname = self.settings.pg_database
        self._ensure_database_exists()
        self.conn = psycopg2.connect(
            host=self.settings.pg_host,
            port=self.settings.pg_port,
            dbname=self.dbname,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
        )
        self.conn.autocommit = True
        logger.info(f"Connected to database: {self.dbname}")

    def _ensure_database_exists(self):
        """Connect to the default 'postgres' DB and create our target DB if it's missing."""
        try:
            default_conn = psycopg2.connect(
                host=self.settings.pg_host,
                port=self.settings.pg_port,
                dbname="postgres",
                user=self.settings.pg_user,
                password=self.settings.pg_password,
            )
            default_conn.autocommit = True

            with default_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.dbname,))
                exists = cur.fetchone()

                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.dbname)))
                    logger.info(f"Database '{self.dbname}' created.")
                else:
                    logger.info(f"Database '{self.dbname}' already exists.")

            default_conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame, batch_size: int = 1000):
        """
        Create a table from a DataFrame (all columns as TEXT) and bulk-insert rows.
        Uses execute_batch for efficient batched inserts instead of row-by-row.
        Does nothing if the DataFrame is empty.
        """
        if df.empty:
            logger.warning(f"DataFrame is empty. Table '{table_name}' not created.")
            return

        # Build CREATE TABLE ... with one TEXT column per DataFrame column
        columns = [sql.SQL("{} TEXT").format(sql.Identifier(col)) for col in df.columns]
        create_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(columns),
        )
        with self.conn.cursor() as cur:
            cur.execute(create_query)
            logger.info(f"Table '{table_name}' created (if not exists).")

        insert_query = sql.SQL("INSERT INTO {} VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Placeholder() * len(df.columns)),
        )

        with self.conn.cursor() as cur:
            execute_batch(
                cur,
                insert_query.as_string(self.conn),
                df.values.tolist(),
                page_size=batch_size,
            )
        logger.info(f"Table '{table_name}' populated with {len(df)} rows.")

    def execute_query(self, query: str, params: tuple = None):
        """Run an arbitrary SQL query. Returns rows for SELECT; empty list for others."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise