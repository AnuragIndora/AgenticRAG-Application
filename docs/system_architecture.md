# System Architecture — Agentic RAG System

## 1. Overview

This system is a **multi-agent Retrieval-Augmented Generation (RAG)** platform that handles queries over both **unstructured documents** (PDF, DOCX, PPTX, TXT, MD) and **structured tabular data** (CSV, XLSX → PostgreSQL). It uses a local Ollama LLM for both embeddings and text generation, Milvus as the vector store, and exposes both a FastAPI REST API and a Streamlit chat UI.

---

## 2. High-Level Component Map
![alt text](images/image3.png)

---

## 3. Query Routing Flow

Every incoming query goes through a **two-pass routing system** before an agent is invoked.

![alt text](images/image4.png)


### SQL Heuristic Scoring
| Signal | Points |
|---|---|
| SQL keyword found (`select`, `from`, `where`, `join`, etc.) | +3 |
| A real table name appears in the query | +2 |
| A real column name appears in the query | +1 |
| Both `select` AND `from` appear | +5 |
| **Threshold to trigger SQL route** | **≥ 3** |

> The cache TTL for table/column metadata is **300 seconds** (configurable via `metadata_ttl`).

---

## 4. Document Ingestion Pipeline

Files dropped into `uploaded_docs/` are picked up by a background daemon thread and pushed through the full ingestion pipeline.

![alt text](images/image5.png)


### Milvus Schema

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Auto-generated primary key |
| `embedding` | FLOAT_VECTOR | dim=768, HNSW COSINE index |
| `text` | VARCHAR(65535) | Raw chunk text |
| `doc_id` | VARCHAR(256) | `{stem}-{sha1[:12]}` |
| `file_name` | VARCHAR(512) | Original filename |
| `file_type` | VARCHAR(256) | e.g. `pdf`, `docx` |
| `page_number` | INT64 | Source page |
| `chunk_id` | VARCHAR(256) | `{doc_id}-chunk-N` |
| `chunk_type` | VARCHAR(256) | Default: `text` |

---

## 5. Retrieval Pipeline (RAG)

Retrieval uses a **two-stage** strategy: broad ANN search followed by keyword-based re-ranking.

![alt text](images/image6.png)


### Hybrid Re-Rank Formula

```
hybrid_score = (cosine_similarity × 0.8) + (keyword_overlap_count × 0.2)
```

> Short words (≤ 2 characters) are excluded from keyword overlap to avoid noise from stop words.

---

## 6. Structured Data Pipeline (SQL)

CSV and XLSX files are loaded directly into PostgreSQL. Queries against them go through the SQL agent.

![alt text](images/image7.png)

---

## 7. Class Responsibility Summary

![alt text](images/image8.png)

---

## 8. Data Models (Pydantic)

![alt text](images/image9.png)

---

## 9. Configuration Reference (`settings.py`)

All values are loaded from `.env` via `pydantic-settings`. The `get_settings()` function is cached with `@lru_cache` so the file is parsed only once per process.

| Group | Setting | Default | Notes |
|---|---|---|---|
| **API** | `api_host` | `0.0.0.0` | |
| | `api_port` | `8000` | |
| **Milvus** | `milvus_host` | `localhost` | `milvus` in Docker |
| | `milvus_port` | `19530` | |
| | `embedding_dim` | `768` | Must match embedding model output |
| | `chunk_size_tokens` | `1000` | |
| | `chunk_overlap_tokens` | `150` | |
| | `retrieval_top_k` | `8` | |
| **Postgres** | `pg_host` | `localhost` | `postgres` in Docker |
| | `pg_database` | `agentic_rag` | Created automatically if missing |
| **Ollama** | `ollama_base_url` | `http://127.0.0.1:11434` | `http://ollama:11434` in Docker |
| | `llm_model` | `gemma3n:latest` | |
| | `embedding_model` | `embeddinggemma:latest` | |
| | `request_timeout_seconds` | `90` | |

---

## 10. API Endpoints Summary

| Method | Endpoint | Handler | Description |
|---|---|---|---|
| `POST` | `/query` | `query_agent` | Route a NL query through the agent pipeline |
| `POST` | `/upload` | `upload_file` | Upload a document or CSV/XLSX |
| `GET` | `/status` | `status` | List already-ingested files |
| `DELETE` | `/reset` | `reset_database` | Wipe Milvus (+ optionally Postgres) |

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Two-pass routing** (LLM + heuristic) | LLM alone can misclassify obvious SQL queries; keyword scoring acts as a safety net |
| **SQL fallback to DocumentAgent** | If the SQL agent raises (bad query, missing table etc.), the user still gets an answer |
| **SHA-1 doc_id** | Short stable ID ties all chunks from the same file together without relying on the filename alone |
| **Parallel embedding with ThreadPoolExecutor** | Embedding is the bottleneck at ingestion time; parallelism gives near-linear speedup |
| **Fail-fast on embedding mismatch** | If the embedding model changes and dim no longer matches, we crash loudly rather than insert garbage |
| **Metadata TTL cache (300s)** | Avoids querying Postgres on every request to get table/column names for heuristic routing |
| **`stream=False` on Ollama generate** | Simplifies the response path — no streaming parser needed |
| **All SQL columns as TEXT** | Avoids type-inference errors when loading arbitrary CSVs from users |
| **Daemon watcher thread** | Background ingestion without blocking the API; dies automatically when the main process exits |
