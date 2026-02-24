# Technology Choices & Rationale

## Stack Summary

| Layer | Technology | Version |
|---|---|---|
| API Framework | FastAPI | 0.115.6 |
| UI | Streamlit | 1.41.1 |
| Vector Database | Milvus (standalone) | 2.4.16 |
| Relational Database | PostgreSQL | 15 |
| LLM & Embeddings | Ollama (local server) | latest |
| LLM Model | gemma3n | latest |
| Embedding Model | embeddinggemma | latest |
| Document Parsing | LlamaIndex `SimpleDirectoryReader` | 0.14.15 |
| HTTP Client | requests | 2.32.3 |
| ORM / DB Driver | psycopg2 | 2.9.11 |
| Data Validation | Pydantic v2 | 2.12.5 |
| Config Management | pydantic-settings | 2.7.1 |
| Containerization | Docker + Docker Compose | — |
| ASGI Server | Uvicorn | 0.34.0 |

---

## Component-by-Component Rationale

### FastAPI
**Why FastAPI over Flask / Django?**
- Async-native and production-ready with Uvicorn
- Auto-generates `/docs` (Swagger UI) from type annotations — zero extra effort
- Pydantic request/response validation is built-in, directly compatible with the data models we already use

### Streamlit
**Why Streamlit?**
- Zero frontend code required — entire UI written in Python
- Built-in file uploader, chat message components, and session state handle 90% of the features needed
- Runs as a separate service and talks to FastAPI via HTTP — clean separation of concerns

### Milvus (local standalone)
**Why Milvus over Pinecone / Qdrant / Weaviate / ChromaDB?**

| Consideration | Decision |
|---|---|
| Fully local | Milvus standalone runs in Docker, no cloud dependency |
| HNSW index | Excellent recall-speed tradeoff for our chunk sizes |
| COSINE metric | Scale-invariant — works well even when embedding magnitudes vary |
| Schema flexibility | Stores custom metadata alongside vectors (doc_id, page_number, etc.) |
| Production-grade | Battle-tested at scale (not a research prototype) |

> ChromaDB was considered — it's simpler, but lacks production-level reliability and index control.

### PostgreSQL
**Why PostgreSQL for structured data?**
- Full SQL support — CTEs, window functions, joins — without the SQL generation needing to be model-specific
- `psycopg2` with `execute_batch` gives fast bulk inserts for uploaded CSVs
- `pg_tables` and `information_schema` are used by the heuristic router to dynamically discover table/column names — this works out-of-the-box with Postgres
- Auto-creates the target database if missing — zero manual DB setup needed

### Ollama (local LLM server)
**Why Ollama over OpenAI / Anthropic API?**

| Concern | Ollama advantage |
|---|---|
| Privacy | Data never leaves the machine |
| Cost | Zero per-token cost |
| Latency | No network round-trip to a remote API |
| Model flexibility | Swap models by changing two env vars, no code change |
| Offline capability | Works without internet access |

Ollama serves both roles — **embedding** (`/api/embeddings`) and **generation** (`/api/chat`) — from a single running server, keeping the infrastructure simple.

### gemma3n (LLM) + embeddinggemma (Embedding model)
- `gemma3n` — Google's Gemma 3 Nano, efficient and capable for Q&A and SQL generation on consumer hardware
- `embeddinggemma` — Matched embedding model; using the same model family for both roles keeps the semantic space consistent

### LlamaIndex `SimpleDirectoryReader`
**Why LlamaIndex for document loading?**
- Handles PDF, DOCX, PPTX, TXT, MD out-of-the-box with a single call
- Returns structured `Document` objects with `metadata` (page number, file type, etc.) already populated
- Avoids writing custom parsers for each file format

### Pydantic v2
- Used for all data models (`ChunkRecord`, `RetrievalResult`, `AgentRunOutput`, etc.)
- `pydantic-settings` loads `.env` config with zero boilerplate
- `@lru_cache` on `get_settings()` makes config a singleton — parsed once, shared everywhere

### requests (HTTP client for Ollama)
- Simple synchronous HTTP — appropriate since Uvicorn handles concurrency at the ASGI level
- Explicit `timeout` passed on every call, raising `OllamaTimeoutError` on failure
- Could be replaced with `httpx` for async if needed in the future

### psycopg2 + execute_batch
- `execute_batch` from `psycopg2.extras` sends rows to Postgres in configurable page sizes (default 1000) — far faster than individual `INSERT` calls
- `autocommit=True` chosen because each operation is intentionally atomic

### Docker + Docker Compose
- Single `Dockerfile` with a multi-stage build (builder + runtime stages) — keeps the final image small by not shipping build tools
- `docker-compose.yml` wires all 6 services with health checks and explicit dependency ordering
- Same image used for both API and UI services — only the `CMD` differs, reducing image maintenance

---

## What Was Intentionally NOT Used

| Tool | Reason skipped |
|---|---|
| LangChain / LlamaIndex agent framework | Overhead, black-box routing — custom orchestration gives full control over routing logic |
| OpenAI / Anthropic | Privacy, cost, offline requirement |
| Redis | No cross-request caching needed; metadata TTL cache is in-process |
| Celery / task queue | Background ingestion handled with a simple daemon thread — sufficient for single-server deployment |
| SQLAlchemy ORM | Not needed — all queries are either dynamically generated SQL or simple DDL; raw psycopg2 is simpler |
