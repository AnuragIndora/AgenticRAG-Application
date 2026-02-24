# Agentic RAG System

An **Agentic Retrieval-Augmented Generation (RAG)** system that lets you query both **unstructured documents** (PDF, DOCX, PPTX, TXT, MD) and **structured data** (CSV, XLSX → PostgreSQL) using natural language. Built with FastAPI, Milvus, PostgreSQL, and a local Ollama LLM — no cloud AI needed.

---

## Features

- 📄 **Document Q&A** — Upload files and ask questions grounded in the document content
- 🗄️ **SQL Q&A** — Ask questions about structured data (CSVs/Excel) in plain English
- ✂️ **Summarization** — Get concise summaries of uploaded documents
- 🔄 **Auto-ingestion** — Drop files into the watch folder and they're indexed automatically
- 🖥️ **Streamlit UI** — Chat interface with file upload and database reset controls
- 🔒 **SQL Safety** — Only `SELECT` and `WITH` queries are ever executed

---

## System Architecture

![alt text](images/image10.png)

### Routing Logic

1. **IntentRouterAgent** asks the LLM to classify the query as `structured_data`, `document_qa`, or `summarize`
2. A **score-based SQL heuristic** runs in parallel as a sanity check (catches obvious SQL queries the LLM might miss)
3. The orchestrator routes to the right agent — and falls back to `DocumentAgent` if the SQL agent fails

---

## Project Structure

```
agentic_rag2/
├── main.py                  # FastAPI app — upload, query, reset endpoints
├── streamlit-app.py         # Chat UI
├── agents.py                # DocumentAgent, SummarizationAgent, StructuredDataAgent
├── agents_pipeline.py       # AgentsOrchestrator — routing + metadata caching
├── intent_router_agent.py   # LLM-based intent classifier
├── rag_pipeline.py          # Two-stage retrieval (ANN + keyword re-rank)
├── ingestion_pipeline.py    # Load → clean → chunk → embed → insert
├── milvuous_client.py       # Milvus connection, insert, search, re-rank
├── ollama_client.py         # Ollama HTTP client (embed + generate)
├── postgres_client.py       # PostgreSQL connection and query helpers
├── sql_agent.py             # NL → SQL generation + execution
├── text_preprocess.py       # Text cleaning and token chunking utilities
├── models.py                # Pydantic data models
├── settings.py              # Config loaded from .env
├── exceptions.py            # Custom exception hierarchy
├── logger.py                # Centralized logging setup
└── uploaded_docs/           # Watch folder — files dropped here get auto-ingested
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| API | FastAPI | Lightweight, async, clean REST |
| Vector DB | Milvus (HNSW index) | Fast ANN search, cosine similarity |
| SQL DB | PostgreSQL | Structured data + relational queries |
| LLM & Embeddings | Ollama (local) | Fully local, no API keys needed |
| Document Parsing | LlamaIndex `SimpleDirectoryReader` | Handles PDF, DOCX, PPTX, TXT, MD |
| UI | Streamlit | Quick chat interface |

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- Milvus running locally (or via Docker)
- PostgreSQL running locally

### 1. Clone the repo

```bash
git clone <repo_url>
cd agentic_rag2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
ollama pull gemma3n         # LLM for chat/generation
ollama pull embeddinggemma  # Embedding model
```

### 4. Start Milvus with Docker

```bash
docker-compose -f docker-compose.milvus.yml up -d
```

### 5. Configure environment

Create a `.env` file in the project root:

```env
# FastAPI
API_HOST=0.0.0.0
API_PORT=8000

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=agentic_rag_chunks
MILVUS_USERNAME=
MILVUS_PASSWORD=
EMBEDDING_DIM=768

# PostgreSQL
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=agentic_rag
PG_USER=raguser
PG_PASSWORD=ragpass

# Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=gemma3n:latest
EMBEDDING_MODEL=embeddinggemma:latest
REQUEST_TIMEOUT_SECONDS=90

# Chunking
CHUNK_SIZE_TOKENS=1000
CHUNK_OVERLAP_TOKENS=150
RETRIEVAL_TOP_K=8
```

### 6. Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. (Optional) Start the Streamlit UI

```bash
streamlit run streamlit-app.py
```

---

## API Reference

### `POST /query`
Send a natural language question.

```json
{
  "query": "What were the total sales last quarter?",
  "task_type": null
}
```

**Response:**
```json
{
  "answer": "...",
  "intent": "structured_data",
  "confidence_score": 0.9,
  "sources": [],
  "reasoning_steps": ["Generated SQL query", "Executed SQL", "Formatted result"],
  "sql_query": "SELECT ..."
}
```

---

### `POST /upload`
Upload a file for ingestion.

- **PDF, DOCX, PPTX, TXT, MD** → chunked, embedded, stored in Milvus
- **CSV, XLSX** → loaded directly into a PostgreSQL table (table name = filename without extension)

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf"
```

---

### `GET /status`
Returns the list of files already ingested.

```json
{ "ingested_files": ["report.pdf", "sales_data.csv"] }
```

---

### `DELETE /reset`
Wipe all Milvus vectors and clear the ingestion tracker.

```bash
# Reset only Milvus
curl -X DELETE http://localhost:8000/reset

# Also drop all PostgreSQL tables
curl -X DELETE "http://localhost:8000/reset?delete_sql=true"
```

---

## How Ingestion Works

```
File Upload
    │
    ├── DocumentLoader       → Reads file using LlamaIndex
    ├── TextCleaner          → Strips control chars, BOM, normalizes whitespace
    ├── Chunker              → Splits into overlapping token windows (1000 tokens, 150 overlap)
    ├── OllamaClient.embed() → Generates 768-dim vector per chunk (parallel, thread pool)
    └── MilvusVectorStore.insert() → Stores vectors + metadata
```

## How Retrieval Works

```
User Query
    │
    ├── OllamaClient.embed()       → Embed the query
    ├── MilvusVectorStore.search() → ANN search, top-20 candidates
    ├── keyword_filter()           → Hybrid re-rank: 80% vector + 20% keyword overlap
    └── Top-5 chunks → LLM context → Answer
```

---

---

## Docker Deployment

The project ships with three Docker files:

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build — produces a single image used for both the API and UI |
| `docker-compose.yml` | Spins up all 6 services: API, UI, PostgreSQL, Milvus, MinIO, etcd, Ollama |
| `.dockerignore` | Keeps the image clean (excludes cache, notebooks, local data, `.env`) |

### Build & push your image to Docker Hub

```bash
# 1. Log in to Docker Hub
docker login

# 2. Build the image (replace with your Docker Hub username)
docker build -t YOUR_DOCKERHUB_USERNAME/agentic-rag:latest .

# 3. Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/agentic-rag:latest
```

Then update the two `image:` lines in `docker-compose.yml` with your actual username:
```yaml
image: YOUR_DOCKERHUB_USERNAME/agentic-rag:latest
```

### Run everything with Docker Compose

```bash
# Start all services (API, UI, Postgres, Milvus, Ollama)
docker compose up -d

# Check all containers are healthy
docker compose ps

# Pull LLM models into the Ollama container (first time only)
docker exec agentic-ollama ollama pull gemma3n
docker exec agentic-ollama ollama pull embeddinggemma

# View API logs
docker compose logs -f api

# Stop everything
docker compose down
```

Access points after startup:
- **FastAPI** → http://localhost:8000
- **Streamlit UI** → http://localhost:8501
- **API Docs** → http://localhost:8000/docs

> **Note:** GPU support for Ollama is enabled in `docker-compose.yml` via the `deploy.resources` block.  
> Remove that block if you're running on CPU only.

---

## Limitations

- SQL agent only supports `SELECT` and `WITH` (CTE) queries — no writes
- Large files may use significant memory during parallel embedding
- No user authentication or multi-tenancy
- Summarization quality depends on how well relevant chunks are retrieved
- Milvus `VARCHAR` fields have max-length constraints (metadata must fit within them)
