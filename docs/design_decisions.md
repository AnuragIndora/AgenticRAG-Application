# Key Design Decisions

Each decision here is directly traceable to code in the project.

---

## 1. Two-Pass Intent Routing (LLM + Heuristic)

**Decision:** Run LLM-based intent classification AND a score-based keyword heuristic, then combine them.

**Why not LLM alone?**  
LLMs can hallucinate or misclassify queries like *"Show me all employees where salary > 50000"* as `document_qa` because the phrasing sounds like a question. The heuristic catches these with high precision.

**Why not heuristic alone?**  
Pure keyword matching fails on natural language SQL requests like *"What's the highest-paid department?"* — no SQL keywords present.

**Code location:** `agents_pipeline.py` → `handle_query()` + `looks_like_sql_query()`

---

## 2. SQL Agent Falls Back to Document Agent

**Decision:** Wrap `StructuredDataAgent.handle_query()` in a try/except. On any failure, silently fall back to `DocumentAgent`.

**Rationale:** SQL failures are common — the LLM may reference a non-existent column, generate invalid SQL, or the DB may be empty. Crashing the entire request on a routing mistake is a bad user experience. The user gets a document-grounded answer instead of an error.

**Code location:** `agents_pipeline.py` → `handle_query()` lines inside `if route == "structured_data":`

```python
try:
    return self.structured_agent.handle_query(query, ...)
except Exception:
    logger.warning("Structured agent failed. Falling back to DocumentAgent.")
    return self.document_agent.handle_query(query, ...)
```

---

## 3. SQL Metadata Cache with TTL

**Decision:** Cache Postgres table/column names in memory with a 300-second TTL.

**Why:** The heuristic scorer needs real table/column names to score queries accurately. Hitting Postgres on every single request (potentially every few seconds) is wasteful. 300 seconds is a sensible balance — schema changes are rare.

**Code location:** `agents_pipeline.py` → `get_sql_metadata()`

```python
if self._cached_tables and now - self._metadata_last_refresh < self._metadata_ttl:
    return self._cached_tables, self._cached_columns
```

---

## 4. Stable SHA-1 Doc ID

**Decision:** Generate `doc_id` as `{filename_stem}-{sha1_of_file[:12]}` rather than a UUID or timestamp.

**Why:** Re-uploading the same file should produce the same `doc_id`. UUIDs are random — they'd create duplicate chunks on re-upload. Hashing the raw bytes gives a stable, file-content-based identity. The first 12 characters of SHA-1 is more than enough uniqueness (48 bits of entropy).

**Code location:** `ingestion_pipeline.py` → `process_file()`

```python
file_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
doc_id = f"{path.stem}-{file_hash}"
```

---

## 5. Parallel Embedding with Fail-Fast

**Decision:** Use `ThreadPoolExecutor` with 8 workers for embedding, and raise immediately if any chunk fails.

**Why parallel:** Embedding is the bottleneck during ingestion. Each call to Ollama takes ~100–500ms. 8 parallel workers give near-linear speedup for large batches.

**Why fail-fast:** A partial ingestion (some chunks embedded, some not) is worse than no ingestion — it silently corrupts the retrieval results. Better to fail loudly and let the user try again.

**Code location:** `ingestion_pipeline.py` → `ingest_to_milvus()`

---

## 6. Embedding Dimension Validation

**Decision:** After every embed call, check `len(emb) == settings.embedding_dim` and raise if it doesn't match.

**Why:** If someone swaps the embedding model in `.env` to one with a different output dimension, the Milvus insert would silently fail or corrupt the schema. The check surfaces this immediately with a clear error message.

**Code location:** `ingestion_pipeline.py` → embedding loop in `ingest_to_milvus()`

---

## 7. Two-Stage Retrieval (Broad ANN → Narrow Re-rank)

**Decision:** Fetch 20 ANN candidates, re-rank by hybrid score, return top 5.

**Why not just fetch 5?** ANN search is approximate. The top-5 by pure cosine similarity may miss chunks that contain the exact keywords the user used. Fetching a wider pool and re-ranking gives better precision without a significant latency cost.

**Hybrid formula:** `0.8 × cosine_score + 0.2 × keyword_overlap`

The 80/20 split keeps semantic meaning dominant while giving a tie-breaking boost to lexically matching chunks.

**Code location:** `rag_pipeline.py` → `retrieve()` + `milvuous_client.py` → `keyword_filter()`

---

## 8. Milvus `flush()` After Every Insert

**Decision:** Call `self.collection.flush()` immediately after `collection.insert()`.

**Why:** Milvus buffers inserts in memory before persisting. Without `flush()`, a query immediately after an upload could miss the newly inserted chunks. `flush()` forces the data to be available for search right away.

**Code location:** `milvuous_client.py` → `insert()`

---

## 9. `stream=False` on Ollama Generate

**Decision:** Always use `stream=False` in the Ollama chat API call.

**Why:** Streaming would require a response parser to accumulate chunks — additional complexity with no benefit here since the API is already async-friendly via FastAPI. The full response is returned in one shot and processed synchronously.

**Code location:** `ollama_client.py` → `generate()`

---

## 10. All Uploaded CSV/XLSX Columns Stored as TEXT

**Decision:** When loading a CSV/XLSX into Postgres, all columns are created as `TEXT`.

**Why:** Auto-inferring types from user-uploaded files is risky — a column that looks numeric may contain nulls, currency symbols, or mixed types. TEXT is permissive and avoids insert failures. The SQL agent's LLM can still perform numeric comparisons (Postgres casts automatically in `WHERE salary::numeric > 50000`).

**Code location:** `postgres_client.py` → `create_table_from_dataframe()`

---

## 11. Daemon Thread for Folder Watcher

**Decision:** The folder watcher runs as `daemon=True`.

**Why:** A daemon thread automatically dies when the main process (Uvicorn) exits. Using a non-daemon thread would block the process from shutting down cleanly after a `Ctrl+C` or container stop signal — requiring an explicit shutdown mechanism.

**Code location:** `real_time_agent.py` → `__init__()`

---

## 12. `@lru_cache` on `get_settings()`

**Decision:** Settings are parsed once and cached as a singleton via `@lru_cache(maxsize=1)`.

**Why:** `pydantic-settings` reads and validates all `.env` values every time `Settings()` is called. Calling it once and caching the result avoids repeated I/O on every request. All modules that need settings call `get_settings()` — they all get the same object.

**Code location:** `settings.py` → `get_settings()`

---

## 13. Only SELECT / WITH Queries Allowed (SQL Safety)

**Decision:** After SQL generation, reject any query that doesn't start with `SELECT` or `WITH`.

**Why:** The LLM could theoretically generate `DROP TABLE users;` or `DELETE FROM orders;`. The `generate_sql` method validates the output before execution. Additionally, `StructuredDataAgent.handle_query()` has a second guard for the same.

**Code location:** `sql_agent.py` → `generate_sql()`

```python
if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
    raise ValueError(f"Only SELECT queries are allowed.")
```
