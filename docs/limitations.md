# Limitations

An honest breakdown of the current system's constraints — with the root cause and potential fix for each.

---

## 1. SQL Agent — Read-Only Queries Only

**Limitation:** Only `SELECT` and `WITH` (CTE) queries are allowed. No `INSERT`, `UPDATE`, `DELETE`, or `CREATE`.

**Root cause:** A deliberate safety decision to prevent the LLM from mutating or destroying data.

**Impact:** Users cannot ask the system to modify data — only retrieve and analyze it.

**Potential fix:** Add a role-based permission system where trusted users can enable write operations, scoped to specific tables.

---

## 2. No Deduplication on Re-Upload

**Limitation:** Uploading the same file twice inserts duplicate chunks into Milvus.

**Root cause:** The ingestion pipeline does not check if a `doc_id` already exists in Milvus before inserting. Milvus auto-IDs prevent exact-duplicate detection at the DB level.

**Impact:** Repeated uploads bloat the collection and inflate retrieval scores for repeated documents.

**Potential fix:** Before insertion, query Milvus for the `doc_id` and skip ingestion if it already exists. The SHA-1-based `doc_id` makes this check straightforward.

---

## 3. No Multi-User / Session Isolation

**Limitation:** All users share the same Milvus collection and Postgres schema. There is no concept of user-scoped data or access control.

**Root cause:** The system was designed as a single-user or single-team tool.

**Impact:** In a multi-tenant environment, User A can query data uploaded by User B.

**Potential fix:** Add a `user_id` or `tenant_id` field to Milvus and filter on it at search time. Separate Postgres schema per tenant.

---

## 4. Memory Spike During Large File Ingestion

**Limitation:** The `ingest_to_milvus` method loads all chunks for a batch into memory simultaneously, plus their embeddings.

**Root cause:** For a 1000-token chunk with 768-dim embeddings, each `float32` embedding is ~3 KB. A 1000-chunk batch holds ~3 MB of embeddings in RAM — fine for normal documents, but large files (e.g. a 500-page PDF) produce thousands of chunks.

**Impact:** Memory usage scales linearly with document size. Very large files may cause OOM on memory-constrained servers.

**Potential fix:** Reduce `batch_size` from 1000, or implement streaming ingestion where chunks are embedded and inserted one batch at a time without accumulating all embeddings first.

---

## 5. Watcher Thread Has No File-Type Guard

**Limitation:** The `_watch_folder` background thread attempts to ingest any file that appears in `uploaded_docs/`, including temporary files (`.tmp`, `.part`) created by some OS or upload tools.

**Root cause:** The watcher only checks `if f.is_file()` — no extension filter.

**Impact:** Temporary files can trigger failed ingestion attempts and error logs.

**Potential fix:** Add an allowed-extensions check in `_watch_folder` before calling `ingest_file()`, mirroring the `SUPPORTED_EXTENSIONS` set in `DocumentLoader`.

---

## 6. SQL Table Validation is Crude

**Limitation:** The table existence check in `SQLAgent.generate_sql()` tokenizes the SQL string and checks if any word matches a table name — it doesn't actually parse the SQL AST.

**Root cause:** A full SQL parser (e.g. `sqlparse` or `pglast`) was not used to keep dependencies minimal.

**Impact:** A query referencing a valid table via an alias or a subquery alias may fail the check even when the underlying table exists. False negatives are possible.

**Potential fix:** Use `sqlparse` for proper SQL parsing, or use `EXPLAIN` on the generated query and catch any Postgres error as the validation step.

---

## 7. Summarization Quality Depends on Retrieval

**Limitation:** `SummarizationAgent` retrieves chunks the same way `DocumentAgent` does — by semantic similarity to the query. This means a vague query like *"summarize the document"* may retrieve only 5 chunks from a large document, giving an incomplete summary.

**Root cause:** Summarization is treated as a retrieval problem rather than a full-document scan.

**Impact:** Summaries of long documents may miss important sections.

**Potential fix:** For summarization, fetch significantly more chunks (e.g. 30–50) or implement a map-reduce summarization strategy: summarize each chunk, then summarize the summaries.

---

## 8. No Authentication or Rate Limiting

**Limitation:** The FastAPI endpoints have no authentication, rate limiting, or request validation beyond type checking.

**Root cause:** The system was built as an internal tool, not a public API.

**Impact:** Anyone with network access to port 8000 can upload files, query the system, or call `/reset`.

**Potential fix:** Add FastAPI dependency injection for API key or JWT auth. Use a reverse proxy (nginx) for rate limiting.

---

## 9. Ollama Timeout Errors Surface as 500s

**Limitation:** If Ollama is slow or unavailable, the API returns a 500 Internal Server Error with a raw `OllamaTimeoutError` message.

**Root cause:** The exception is caught at the orchestrator level but re-raised as a generic 500 in the FastAPI handler.

**Impact:** Poor user experience — the error message is technical and doesn't tell the user what to do.

**Potential fix:** Catch `OllamaTimeoutError` specifically in the FastAPI route handler and return a 503 Service Unavailable with a friendly message like *"The AI model is currently unavailable. Please try again in a few seconds."*

---

## 10. CSV Columns Always Stored as TEXT

**Limitation:** When uploading a CSV/XLSX, all columns are created as `TEXT` in Postgres — even if they contain numbers or dates.

**Root cause:** Deliberate choice to avoid type-inference failures on arbitrary user data.

**Impact:** Numeric comparisons in SQL require explicit casting (`salary::numeric > 50000`). Sorting by "number" columns sorts lexicographically unless cast. The LLM may not always generate the correct cast.

**Potential fix:** Use pandas `dtype` inference to detect numeric and date columns and create them with appropriate Postgres types (`NUMERIC`, `DATE`, `TIMESTAMP`), with a fallback to `TEXT`.
