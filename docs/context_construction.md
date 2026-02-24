# Context Construction Strategy

## Overview

The quality of the LLM's answer is directly tied to the quality of the context fed into the prompt. This system uses a **chunking → embedding → ANN search → hybrid re-rank → prompt injection** pipeline to build the best possible context window for each query.

---

## Step 1 — Text Cleaning (`TextCleaner` / `text_preprocess.py`)

Before any chunking happens, raw text from documents is normalized:

| Operation | Why |
|---|---|
| Strip control characters (0x00–0x1F, except \\t \\n \\r) | Corrupt bytes break tokenization |
| Remove soft-hyphen `\u00ad` | Invisible character that silently breaks word boundaries |
| Remove BOM `\ufeff` | Appears at the start of UTF-8 files, confuses tokenizers |
| Collapse multiple whitespace into single space | Removes artifacts from PDF extraction |

---

## Step 2 — Chunking Strategy (`Chunker`)

Documents are split into **overlapping token windows** using a simple word-boundary tokenizer (`re.findall(r"\b\w+\b|[^\w\s]", ...)`).

| Parameter | Value | Config key |
|---|---|---|
| Chunk size | 1000 tokens | `chunk_size_tokens` |
| Overlap | 150 tokens | `chunk_overlap_tokens` |

### Why overlapping chunks?

Without overlap, sentences that fall exactly on a chunk boundary get split — the context on either side loses meaning. A 150-token overlap ensures every sentence has full context from both its predecessor and successor chunk.

```
Token positions:
  Chunk 1:  [0 ─────────────────── 999]
  Chunk 2:          [850 ─────────────────── 1849]
  Chunk 3:                  [1700 ─────────────────── 2699]
                    ↑ 150-token overlap
```

Each chunk is wrapped in a `ChunkRecord` with:
- `chunk_id` → `{doc_id}-chunk-N` (globally unique)
- `doc_id` → `{filename_stem}-{sha1[:12]}` (stable across re-uploads of the same file)
- `page_number`, `file_name`, `file_type` (for source attribution)

---

## Step 3 — Embedding (`OllamaClient.embed`)

Each chunk is converted to a **768-dimensional float vector** using the local Ollama embedding model (`embeddinggemma:latest`).

- Embedding is done **in parallel** via `ThreadPoolExecutor` (8 workers by default)
- Each embedding is validated: `len(emb) == settings.embedding_dim` — mismatch raises immediately
- Vectors are stored in Milvus alongside the chunk metadata

---

## Step 4 — Retrieval: ANN Search (`MilvusVectorStore.search`)

At query time, the query string is embedded with the same model. Milvus runs an **Approximate Nearest Neighbour (ANN)** search using the **HNSW index**:

| Parameter | Value | Meaning |
|---|---|---|
| `metric_type` | `COSINE` | Angle-based similarity (scale-invariant) |
| `index_type` | `HNSW` | Graph-based index, fast at high recall |
| `M` | 16 | Connections per node during index build |
| `efConstruction` | 200 | Search width during index construction |
| `ef` (search) | 128 | Search width at query time |
| `candidate_k` | 20 | Initial pool of candidates fetched |

We intentionally fetch **20 candidates** for a final `top_k=5` because the re-ranking step below can significantly reorder them.

---

## Step 5 — Re-Ranking: Hybrid Keyword Filter (`keyword_filter`)

Pure vector similarity can rank chunks that are semantically close but miss critical keywords. The re-ranker corrects for this:

```
hybrid_score = (cosine_score × 0.8) + (keyword_overlap_count × 0.2)
```

- **80% weight** — vector similarity (semantic relevance)
- **20% weight** — keyword overlap (lexical precision)
- Short words ≤ 2 characters are excluded (removes stop-word noise like "is", "in", "of")

Result: the top-5 chunks by hybrid score are kept as the final context.

---

## Step 6 — Prompt Assembly

The 5 retrieved chunk texts are joined with `\n\n` and injected into a structured prompt:

### DocumentAgent prompt template

```
You are a precise AI assistant.

Use ONLY the provided context.
If the answer is not found, say "I don't know".

Context:
{chunk_1_text}

{chunk_2_text}
...

Question:
{user_query}

Provide a clear answer.
```

### SummarizationAgent prompt template

```
Summarize the following content concisely:

{chunk_1_text}

{chunk_2_text}
...
```

### Key prompting constraints
- LLM is **explicitly told to use ONLY the provided context** → prevents hallucination
- Responses are bounded by retrieved content → factual grounding
- `temperature=0.5` → balanced between deterministic and creative

---

## Context Window Budget

| Component | Limit |
|---|---|
| Max chunks | 5 |
| Max chunk size | 1000 tokens |
| Max total context | ~5000 tokens (bounded by `max_context_tokens=3500` in settings) |
| LLM model | gemma3n (supports long context) |

---

## SQL Context Construction

For structured queries, context is constructed differently — no vector retrieval is involved:

```
User question
  ↓
LLM generates SQL (SELECT only)
  ↓
PostgreSQL returns raw rows
  ↓
Prompt:
  "User Question: {query}
   SQL Query Used: {sql}
   SQL Result: {rows}
   Provide a concise and clear answer."
```

The raw SQL result acts as the grounding context — the LLM is given the exact data and asked to format it for a human.
