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