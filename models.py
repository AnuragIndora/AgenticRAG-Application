from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# 🔥 Updated Intent Types (Hybrid System)
IntentType = Literal[
    "document_qa",
    "summarization",
    "structured_data",
    "comparison",
    "multi_hop",
]


class ChunkRecord(BaseModel):
    text: str
    doc_id: str
    file_name: str
    file_type: Optional[str] = None
    page_number: int = 0
    chunk_id: str
    chunk_type: str = "text"


class RetrievalResult(BaseModel):
    id: int
    text: str
    doc_id: str
    file_name: str
    page_number: int
    chunk_id: str
    chunk_type: str
    score: float


# 🔥 Query Analysis now supports structured
class QueryAnalysis(BaseModel):
    original_query: str
    intent: IntentType
    expanded_queries: list[str] = Field(default_factory=list)
    requires_sql: bool = False


# 🔥 SQL Execution Result
class SQLExecutionResult(BaseModel):
    query: str
    rows: list
    row_count: int


class ReasoningResponse(BaseModel):
    answer: str
    confidence_score: float
    citations: list[str] = Field(default_factory=list)
    sql_query: Optional[str] = None


# 🔥 Final agent output supports both RAG and SQL
class AgentRunOutput(BaseModel):
    answer: str
    intent: IntentType
    confidence_score: float
    reasoning_steps: list[str] = Field(default_factory=list)
    sources: list[RetrievalResult] = Field(default_factory=list)
    sql_query: Optional[str] = None


class IngestResponse(BaseModel):
    ingested_files: int
    ingested_chunks: int
    failed_files: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    intent: IntentType
    confidence_score: float
    sources: list[dict] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    sql_query: Optional[str] = None