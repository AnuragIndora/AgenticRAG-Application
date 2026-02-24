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