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