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
    Production-ready orchestrator for routing queries to:
    - StructuredDataAgent (SQL/database queries)
    - DocumentAgent (unstructured QA)
    - SummarizationAgent (summarization tasks)
    """

    def __init__(self, metadata_ttl: int = 300):
        self.ingestion_pipeline = IngestionPipeline()
        self.router = IntentRouterAgent()

        self.document_agent = DocumentAgent()
        self.summary_agent = SummarizationAgent()
        self.structured_agent = StructuredDataAgent()

        # --- SQL metadata cache ---
        self._cached_tables: List[str] = []
        self._cached_columns: List[str] = []
        self._metadata_last_refresh: float = 0
        self._metadata_ttl = metadata_ttl

        # Warm cache at startup
        self.get_sql_metadata()

    # ============================================================
    # SQL METADATA
    # ============================================================

    def get_sql_metadata(self) -> Tuple[List[str], List[str]]:
        """
        Fetch and cache table + column metadata with TTL.
        """
        now = time.time()

        # Return cached metadata if valid
        if (
            self._cached_tables
            and now - self._metadata_last_refresh < self._metadata_ttl
        ):
            return self._cached_tables, self._cached_columns

        try:
            logger.info("Refreshing SQL metadata cache")

            tables_rows = self.structured_agent.sql_agent.db.execute_query(
                "SELECT tablename FROM pg_tables WHERE schemaname='public';"
            )
            tables = [t[0] for t in tables_rows]

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

            # Cache results
            self._cached_tables = tables
            self._cached_columns = columns
            self._metadata_last_refresh = now

            return tables, columns

        except Exception as e:
            logger.error(f"Failed to fetch SQL metadata: {e}")
            return [], []

    # ============================================================
    # SQL DETECTION
    # ============================================================

    def looks_like_sql_query(self, query: str, tables: list, columns: list) -> bool:
        """
        Score-based heuristic SQL detection to reduce false positives.
        """
        q = query.lower()
        score = 0

        sql_keywords = [
            "select", "from", "where",
            "group by", "order by",
            "join", "having"
        ]

        # Strong keyword match
        if any(keyword in q for keyword in sql_keywords):
            score += 3

        # Table name match
        if any(table.lower() in q for table in tables):
            score += 2

        # Column name match
        if any(col.lower() in q for col in columns):
            score += 1

        # Strong structural signal
        if "select" in q and "from" in q:
            score += 5

        return score >= 3

    # ============================================================
    # FILE INGESTION
    # ============================================================

    def ingest_file(self, file_path: str):
        """
        Process a document file and ingest into vector store.
        """
        try:
            chunks = self.ingestion_pipeline.process_file(file_path)
            self.ingestion_pipeline.ingest_to_milvus(chunks)
            logger.info(f"Successfully ingested {file_path}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise

    # ============================================================
    # MAIN ROUTER
    # ============================================================

    def handle_query(self, query: str, task_type=None) -> AgentRunOutput:
        """
        Main routing logic with:
        - Intent detection
        - SQL heuristic override
        - Safe fallback
        - Structured logging
        """
        start_time = time.time()

        try:
            # Step 1: LLM intent detection
            intent = self.router.detect_intent(query)
            logger.info(f"Detected intent: {intent}")

            # Step 2: SQL heuristic check
            tables, columns = self.get_sql_metadata()
            sql_like = self.looks_like_sql_query(query, tables, columns)

            # Step 3: Routing decision
            if intent == "structured_data":
                route = "structured_data"
            elif sql_like:
                route = "structured_data"
                logger.info("SQL heuristic override triggered")
            elif intent == "summarize":
                route = "summarize"
            else:
                route = "document_qa"

            logger.info(
                f"Routing decision: {route} | "
                f"latency={time.time() - start_time:.2f}s"
            )

            # Step 4: Execute
            if route == "structured_data":
                try:
                    return self.structured_agent.handle_query(
                        query, task_type or route
                    )
                except Exception:
                    logger.warning(
                        "Structured agent failed. Falling back to DocumentAgent."
                    )
                    return self.document_agent.handle_query(
                        query, task_type or "document_qa"
                    )

            if route == "summarize":
                return self.summary_agent.handle_query(
                    query, task_type or route
                )

            return self.document_agent.handle_query(
                query, task_type or "document_qa"
            )

        except Exception as e:
            logger.exception("Orchestrator failure")
            return AgentRunOutput(
                answer="System error occurred while processing the query.",
                intent="error",
                confidence_score=0.0,
                reasoning_steps=["Exception in orchestrator"],
                sources=[],
            )