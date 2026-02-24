import logging
import psycopg2
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_batch
from settings import get_settings

logger = logging.getLogger("PostgresClient")


class PostgresClient:
    """
    Handles Postgres connectivity and convenience methods for structured data.
    Auto-creates the target database on first run if it doesn't exist.
    """

    def __init__(self):
        self.settings = get_settings()
        self.dbname = self.settings.pg_database
        self._ensure_database_exists()
        self.conn = psycopg2.connect(
            host=self.settings.pg_host,
            port=self.settings.pg_port,
            dbname=self.dbname,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
        )
        self.conn.autocommit = True
        logger.info(f"Connected to database: {self.dbname}")

    def _ensure_database_exists(self):
        """Connect to the default 'postgres' DB and create our target DB if it's missing."""
        try:
            default_conn = psycopg2.connect(
                host=self.settings.pg_host,
                port=self.settings.pg_port,
                dbname="postgres",
                user=self.settings.pg_user,
                password=self.settings.pg_password,
            )
            default_conn.autocommit = True

            with default_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.dbname,))
                exists = cur.fetchone()

                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.dbname)))
                    logger.info(f"Database '{self.dbname}' created.")
                else:
                    logger.info(f"Database '{self.dbname}' already exists.")

            default_conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame, batch_size: int = 1000):
        """
        Create a table from a DataFrame (all columns as TEXT) and bulk-insert rows.
        Uses execute_batch for efficient batched inserts instead of row-by-row.
        Does nothing if the DataFrame is empty.
        """
        if df.empty:
            logger.warning(f"DataFrame is empty. Table '{table_name}' not created.")
            return

        # Build CREATE TABLE ... with one TEXT column per DataFrame column
        columns = [sql.SQL("{} TEXT").format(sql.Identifier(col)) for col in df.columns]
        create_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(columns),
        )
        with self.conn.cursor() as cur:
            cur.execute(create_query)
            logger.info(f"Table '{table_name}' created (if not exists).")

        insert_query = sql.SQL("INSERT INTO {} VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Placeholder() * len(df.columns)),
        )

        with self.conn.cursor() as cur:
            execute_batch(
                cur,
                insert_query.as_string(self.conn),
                df.values.tolist(),
                page_size=batch_size,
            )
        logger.info(f"Table '{table_name}' populated with {len(df)} rows.")

    def execute_query(self, query: str, params: tuple = None):
        """Run an arbitrary SQL query. Returns rows for SELECT; empty list for others."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise