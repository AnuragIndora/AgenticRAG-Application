# postgres_client.py
import logging
import psycopg2
import pandas as pd
from psycopg2 import sql
from settings import get_settings

logger = logging.getLogger("PostgresClient")


class PostgresClient:
    def __init__(self):
        self.settings = get_settings()
        self.dbname = self.settings.pg_database

        self.conn = psycopg2.connect(
            host=self.settings.pg_host,
            port=self.settings.pg_port,
            dbname=self.settings.pg_database,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
        )
        self.conn.autocommit = True

        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (self.dbname,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.dbname)))
                logger.info(f"Database {self.dbname} created.")

        # Reconnect to the target database
        self.conn.close()
        self.conn = psycopg2.connect(
            host=self.settings.pg_host,
            port=self.settings.pg_port,
            dbname=self.dbname,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
        )
        self.conn.autocommit = True


    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame):
        columns = []
        for col in df.columns:
            columns.append(sql.SQL("{} TEXT").format(sql.Identifier(col)))

        create_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(columns)
        )

        with self.conn.cursor() as cur:
            cur.execute(create_query)

        for _, row in df.iterrows():
            insert_query = sql.SQL("INSERT INTO {} VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(sql.Placeholder() * len(row))
            )
            with self.conn.cursor() as cur:
                cur.execute(insert_query, list(row.astype(str)))

        logger.info(f"Table {table_name} created and populated.")

    def execute_query(self, query: str, params: tuple = None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
            return []