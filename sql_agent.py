from ollama_client import OllamaClient
from postgres_client import PostgresClient


class SQLAgent:
    """Translates natural language questions into safe SELECT queries and executes them."""

    def __init__(self):
        self.llm = OllamaClient()
        self.db = PostgresClient()

    def generate_sql(self, user_query: str) -> str:
        """
        Ask the LLM to produce a safe SQL query, then validate it before returning.
        Only SELECT and WITH-SELECT (CTE) queries are allowed through.
        Also verifies that at least one referenced table actually exists in the DB.
        """
        prompt = f"""
You are a PostgreSQL expert.
Generate ONLY a valid SQL query for this question.
- MUST be a SELECT query or WITH/SELECT (CTE).
- Never use DELETE, UPDATE, INSERT, or DROP.
- Only reference tables in the public schema.
- Return query only, no explanations.

Question:
{user_query}
"""
        response = self.llm.generate([{"role": "user", "content": prompt}])
        sql_query = response.strip()

        # Strip Markdown code fences if the LLM wrapped the query in them
        if sql_query.startswith("```"):
            sql_query = "\n".join(sql_query.split("\n")[1:-1]).strip()

        # Reject anything that isn't a read-only query
        sql_lower = sql_query.lower()
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            raise ValueError(f"Only SELECT queries are allowed. Generated SQL: {sql_query}")

        # Cross-check referenced identifiers against actual tables in the DB
        existing_tables_rows = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )
        existing_tables = [t[0].lower() for t in existing_tables_rows]

        # Crude but fast: tokenize the SQL and see if any token matches a real table name
        referenced_tables = [
            word.strip('"') for word in sql_lower.replace(",", " ").split() if word.isidentifier()
        ]
        if not any(table in existing_tables for table in referenced_tables):
            raise ValueError(f"No referenced table exists in DB. Generated SQL: {sql_query}")

        return sql_query

    def handle_query(self, query: str) -> str:
        """Generate SQL, execute it, and use the LLM to summarize the result in plain English."""
        sql_query = self.generate_sql(query)
        result = self.db.execute_query(sql_query)

        formatted_prompt = f"""
User Question: {query}

SQL Query Used:
{sql_query}

SQL Result:
{result}

Provide a concise and clear answer to the user.
"""
        return self.llm.generate([{"role": "user", "content": formatted_prompt}])

    def delete_all_tables(self):
        """
        Drop every table in the public schema.
        WARNING: This permanently deletes all structured data!
        """
        tables = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )
        for (table_name,) in tables:
            self.db.execute_query(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')