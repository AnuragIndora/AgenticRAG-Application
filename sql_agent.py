# # sql_agent.py
# from ollama_client import OllamaClient
# from postgres_client import PostgresClient

# class SQLAgent:
#     def __init__(self):
#         self.llm = OllamaClient()
#         self.db = PostgresClient()

#     def generate_sql(self, user_query: str) -> str:
#         """
#         Generates a safe SQL SELECT query from natural language.
#         Only SELECT queries or WITH (CTE) are allowed.
#         """
#         prompt = f"""
# You are a PostgreSQL expert.
# Generate ONLY a valid SQL query for this question.
# - MUST be a SELECT query or WITH/SELECT (CTE). 
# - Never use DELETE, UPDATE, INSERT, or DROP.
# - Only reference tables in the public schema.
# - Return query only, no explanations.

# Question:
# {user_query}
# """
#         response = self.llm.generate([{"role": "user", "content": prompt}])
#         sql_query = response.strip()

#         # Strip Markdown code fences if present
#         if sql_query.startswith("```"):
#             sql_query = "\n".join(sql_query.split("\n")[1:-1]).strip()

#         # Safety check: only allow SELECT or WITH
#         sql_lower = sql_query.lower()
#         allowed_prefixes = ["select", "with"]
#         if not any(sql_lower.startswith(p) for p in allowed_prefixes):
#             raise ValueError(f"Only SELECT queries are allowed. Generated SQL: {sql_query}")

#         # Optional: validate referenced tables exist
#         existing_tables_rows = self.db.execute_query(
#             "SELECT tablename FROM pg_tables WHERE schemaname='public';"
#         )
#         existing_tables = [t[0].lower() for t in existing_tables_rows]

#         # crude parse: check table names appear in SQL
#         referenced_tables = [
#             word.strip('"') for word in sql_lower.replace(",", " ").split() if word.isidentifier()
#         ]
#         if not any(table in existing_tables for table in referenced_tables):
#             raise ValueError(f"No referenced table exists in DB. Generated SQL: {sql_query}")

#         return sql_query

#     def handle_query(self, query: str) -> str:
#         sql_query = self.generate_sql(query)

#         # Execute query using PostgresClient
#         result = self.db.execute_query(sql_query)

#         # Convert SQL result into a human-readable answer using LLM
#         formatted_prompt = f"""
# User Question: {query}

# SQL Query Used:
# {sql_query}

# SQL Result:
# {result}

# Provide a concise and clear answer to the user.
# """
#         return self.llm.generate([{"role": "user", "content": formatted_prompt}])

#     def delete_all_tables(self):
#         """
#         Drop all tables in the current database.
#         WARNING: This deletes all data permanently!
#         """
#         # Fetch all table names
#         tables = self.db.execute_query(
#             "SELECT tablename FROM pg_tables WHERE schemaname='public';"
#         )

#         for table_name_row in tables:
#             table_name = table_name_row[0]  # each row is a tuple
#             self.db.execute_query(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')


#sql_agents.py
from ollama_client import OllamaClient
from postgres_client import PostgresClient

class SQLAgent:
    def __init__(self):
        self.llm = OllamaClient()
        self.db = PostgresClient()

    def generate_sql(self, user_query: str) -> str:
        """
        Generates a safe SQL SELECT query from natural language.
        Only SELECT queries or WITH (CTE) are allowed.
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

        # Strip Markdown code fences if present
        if sql_query.startswith("```"):
            sql_query = "\n".join(sql_query.split("\n")[1:-1]).strip()

        # Safety check: only allow SELECT or WITH
        sql_lower = sql_query.lower()
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            raise ValueError(f"Only SELECT queries are allowed. Generated SQL: {sql_query}")

        # Optional: validate referenced tables exist
        existing_tables_rows = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )
        existing_tables = [t[0].lower() for t in existing_tables_rows]

        # crude parse: check table names appear in SQL
        referenced_tables = [
            word.strip('"') for word in sql_lower.replace(",", " ").split() if word.isidentifier()
        ]
        if not any(table in existing_tables for table in referenced_tables):
            raise ValueError(f"No referenced table exists in DB. Generated SQL: {sql_query}")

        return sql_query

    def handle_query(self, query: str) -> str:
        """
        Generate SQL, execute it, and convert result into a human-readable answer.
        """
        sql_query = self.generate_sql(query)

        # Execute query using PostgresClient
        result = self.db.execute_query(sql_query)

        # Convert SQL result into a human-readable answer using LLM
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
        Drop all tables in the current database.
        WARNING: This deletes all data permanently!
        """
        # Fetch all table names
        tables = self.db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname='public';"
        )

        for table_name_row in tables:
            table_name = table_name_row[0]  # each row is a tuple
            self.db.execute_query(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')