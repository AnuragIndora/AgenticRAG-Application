from ollama_client import OllamaClient

class IntentRouterAgent:
    """
    Uses LLM to detect high-level intent of the query:
    - structured_data: SQL/database queries
    - document_qa: unstructured document questions
    - summarize: summarization tasks
    """

    def __init__(self):
        self.llm = OllamaClient()

    def detect_intent(self, query: str) -> str:
        """
        Returns intent as one of: structured_data, document_qa, summarize
        """
        prompt = f"""
Classify the user query into one category:
1. structured_data
2. document_qa
3. summarize

Return only one label in lowercase.

Query: {query}
"""
        try:
            result = self.llm.generate([{"role": "user", "content": prompt}])
            return result.strip().lower()
        except Exception:
            # fallback heuristic if LLM fails
            lower_q = query.lower()
            if any(keyword in lower_q for keyword in ["select", "from", "where", "join"]):
                return "structured_data"
            return "document_qa"