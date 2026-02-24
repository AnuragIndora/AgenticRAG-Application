from ollama_client import OllamaClient


class IntentRouterAgent:
    """
    Classifies a user query into one of three intents using the LLM.
    Falls back to a simple keyword check if the LLM call fails.
    """

    INTENTS = {"structured_data", "document_qa", "summarize"}

    def __init__(self):
        self.llm = OllamaClient()

    def detect_intent(self, query: str) -> str:
        """
        Returns one of: 'structured_data', 'document_qa', 'summarize'.
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
            # LLM unavailable — fall back to keyword heuristic
            lower_q = query.lower()
            if any(kw in lower_q for kw in ["select", "from", "where", "join"]):
                return "structured_data"
            return "document_qa"