from typing import List

from milvus_client import MilvusVectorStore
from ollama_client import OllamaClient
from models import RetrievalResult


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Fetches a wide candidate pool from Milvus, re-ranks by keyword overlap, and
    returns only the top-k most relevant chunks.
    """

    def __init__(self):
        self.milvus = MilvusVectorStore()
        self.ollama = OllamaClient()

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[RetrievalResult]:
        """
        Two-stage retrieval:
        1. ANN search for `candidate_k` approximate neighbors.
        2. Re-rank by keyword overlap and keep only the top `top_k`.
        """
        query_emb = self.ollama.embed(query)
        candidate_results = self.milvus.search(query_emb, top_k=candidate_k)
        reranked_results = self.milvus.keyword_filter(candidate_results, query)
        return reranked_results[:top_k]

    def generate_answer(self, query: str, top_k: int = 5, candidate_k: int = 20) -> str:
        """Retrieve relevant context and generate a grounded answer using the LLM."""
        retrieved_chunks = self.retrieve(query, top_k=top_k, candidate_k=candidate_k)
        context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
        prompt = (
            f"Answer the following question using the context below:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        return self.ollama.generate([{"role": "user", "content": prompt}])