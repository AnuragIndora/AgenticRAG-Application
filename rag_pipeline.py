from typing import List
from milvuous_client import MilvusVectorStore
from ollama_client import OllamaClient
from models import RetrievalResult

class RAGPipeline:
    def __init__(self):
        self.milvus = MilvusVectorStore()
        self.ollama = OllamaClient()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top_k chunks from Milvus based on query embedding."""
        query_emb = self.ollama.embed(query)
        results = self.milvus.search(query_emb, top_k=top_k)
        # Optionally rerank using keyword overlap
        results = self.milvus.keyword_filter(results, query)
        return results

    def generate_answer(self, query: str, top_k: int = 5) -> str:
        """Generate answer using retrieved chunks."""
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
        prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.ollama.generate([{"role": "user", "content": prompt}])
    