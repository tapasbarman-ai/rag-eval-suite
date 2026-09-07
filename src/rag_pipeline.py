import os

from src.reranker import RerankingRetriever
from src.generator import GeminiGenerator


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5, model=None):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
        self.generator = GeminiGenerator(
            model=model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        )

    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        generated = self.generator.generate(query, context=context)

        # Keep both the text context and retrieved documents available to evals.
        return {
            "query": query,
            "context": context,
            "answer": generated["answer"],
            "retrieved_contexts": generated["retrieved_contexts"],
            "raw_retrieval": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("what is drift and why does it matter after deployment?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")