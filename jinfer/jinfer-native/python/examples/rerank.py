"""Reranking: score documents against a query with a reranker model."""
import jinfer

with jinfer.load("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0") as reranker:
    query = "capital of France"
    documents = ["Paris is the capital and largest city of France.", "Bananas are yellow when ripe.", "The Seine flows through Paris."]
    for score, doc in sorted(zip(reranker.rerank(query, documents), documents), reverse=True):
        print(f"{score:6.3f}  {doc}")
