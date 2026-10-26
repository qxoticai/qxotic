"""Embeddings for search: one vector per text, cosine similarity in two lines."""
import math, jinfer

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

with jinfer.load("LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0") as embedder:
    texts = ["AI on the JVM", "the JVM thinks now", "the python shed its skin"]
    vectors = embedder.embed(texts)                  # three float32 arrays; np.asarray(vectors) is the matrix
    query = embedder.embed("Java machine learning")[0]  # a list of one -> the vector
    print(f"{len(texts)} vectors of {embedder.dimension}")
    for text, vector in zip(texts, vectors):
        print(f"{cosine(query, vector):.3f}  {text}")
