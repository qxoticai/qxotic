# jinfer in a few lines (jbang)

Single-file demos of what jinfer does that other local engines mostly don't. Every script is
runnable as-is and does its work **in-process** — no server, no Python, no JNI glue.

```bash
cd ../.. && ./mvnw -q -DskipTests install    # publish jinfer to ~/.m2 once
export JINFER_MODELS=/path/to/models         # default: ~/models
jbang Chat.java "Explain HTTP/3 in two sentences."
```

Each script takes the model path as a trailing argument if you'd rather be explicit than set
`JINFER_MODELS`. Model layout matches `scripts/download-models.sh` (`{source}/{user}/{repo}/{file}`).

| script | shows | model |
|---|---|---|
| `Chat.java` | streaming chat, token by token | Llama-3.2-1B |
| `Json.java` | **grammar-constrained** JSON — malformed output is unrepresentable | Llama-3.2-1B |
| `Speak.java` | text to speech from a 4 MB model | Inflect-Nano-v2 |
| `Search.java` | semantic search, no vector database | Qwen3-Embedding-0.6B |
| `Rerank.java` | cross-encoder reranking, the second stage of RAG | Qwen3-Reranker-0.6B |
| `CachedPrompt.java` | prompt caching, with a measured speedup | Llama-3.2-1B |

## The two worth reading first

**`Json.java`** — the grammar constrains *sampling*, so the model cannot emit a token that breaks
the schema. Not "please reply in JSON" plus a retry loop; the invalid output does not exist.

```
{"name": "Ada Lovelace", "year": 1815, "city": "London"}
```

**`Rerank.java` next to `Search.java`** — the same corpus, two retrieval stages. Embeddings score
the whole corpus quickly but flatly (0.518 / 0.482 / 0.480); the cross-encoder reads query and
document together and separates decisively (0.3675 / 0.0007 / 0.0004). That gap is why real RAG
pipelines use both.

## Notes

`Models.java` is a shared path helper, not a demo — it is pulled in via `//SOURCES`.

`CachedPrompt.java` warms the JIT before timing and compares like with like (same questions, same
warm JVM, only the prefix handling differs). Caching never changes the answer; byte-identity with an
uncached run is a project law, so the numbers are purely about cost.

Native matmul is used automatically when `libjam` is on the path; otherwise the Java Vector backend
runs and everything still works, slower.
