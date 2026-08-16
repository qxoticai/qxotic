# jinfer in a few lines (jbang)

Single-file demos of what jinfer does that other local engines mostly don't. Every script is
runnable as-is and does its work **in-process** — no server, no Python, no JNI glue.

```bash
cd ../.. && mvn -q -DskipTests install    # publish jinfer to ~/.m2 once
jbang Chat.java "Explain HTTP/3 in two sentences."
```

The defaults are complete Hugging Face references, resolved through jinfer's model store. Pass a
local path, URL or another hub reference as the trailing model argument to override one. Downloaded
models are kept in jinfer's local model cache.

| script | shows | model |
|---|---|---|
| `Chat.java` | streaming chat, token by token | Llama-3.2-1B |
| `Json.java` | **grammar-constrained** JSON — malformed output is unrepresentable | Llama-3.2-1B |
| `Speak.java` | text to speech from a 4 MB model | Inflect-Nano-v2 |
| `Search.java` | semantic search, no vector database | Qwen3-Embedding-0.6B |
| `Rerank.java` | cross-encoder reranking, the second stage of RAG | Qwen3-Reranker-0.6B |
| `CachedPrompt.java` | prompt caching, with restored-token accounting | Llama-3.2-1B |
| `Detect.java` | object detection with the boxes **drawn** on the image | Gemma 4 12B + mmproj |

## Vision

`Detect.java` is the Gemma detection guide, ported and finished: the model returns normalized
0-1024 boxes as JSON, and the script rescales them onto the real pixels and writes `detected.png`.

```bash
jbang Detect.java photo.jpg "person, dog, bicycle"
```

Detection is prompt-driven — the same model that describes an image localizes in it, so there is no
detector to train, load or wire up. Two things that cost me time and are now handled:

- **Field order is not stable.** `box_2d` comes before `label` about as often as after, so the
  parser reads each object's fields independently instead of assuming a layout. The first version
  assumed one and silently drew zero boxes on a perfectly good detection.
- **Detection needs a bigger model than description does.** E2B labels correctly and places badly —
  asked for the llama and the mug, it named both right and put the llama's box inside the mug. 12B
  places both correctly from the same prompt and the same code, so `Detect.java` defaults to 12B.
  Every other vision script here is fine on E2B.

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

`Models.java` only holds the shared default model references — it is pulled in via `//SOURCES`.

`CachedPrompt.java` reports restored prompt tokens from the response itself. It intentionally avoids
a tiny wall-clock benchmark, where model loading, JIT warmup and the ordinary block cache obscure
the feature being demonstrated.

Native matmul is used automatically when `libjam` is on the path; otherwise the Java Vector backend
runs and everything still works, slower.
