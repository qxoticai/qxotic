---
sidebar_position: 4
---

# libjinfer and Python

**jinfer from any language.** `libjinfer` is jinfer as a shared library with a C API: one header, `jinfer.h`, with twenty-nine functions, plus `toknroll.h`, the tokenizer API it includes; a Python package is the first binding.
No server, no Java visible, no Python ML stack: the library carries the engine, every model architecture and the jam kernels.

```python
import jinfer

model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
print(model.chat("What is the capital of France?").text)
for text in model.chat("Tell me a story.", max_tokens=200):
    print(text, end="", flush=True)

chat = model.conversation(system="You are terse.")
chat.chat("My name is Zoltan.")
print(chat.chat("What is my name?").text)     # only the new tokens are computed
```

## The shape

Three ideas, and the whole API follows from them.

- **A model is whatever the file is.** `jinfer_model_load` takes a hub reference or a path and the model selects its face: language, embedding, reranking, speech or transcription. `jinfer_model_property` says what it can do; `jinfer_model_info` describes it as JSON.
- **Every generation is a stream, stepped like a statement.** `jinfer_chat`, `jinfer_speak` and `jinfer_transcribe` return a `jinfer_stream`; `jinfer_stream_step` blocks and says whether a chunk is ready, `jinfer_stream_text` or `jinfer_stream_samples` is the chunk, `jinfer_stream_result` the finished result as JSON. `jinfer_stream_cancel` may be called from any thread; the blocked step returns `JINFER_ECANCELLED` and the result still carries the text produced so far and counts what ran, with `finish_reason` `"cancelled"`. Freeing a running stream cancels it. Blocking is `while (jinfer_stream_step(s) == JINFER_OK) {}`.
- **OpenAI JSON in and out.** `jinfer_speak` takes the `/v1/audio/speech` fields that apply (`input`, `speed`); `jinfer_chat` takes exactly the jinfer server's `/v1/chat/completions` request and returns its response: messages, tools, `response_format` with a JSON schema, images and audio as data URLs. The same code lowers both; a reply constrained to a schema cannot come back malformed.

Numbers never travel as JSON: tokens, logits, embeddings, scores and PCM cross as caller buffers that follow `snprintf` (write at most `capacity`, return the count needed).
Everything returned is borrowed from its handle until the next call on that handle; a model stays alive while any state or stream made from it lives.
Errors: pointer-returning functions return NULL, int functions a negative code, `jinfer_last_error(&code)` says what happened, the message and the code.

The raw door, for callers who sample themselves: `jinfer_state_new`, a state of your own holding up to `context_capacity` tokens; `jinfer_state_ingest` (the forward pass), `jinfer_state_logits`, `jinfer_state_embedding`.
Tokenization is toknroll's API, `toknroll.h`, which `jinfer.h` includes: `jinfer_model_tokenizer` hands out the model's tokenizer as a `toknroll_tokenizer` (`toknroll_encode`, `toknroll_decode`, `toknroll_token`, `toknroll_id`, `toknroll_vocabulary_size`), and `toknroll_load` gives one from a tokenizer.json, a GGUF file or a hub reference without the weights.

## Python

The package wraps the header and adds what Python expects, and nothing else.
`load` returns a `LanguageModel`, `Embedder`, `Reranker`, `Speaker` or `Transcriber` with only the methods that apply.
Results are typed objects instead of dicts, a reply is both the stream and the result, options are keyword-only with `None` for the model's default, log lines go to `logging`, errors form a hierarchy under `jinfer.JinferError`.
Python types go in and Python values come out: `schema=` takes a dataclass, a TypedDict, a Pydantic class, `list[T]`, `Literal` or `Enum`, and a function is a tool.
What is only possible locally gets a face of its own: a `Conversation` whose turns are served from the engine's kept state (`chat`, `fork`, `rewind`, `prepare`).
There are no prompt templates, chains, memory classes, callbacks, retries or option objects.

| You want | Call |
|---|---|
| a reply | `model.chat(prompt).text` |
| a streamed reply | `for text in model.chat(prompt)`, or `async for` |
| the rest of a reply | `reply.tool_calls`, `.reasoning`, `.usage`, `.finish_reason` |
| a conversation | `chat = model.conversation(system=...)`; `chat.chat(text)`, `chat.fork()`, `chat.rewind()`, `chat.prepare()` |
| structured output | `model.chat(prompt, schema=Person).parsed`, a dataclass, TypedDict, Pydantic class, `list[T]`, `Literal`, `Enum` or a JSON schema dict |
| tools | `tools=[fn]`; a conversation runs the calls, a plain `chat` returns them as `reply.tool_calls` |
| images, audio in | a `Path` or `bytes` in the content list: `model.chat(["What is this?", Path("photo.jpg")])` |
| embeddings, reranking | `embedder.embed(texts)`, `reranker.rerank(query, documents)` |
| speech, transcription | `speaker.speak(text).save("out.wav")`, `transcriber.transcribe("speech.wav").words` |
| a warm prompt cache | `chat.prepare()` once; `prompt_cache="catalog.jkvf"` at load keeps it across processes |
| the context size | `jinfer.load(ref, context_capacity=8192)`, up to `model.max_context_length`; the default is 4096 or the maximum when smaller |
| log lines | `logging.getLogger("jinfer")` |

`jinfer.OpenAI(model_ref)` replaces `OpenAI(api_key=...)` as a migration aid; code written for the OpenAI SDK then runs in-process, unchanged.

## Build

From `jinfer/jinfer-native`, with GraalVM 25.0.4 or newer and a C compiler:

```sh
make libjinfer            # -O3 -march=x86-64-v3 (GraalVM's default on x86-64, AVX2): target/libjinfer.so; MARCH=x86-64-v4 for AVX-512
make check                # header vs exports, the C example, the Python test
make python               # stage the library into the Python package; then pip install python/
```

The library links jam statically: nothing is extracted at run time.
An image is built for one ISA level, `-march`, and the kernels' Vector API width is pinned to that level at build time, never inherited from the build machine; x86-64-v3 (AVX2) is the floor.
Model options are a versioned struct (`jinfer_model_options_init`, then set fields): companions such as `media=<ref>` or `voice=<ref>`, a prompt-cache catalog, and `context_capacity`, the tokens of context the model's states hold (0 = the default, -1 = the model's maximum, above it refused).
A model is the weights and configuration plus the states chat and retrieval run in; `JINFER_CONTEXT_CAPACITY` reports what they hold and `JINFER_MAX_CONTEXT_LENGTH` the model's bound.
The runtime is configured once, before the first call, with `jinfer_configure`, a second call being refused: the options a java command line takes, `-Djinfer.threads=8` for the process's pool, `-Djinfer.cache.sessions=16` and `-Djinfer.cache.budget=8g` for the prompt cache's warm conversations and KV-block bytes (4 and 2g by default, budget 0 switches the block layer off), `-Djinfer.offline=true`, any `jinfer.*` or `jam.*` property the engine reads, `-XX:MaxHeapSize` for the heap; the `JINFER_OPTIONS` environment variable holds the same options and overrides them, so a deployment retunes a program without a code change; `jinfer_model_info` reports the pool size in effect.
