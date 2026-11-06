# jinfer

**Local AI inference for Python. Chat, vision, audio, embeddings, reranking, speech and transcription in one wheel: no server, no ML stack, no GPU needed.**

[![PyPI](https://img.shields.io/pypi/v/jinfer)](https://pypi.org/project/jinfer/)
[![Python](https://img.shields.io/pypi/pyversions/jinfer)](https://pypi.org/project/jinfer/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/qxoticai/qxotic/blob/main/LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)

`jinfer` is [jinfer](https://github.com/qxoticai/qxotic/tree/main/jinfer), the JVM inference engine, for Python: `libjinfer`, the engine, every ported model architecture and its native kernels compiled into one shared library with a C API, which the wheel carries.
A GGUF model runs in your process, on the CPU, from one `load` call, and comes back as a typed object with only the methods the file supports.
`toknroll` is the only dependency, for the tokenizer.

```sh
pip install jinfer
```

```python
import jinfer

model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")       # a LanguageModel; the file decides
reply = model.chat("Why is the sky blue?", max_tokens=200)   # already generating
for text in reply:                                           # stream it ...
    print(text, end="", flush=True)
reply.text, reply.usage.prompt_ms                            # ... or just read it

@dataclass
class Person:
    name: str
    born: int
person = model.chat("Ada Lovelace, born 1815.", schema=Person).parsed   # a Person, guaranteed

def weather(city: str) -> dict:
    """Current weather for a city."""
    return {"celsius": 18}
conversation = model.conversation(system="Be brief.", tools=[weather])
print(conversation.chat("Weather in Zurich?").text)          # runs weather, then answers
branch = conversation.fork(); conversation.rewind()          # free: nothing is recomputed
```

## Why jinfer

- **Three lines to a reply, and the reply is an object.** `load`, `chat`, `.text`; no messages list to assemble for a first question, no `["choices"][0]["message"]["content"]` to dig through, no server to start first.
- **Everything local in one package.** Chat, vision, audio in, embeddings, reranking, speech synthesis and transcription, each a method on the class the model file calls for, from one 19 MB wheel; no `torch`, no `transformers`, no daemon.
- **Structured output that cannot come back malformed.** `schema=Person` takes a dataclass, a TypedDict, a Pydantic class, `list[T]`, `Literal` or `Enum`, the engine enforces it while decoding, and `reply.parsed` is the Python value; there is nothing to validate and retry.
- **Tools are functions.** The signature is the schema and the docstring the description; a conversation runs the calls itself, between the words of the streamed answer.
- **The engine keeps the conversation.** A turn costs only its new tokens, `fork` and `rewind` recompute nothing, `prepare` warms a prompt before the user types, and a prompt cache file keeps it across processes.
- **Stopping is not losing.** Close a reply early and its partial text, usage and finish reason stay readable, in a conversation too.
- **Typed errors, keyword options, model defaults.** `InvalidArgument` is a `ValueError`, `NotFound` a `FileNotFoundError`; a typo in an option is a `TypeError` at the call; an option left out takes the model file's value, never the library's.
- **One engine, three faces.** The same code serves Java, this package, and [`jinfer.h`](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-native/include/jinfer.h), twenty-nine C functions any language calls.

| | jinfer | ollama | transformers | llama-cpp-python |
|---|---|---|---|---|
| Install | `pip install`, no ML stack | the Ollama app, a running server, `pip install ollama` | `transformers` plus `torch`, gigabytes | compiles llama.cpp, or a per-backend wheel index |
| Lines to a first reply | 3 | 3, plus two shell commands | 4 | 4 |
| Where the text is | `.text` | `.message.content` | `[0]["generated_text"][-1]["content"]` | `["choices"][0]["message"]["content"]` |
| Stop early | close; partial text, usage and finish reason stay | break; nothing stays | custom stopping criteria | break; nothing stays |
| Conversation state | an object; engine cache visible; fork, rewind | your list, re-sent each turn | your list, recomputed each turn | your list, last prefix reused |
| Structured output | any Python type, parsed object back, enforced | JSON schema in, validate yourself | none built in | JSON schema in, dict back, enforced |
| Tools | functions; the conversation runs the loop | functions; you run the loop | functions; you parse and loop | hand-written JSON; you parse and loop |
| Embeddings, reranking | `(n, dim)` float32 array; one call | list of lists; none | pooling by hand; cross-encoder by hand | list of lists; none |
| Speech, transcription | both, streaming, timed words | neither | both, no streaming | neither |
| Extra process | none | required | none | none |
| GPU | no | yes | yes | yes, chosen at install |
| Model breadth | the architectures jinfer ports | the Ollama library | everything on the Hub | every GGUF llama.cpp loads |

Where it trails, so you can decide: the CPU only, the architectures listed below rather than every GGUF, WAV or PCM into `transcribe`, and eight options on purpose, the rest left to the model file.
The [comparison](https://qxotic.ai/jinfer/python-comparison), use case by use case, has every snippet.

## Requirements

- Python 3.10 or later, and the `toknroll` wheel it pins, installed with it.
- The wheel carries the library: Linux x86-64 and aarch64 with glibc 2.28 or later, macOS 14 or later on Apple silicon, Windows 10 or later on x86-64.
  On x86-64 the library is built for x86-64-v3, a CPU with AVX2.
- `curl` on the PATH, to download a model named by a hub reference; a local file needs nothing.
- `ffmpeg` on the PATH, to decode an image or an audio file given to a chat; a WAV file for transcription needs nothing.
- `espeak-ng` (or `espeak`) on the PATH, for speech synthesis with Kokoro voices.
- RAM for the model file, which is read whole, plus its context.

## Models

Any GGUF of a ported architecture, in `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `MXFP4`, `NVFP4`, `Q1_0`, `TQ1_0`, `TQ2_0`, `F32`, `F16` or `BF16`; `Q8_0` is the recommended balance of quality and speed.

| Family | What the model does |
|---|---|
| Google Gemma 4 | chat, vision, audio in |
| Google EmbeddingGemma 2 | embeddings of text, images, audio and video |
| Liquid AI LFM 2.5 | chat, vision, embeddings, reranking |
| Liquid AI LFM3 | chat, vision, audio in and speech out (LFM3-Omni) |
| OpenAI gpt-oss | chat |
| Alibaba Qwen 3.5+ | chat, vision |
| Alibaba Qwen 3 | embeddings, reranking |
| Meta Llama 3+, IBM Granite 4.1+, Mistral AI Ministral 3, Hugging Face SmolLM3, OpenBMB MiniCPM5 | chat |
| Poolside Laguna XS 2.1, JetBrains Mellum 2, inclusionAI Ling 3, NVIDIA Nemotron-H | chat |
| Kokoro, Inflect | speech synthesis |
| NVIDIA Parakeet | speech recognition, with timed words |

The [jinfer README](https://github.com/qxoticai/qxotic/tree/main/jinfer#supported-architectures) keeps the authoritative table, with links to each family.

Models are hub references (`owner/repo:QUANT`, downloaded once and cached) or local paths.
`load` returns the class the file calls for, `LanguageModel`, `Embedder`, `Reranker`, `Speaker` or `Transcriber`, each with only the methods that apply.
Every call returns a plain typed object; nothing is a dict you have to know the shape of, and nothing in the API mentions tokens, JSON or OpenAI field names.

| You want | Call |
|---|---|
| a reply | `model.chat(prompt).text`; `print(reply)` prints the text |
| a streamed reply | `for text in model.chat(prompt)`, or `async for`; close the reply to cancel, what it produced stays readable |
| the rest of a reply | `reply.tool_calls`, `.reasoning`, `.finish_reason`, `.usage` (tokens, cached tokens, milliseconds) |
| a system prompt | `model.chat(prompt, system="You are terse.")` |
| options | `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `tools`, `tool_choice`, `reasoning_effort`; keyword-only, `None` is the model's default, a typo or a wrong type is a `TypeError` |
| structured output | `model.chat(prompt, schema=Person).parsed`: a dataclass, TypedDict, Pydantic class, `list[T]`, `Literal`, `Enum`, `dict` (free-form) or a JSON schema dict; it always parses, there is nothing to retry |
| tools | `tools=[fn]`: the signature is the schema, the docstring the description; a conversation runs the calls, a plain `chat` returns them as `reply.tool_calls`; `tool_choice` is `"auto"`, `"none"`, `"required"`, a function or its name |
| a conversation | `conversation = model.conversation(system=...)`; `.chat(text, schema=...)` streams like `model.chat`, `.fork()`, `.rewind()`, `.prepare()`, `.messages` |
| images, audio in | a `Path` or `bytes` in the content list: `model.chat(["What is this?", Path("photo.jpg")])` |
| embeddings | `embedder.embed(texts)` gives `(n, dimension)`, a single string counts as one; a text is embedded whole, up to `context_capacity`: 4096 tokens by default, or the model's maximum when smaller |
| reranking | `reranker.rerank(query, documents)` gives one score per document |
| speech | `speech = speaker.speak(text, speed=1.2)`; iterate PCM chunks, or `speech.audio`, `speech.save("out.wav")` |
| transcription | `transcriber.transcribe("speech.wav")` or `(pcm, sample_rate=16000)`; `.text` and timed `.words` |
| tokens | `model.tokenizer` is a [`toknroll.Tokenizer`](https://pypi.org/project/toknroll/): `encode(text)` gives an `array('i')`, `decode(ids)`, `count_tokens`, `encode_batch`, `vocabulary_size`, `token(id)`, `id(token)`; `toknroll.load(source)` gives one without the weights |
| token budgets | `len(model.tokenizer.encode(text))` against `model.context_capacity` (the state in use) and `model.max_context_length` (the model's bound) |
| the context size | `jinfer.load(source, context_capacity=8192)`: the tokens of context the model's states hold, up to its maximum; the default (`0`) is 4096, or the maximum when smaller, for every kind of model, `jinfer.MAXIMUM_CONTEXT` (`-1`) is the maximum, above it is refused |
| log lines | `logging.getLogger("jinfer")`, as many as `JINFER_LOG` lets through: `off`, `error`, `warn` (the default), `info` (downloads), `debug` (loads, kernels) |

Errors are `jinfer.JinferError` subclasses: `InvalidArgument` (also a `ValueError`), `NotFound` (also a `FileNotFoundError`), `Cancelled`; each has `.code`.
`jinfer.OpenAI(ref)` is a migration aid: code written for the OpenAI SDK's `chat.completions.create` and `embeddings.create` runs in-process, unchanged.

## Behaviour worth knowing

- A reply is already generating when `chat` returns. Iterating and the properties are two views of the same reply: read `.text` after a partial iteration and you get the whole reply.
- Closing a reply early cancels it, and the reply keeps what was produced: `.text` is the partial text, `.finish_reason` is `"cancelled"`, `.usage` counts what ran. In a conversation the partial text joins the history; a turn cancelled before its first word leaves no trace.
- A conversation turn with functions among the tools streams too: the calls run as you read the reply, between the words, and the text you iterate is the model's answer after them. `reply.usage` is the whole turn's.
- Temperature 0 gives the same tokens every run. Two different kernels can still land on different sides of a near-tie between two tokens, so a test pins behaviour, not prose.
- A conversation's turns are served from the engine's kept state, so a turn costs only its new tokens; `reply.usage.cached_tokens` shows it. A fork shares the prefix; a rewind recomputes nothing and keeps the system prompt; `prepare()` computes the prefix before the user types. With `prompt_cache=path`, what a model prepared is written when it closes and read by the next one.
- Structured output is enforced while decoding, so malformed JSON cannot come back. Validate meaning if you need to; there is no format to retry.
- Quiet by default: only warnings and errors reach `logging`.
  `JINFER_LOG=info` adds the downloads, at a terminal as jinfer's progress board, and as INFO records otherwise, which show once the application logs at INFO (`logging.basicConfig(level=logging.INFO)`); `JINFER_LOG=off` silences the library.
  Ctrl-C during a download ends the process; the download resumes next time.
  `repr(model)` names the kernels and threads that actually run.
- Any thread may call. A model may be shared between threads; a reply, speech or conversation belongs to one thread at a time. `async for` on a reply or speech steps in a worker thread; wrap one-shot calls in `asyncio.to_thread`.
- A model's weights stay alive while any reply or speech made from it lives; `close()` drops your reference. Context managers close; so does garbage collection.
- Embeddings and audio are float32 `array.array` buffers, scores are floats: the standard library, whoever installs the package. `np.asarray(vector)` and `torch.frombuffer(speech.audio, dtype=torch.float32)` are zero-copy.
- `jinfer.configure("-Djinfer.threads=8", "-Djinfer.cache.sessions=16", "-Djinfer.cache.budget=8g", "-Djinfer.offline=true", "-XX:MaxHeapSize=4g")` passes the runtime the options a java command line would, once and before the first load; the engine's `jinfer.*` and `jam.*` properties are all reachable this way. A later call is refused, and a rejected option fails this one. `JINFER_OPTIONS="-Djinfer.threads=4"` in the environment holds the same options and overrides them, so the box a program runs on can retune it without a code change.
- A scalar schema (`int`, `str`, `bool`, `Literal`, `Enum`) travels as `{"value": ...}` on the wire, so the model's first token is anchored; `reply.parsed` is the value, `reply.text` the JSON. Objects and lists go as they are.

## With toknroll

libjinfer carries its own toknroll: a model's tokenizer lives in jinfer's runtime, and `model.tokenizer` is a [`toknroll.Tokenizer`](https://pypi.org/project/toknroll/) served by libjinfer.
The toknroll wheel is a dependency for that class alone, pinned to the exact toknroll inside libjinfer; the class checks the library's version when it binds, so a mismatched environment fails at `import jinfer` naming both versions rather than later.
`jinfer.configure` and `JINFER_OPTIONS` configure the model's tokenizers too, `-Dtoknroll.*` options included, as they configure everything in libjinfer.
`toknroll.configure` and `TOKNROLL_OPTIONS` configure libtoknroll only, which jinfer never loads.
Using both wheels in one process is fine: `toknroll.load(...)` starts libtoknroll, a second runtime with its own heap and threads, the two never interact, the order they start in does not matter, and every tokenizer remembers the library serving it.
A knob both need, such as a GGUF pre-tokenizer registration, is given to both.

## Examples

One script a use case, in [`examples/`](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native/python/examples), each runnable as it is with the model it names, downloaded on the first run:

| Script | Model |
|---|---|
| `chat.py`, `stream.py`, `conversation.py`, `structured.py`, `tools.py`, `openai_client.py` | `LiquidAI/LFM2.5-350M-GGUF:Q8_0` |
| `vision.py` | `LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0` with `companions={"media": ...}`, the repository's `mmproj` file |
| `embed.py` | `LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0` |
| `rerank.py` | `mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0` |
| `speak.py` | `simonfxr/kokoro.cpp-GGUF:Q8_0` with `companions={"voice": ...}`, one of the repository's voice files |
| `transcribe.py` | `mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf` |

A companion is a second file a model needs, named at load: `jinfer.load(ref, companions={"media": "<ref>"})` for a vision model's projector, `{"voice": "<ref>"}` for a speaker's voice.
The architectures the library carries are jinfer's ports, listed in the [jinfer documentation](https://qxotic.ai/jinfer).

## From a checkout

`make python` in `jinfer/jinfer-native` stages the freshly built library into the package, then `pip install python/`.
`JINFER_LIBRARY=/path/to/libjinfer.so` points the package at any build.
`pydantic` is optional.
The C API the package binds is [`jinfer.h`](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-native/include/jinfer.h), with `toknroll.h` for the tokenizer, usable from any language with a foreign function interface.
## Links


- [jinfer](https://github.com/qxoticai/qxotic/tree/main/jinfer): the engine, the ports, the Java API, the server and the framework providers.
- [libjinfer](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native): the shared library, its C API [`jinfer.h`](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-native/include/jinfer.h), this package and its [examples](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native/python/examples).
- [Documentation](https://qxotic.ai/jinfer), the [Python comparison](https://qxotic.ai/jinfer/python-comparison), and [issues](https://github.com/qxoticai/qxotic/issues).
- Apache 2.0, by [Quixotic AI](https://qxotic.ai).
