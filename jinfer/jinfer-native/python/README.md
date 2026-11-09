# jinfer

**Local AI inference for Python: chat, vision, audio, embeddings, speech and transcription in one wheel, no server, no ML stack.**

[![PyPI](https://img.shields.io/pypi/v/jinfer)](https://pypi.org/project/jinfer/)
[![Python](https://img.shields.io/pypi/pyversions/jinfer)](https://pypi.org/project/jinfer/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/qxoticai/qxotic/blob/main/LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)

`jinfer` runs GGUF models in your own process, on the CPU, from one `load` call, and every answer comes back as a plain typed object.
One 18 MB wheel with the engine inside: no server to start, no `torch`, no `transformers`, and only `toknroll` beside it, for the tokenizer.

```sh
pip install jinfer
```

```python
from dataclasses import dataclass
import jinfer

model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")    # downloaded once, then cached
reply = model.chat("Why is the sky blue?", max_tokens=200)
for text in reply:                                        # stream it...
    print(text, end="", flush=True)
reply.text, reply.usage.completion_tokens                 # ...or read it whole

@dataclass
class Person:
    name: str
    born: int

model.chat("Ada Lovelace, born 1815.", schema=Person).parsed
# Person(name='Ada Lovelace', born=1815)

def weather(city: str) -> dict:
    """Current weather for a city."""
    return {"celsius": 18}

chat = model.conversation(system="Be brief.", tools=[weather])
chat.chat("Weather in Zurich?").text                      # runs weather, then answers
# 'The current temperature in Zurich is 18 degrees Celsius.'
```

## Why jinfer

- **Everything local, in one package.** Chat, vision, audio in, embeddings, reranking, speech synthesis and transcription, each a method on the class the model file calls for; no server, no daemon, no GPU needed.
- **Three lines to a reply, and the reply is an object.** `load`, `chat`, `.text`: no messages list to assemble for a first question, no `["choices"][0]["message"]["content"]` to dig through.
- **Structured output that cannot come back malformed.** `schema=` takes a dataclass, a TypedDict, a Pydantic class, `list[T]`, `Literal` or `Enum`; the engine enforces it while decoding, and `.parsed` is the Python value, with nothing to validate or retry.
- **Tools are functions.** The signature is the schema and the docstring the description; a conversation runs the calls itself, between the words of the streamed answer.
- **The engine keeps the conversation.** A turn costs only its new tokens, `fork` and `rewind` recompute nothing, `prepare` warms a prompt before the user types, and a prompt cache file keeps it across processes.
- **Nothing else to install.** The engine, every ported model and their kernels are compiled ahead of time with GraalVM Native Image into a plain shared library, so no JVM either; [`toknroll`](https://pypi.org/project/toknroll/) is the one dependency.

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
The [comparison](https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/python-comparison.md), use case by use case, has every snippet.

## Requirements

- Python 3.10 or later, and the `toknroll` wheel it pins, installed with it; `pydantic` only to pass a Pydantic class as a schema.
- Linux x86-64 and aarch64 with glibc 2.28 or later, macOS 14 or later on Apple silicon, Windows 10 or later on x86-64: the wheel carries the library.
- On x86-64, a CPU with AVX2 (Intel Haswell, AMD Excavator or later); an older CPU, or a virtual machine that hides AVX2, is refused at load, naming what it lacks.
- RAM for the model file, which is read whole, plus its context.
- On the PATH: `curl` to download a model by Hub reference, `ffmpeg` to decode an image or an audio file given to a chat, `espeak-ng` (or `espeak`) for speech with Kokoro voices; a local file, a WAV file or another voice needs none of them.

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

A model is a Hub reference (`owner/repo:QUANT`, downloaded once and cached) or a local path, and `load` returns the class the file calls for: `LanguageModel`, `Embedder`, `Reranker`, `Speaker` or `Transcriber`, each with only the methods that apply.
The [jinfer README](https://github.com/qxoticai/qxotic/tree/main/jinfer#supported-architectures) keeps the authoritative table.

## Usage

| You want | Call |
|---|---|
| a reply | `model.chat(prompt).text`; `print(reply)` prints the text |
| a streamed reply | `for text in model.chat(prompt)`, or `async for`; closing the reply cancels it, and what it produced stays readable |
| the rest of a reply | `reply.tool_calls`, `.reasoning`, `.finish_reason`, `.usage` (tokens, cached tokens, milliseconds) |
| a system prompt | `model.chat(prompt, system="You are terse.")` |
| options | `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `tools`, `tool_choice`, `reasoning_effort`: keyword-only, `None` is the model's default, and a typo is a `TypeError` |
| structured output | `model.chat(prompt, schema=Person).parsed`: a dataclass, TypedDict, Pydantic class, `list[T]`, `Literal`, `Enum`, `dict` or a JSON schema dict |
| tools | `tools=[fn]`; a conversation runs the calls, a plain `chat` returns them as `reply.tool_calls`; `tool_choice` is `"auto"`, `"none"`, `"required"`, a function or its name |
| a conversation | `model.conversation(system=...)`, with `.chat(...)`, `.fork()`, `.rewind()`, `.prepare()`, `.messages` |
| prepared prompts that outlive the process | `jinfer.load(ref, prompt_cache="prompts.cache")`: read at load, extended at close |
| images, audio in | a `Path` or `bytes` in the content list: `model.chat(["What is this?", Path("photo.jpg")])` |
| embeddings | `embedder.embed(texts)` gives an `(n, dimension)` float32 array; a single string counts as one |
| reranking | `reranker.rerank(query, documents)` gives one score per document |
| speech | `speaker.speak(text, speed=1.2)`: iterate its PCM chunks, or `.audio`, or `.save("out.wav")` |
| transcription | `transcriber.transcribe("speech.wav")`, or `(pcm, sample_rate=16000)`: `.text` and timed `.words` |
| tokens | `model.tokenizer`, a [`toknroll.Tokenizer`](https://pypi.org/project/toknroll/): `model.tokenizer.count_tokens(text)` against `model.context_capacity` |
| the context size | `jinfer.load(ref, context_capacity=8192)`; the default is 4096, or the model's maximum when smaller, and `jinfer.MAXIMUM_CONTEXT` asks for the maximum |

Errors are `jinfer.JinferError` subclasses with a `.code`: `InvalidArgument` (also a `ValueError`), `NotFound` (also a `FileNotFoundError`) and `Cancelled`.
`jinfer.OpenAI(ref)` runs code written for the OpenAI SDK's `chat.completions.create` and `embeddings.create` in process, unchanged.

## Good to know

- A reply is already generating when `chat` returns; iterating it and reading `.text` are two views of the same reply.
- Closing a reply early keeps what it produced: the partial text, a `"cancelled"` finish reason and the usage, in a conversation too.
- Temperature 0 gives the same tokens every run on one machine; two kernels can still land on different sides of a near-tie, so a test pins behaviour, not prose.
- Any thread may call: a model may be shared, a reply or conversation belongs to one thread at a time, and `async for` steps in a worker thread.
- Embeddings and audio are float32 `array.array` buffers: `numpy.asarray(vector)` and `torch.frombuffer(speech.audio, dtype=torch.float32)` read them without a copy.
- `jinfer.configure("-Djinfer.threads=8", "-XX:MaxHeapSize=4g")` sets the runtime's options once, before the first load; `JINFER_OPTIONS` in the environment holds the same options and wins.
- Quiet by default: log lines go to `logging.getLogger("jinfer")`, as many as `JINFER_LOG` lets through (`off`, `error`, `warn` by default, `info`, `debug`), and a download's progress to stderr unless `JINFER_PROGRESS=off`.

## Examples

One script a use case, in [`examples/`](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native/python/examples), each runnable as it is with the model it names:

| Script | Model |
|---|---|
| `chat.py`, `stream.py`, `conversation.py`, `structured.py`, `tools.py`, `openai_client.py` | `LiquidAI/LFM2.5-350M-GGUF:Q8_0` |
| `vision.py` | `LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0`, with its `mmproj` file as `companions={"media": ...}` |
| `embed.py` | `LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0` |
| `rerank.py` | `mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0` |
| `speak.py` | `simonfxr/kokoro.cpp-GGUF:Q8_0`, with a voice file as `companions={"voice": ...}` |
| `transcribe.py` | `mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf` |

## From C

The wheel is the C library too: `jinfer.h`, `toknroll.h`, `libjinfer` and a pkg-config file live in the installed package.
`jinfer-config --cflags --libs` prints the compiler flags, one argument per line, and `--pkgconfigdir` the directory for `PKG_CONFIG_PATH`.
The [libjinfer README](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native) covers Windows and the twenty-nine functions of [`jinfer.h`](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-native/include/jinfer.h).

## With toknroll

`model.tokenizer` is a `toknroll.Tokenizer`, served by the toknroll built into libjinfer; the `jinfer` wheel pins the `toknroll` version that matches it, and `jinfer.configure` reaches its `-Dtoknroll.*` options.
Both wheels can share a process; their libraries never interact.

## Links

- [jinfer](https://github.com/qxoticai/qxotic/tree/main/jinfer): the engine, the ports, the Java API, the server and the framework providers.
- [libjinfer](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native): the shared library and its C API.
- [The Python comparison](https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/python-comparison.md) and the [issue tracker](https://github.com/qxoticai/qxotic/issues).

## License

Apache 2.0, by [Quixotic AI](https://qxotic.ai), for the package and the library's own code.
The library in the wheel is built with Oracle GraalVM and carries its Native Image runtime under the [GraalVM Free Terms and Conditions](https://www.oracle.com/downloads/licenses/graal-free-license.html), spelled out in the wheel's NOTICE.
Use it as you like, commercially too; a product sold for a fee that bundles a copy of it is not covered by those terms, and takes a build with GraalVM CE, see the [libjinfer README](https://github.com/qxoticai/qxotic/tree/main/jinfer/jinfer-native#license).
