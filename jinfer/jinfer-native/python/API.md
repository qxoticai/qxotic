# jinfer Python API

The full reference of the `jinfer` package.
The [README](README.md) is the short introduction; the [comparison](../../../docs/jinfer/python-comparison.md) sets jinfer beside ollama, transformers and llama-cpp-python, use case by use case.

## Loading a model

`jinfer.load(source)` takes a Hub reference or a local path and returns the class the model file calls for, each with only the methods that apply:

| Class | The model | Main call |
|---|---|---|
| `LanguageModel` | chat, vision, audio input | `chat`, `conversation` |
| `Embedder` | embeddings | `embed` |
| `Reranker` | reranking | `rerank` |
| `Speaker` | speech synthesis | `speak` |
| `Transcriber` | speech recognition | `transcribe` |

- A Hub reference is `[host/]owner/repo[@rev][/path][:quant]`, `LiquidAI/LFM2.5-350M-GGUF:Q8_0` for example, downloaded once and cached; `modelscope.cn/...` reads ModelScope.
- A URL is refused, naming the reference to pass: `InvalidArgument`.
- `companions={"media": ...}` attaches a vision or audio projector (`mmproj`), `companions={"voice": ...}` a speech voice.
- `context_capacity=8192` sets the context; the default is 4096, or the model's maximum when smaller, and `jinfer.MAXIMUM_CONTEXT` asks for the maximum.
- `prompt_cache="prompts.cache"` keeps prepared prompts across processes: the file is read at load, and extended when the model is closed and no reply is still generating.
- `load` returns the class the file calls for, so name it for a type checker: `model: jinfer.LanguageModel = jinfer.load(ref)`.

## Chat

`model.chat(prompt)` returns a reply that is already generating; iterating it and reading `.text` are two views of the same reply.

| You want | Call |
|---|---|
| the text | `reply.text`; `print(reply)` prints it |
| a stream | `for text in reply`, or `async for`, which steps in a worker thread |
| the rest | `reply.tool_calls`, `.reasoning`, `.finish_reason`, `.usage` (tokens, cached tokens, milliseconds) |
| a system prompt | `model.chat(prompt, system="You are terse.")` |
| images or audio input | a `Path` or `bytes` in the content list: `model.chat(["What is this?", Path("photo.jpg")])`; a `str` is text |

Options are keyword-only, `None` is the model's default, and a typo is a `TypeError`:

| Option | Meaning |
|---|---|
| `max_tokens` | the reply's budget; `-1` runs to the end of the context |
| `temperature`, `top_p`, `seed` | sampling |
| `stop` | strings that end the reply |
| `tools`, `tool_choice` | see [Tools](#tools) |
| `reasoning_effort` | on or off: `"none"` is off, any level is on |
| `schema` | see [Structured output](#structured-output) |

Closing a reply early cancels it and keeps what it produced: the partial text, a `"cancelled"` finish reason and the usage, in a conversation too.

## Structured output

`model.chat(prompt, schema=Person).parsed` is a Python value of the schema's type.
`schema=` takes a dataclass, a TypedDict, a Pydantic class (with `pydantic` installed), `list[T]`, `Literal`, `Enum`, `dict` or a JSON schema dict, recursive types included.
The engine enforces the schema while decoding, so there is nothing to validate or retry.

## Tools

A tool is a plain function: its signature is the schema and its docstring the description.

```python
def weather(city: str) -> dict:
    """Current weather for a city."""
    return {"celsius": 18}
```

- In a conversation, `tools=[weather]` runs the calls itself, between the words of the streamed answer.
- A plain `chat` returns them as `reply.tool_calls` for you to run.
- `tool_choice` is `"auto"`, `"none"`, `"required"`, a function or its name.

## Conversations

`chat = model.conversation(system=..., tools=[...])` keeps the conversation in the engine, so a turn costs only its new tokens.

- `chat.chat(text)` takes a turn and returns its reply.
- `chat.fork()` branches the conversation and `chat.rewind()` takes back the last turn, recomputing nothing.
- `chat.prepare()` warms the prompt before the user types.
- `chat.messages` is the history; a turn that fails leaves it as it was.

## Embeddings and reranking

- `embedder.embed(texts)` gives a float32 vector per text, and `numpy.asarray` of them is `(n, dimension)`; a single string counts as one.
- `reranker.rerank(query, documents)` gives one score per document.
- Vectors are `array.array` buffers: `numpy.asarray(vector)` reads one without a copy.

## Speech and transcription

- `speaker.speak(text, speed=1.2)` returns speech: iterate its PCM chunks, read `.audio`, or `.save("out.wav")`.
- `transcriber.transcribe("speech.wav")`, or `transcribe(pcm, sample_rate=16000)`, returns `.text` and timed `.words`.
- `transcribe` reads WAV or PCM; an audio file of any format given to a `chat` is decoded by `ffmpeg`.
- Audio is a float32 `array.array`: `torch.frombuffer(speech.audio, dtype=torch.float32)` reads it without a copy.

## Tokens

`model.tokenizer` is a [`toknroll.Tokenizer`](../../../toknroll/toknroll-native/python/API.md), served by the toknroll built into libjinfer: `model.tokenizer.count_tokens(text)` against `model.context_capacity`.
The `jinfer` wheel pins the `toknroll` version that matches it, and both wheels can share a process; their libraries never interact.

## OpenAI compatibility

`jinfer.OpenAI(ref)` runs code written for the OpenAI SDK's `chat.completions.create` and `embeddings.create` in process, unchanged.

## Errors

Errors are `jinfer.JinferError` subclasses, each a `RuntimeError` with a `.code`:

| Error | Also a | When |
|---|---|---|
| `jinfer.InvalidArgument` | `ValueError` | an option out of range, a prompt over the context, a closed handle |
| `jinfer.NotFound` | `FileNotFoundError` | a model, file or companion that does not exist, or is not cached while offline |
| `jinfer.Cancelled` | | a stream another thread cancelled while this one was stepping it |

## Threads and reproducibility

- Any thread may call: a model may be shared, and a reply or conversation belongs to one thread at a time.
- Temperature 0 gives the same tokens every run on one machine; two kernels can still land on different sides of a near-tie, so a test pins behaviour, not prose.

## Runtime options and logging

- `jinfer.configure("-Djinfer.threads=8", "-XX:MaxHeapSize=4g")` sets the runtime's options once, before the first load; `JINFER_OPTIONS` in the environment holds the same options and wins, and `-Dtoknroll.*` options reach the built-in tokenizer.
- Log lines go to `logging.getLogger("jinfer")`, as many as `JINFER_LOG` lets through: `off`, `error`, `warn` (the default), `info` or `debug`.
- A download shows its progress on stderr unless `JINFER_PROGRESS=off`.

## From C

The wheel is the C library too: `jinfer.h`, `toknroll.h`, `libjinfer` and a pkg-config file live in the installed package.
`jinfer-config --cflags --libs` prints the compiler flags, one argument per line, and `--pkgconfigdir` the directory for `PKG_CONFIG_PATH`.
The [libjinfer README](../README.md) covers Windows and the twenty-nine functions of [`jinfer.h`](../include/jinfer.h).

## Examples

One script a use case, in [`examples/`](examples), each runnable as it is with the model it names:

| Script | Model |
|---|---|
| `chat.py`, `stream.py`, `conversation.py`, `structured.py`, `tools.py`, `openai_client.py` | `LiquidAI/LFM2.5-350M-GGUF:Q8_0` |
| `vision.py` | `LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0`, with its `mmproj` file as `companions={"media": ...}` |
| `embed.py` | `LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0` |
| `rerank.py` | `mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0` |
| `speak.py` | `simonfxr/kokoro.cpp-GGUF:Q8_0`, with a voice file as `companions={"voice": ...}` |
| `transcribe.py` | `mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf` |
