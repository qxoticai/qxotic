# jinfer for Python

Local AI inference from Python over `libjinfer`, jinfer's shared library with a C API.
Chat, vision, audio, embeddings, reranking, speech synthesis and transcription; no server, no Python ML stack, no Java visible, `toknroll` the only dependency.

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
| embeddings | `embedder.embed(texts)` gives `(n, dimension)`, a single string counts as one; a text is embedded whole, up to `context_capacity`, the model's maximum by default |
| reranking | `reranker.rerank(query, documents)` gives one score per document |
| speech | `speech = speaker.speak(text, speed=1.2)`; iterate PCM chunks, or `speech.audio`, `speech.save("out.wav")` |
| transcription | `transcriber.transcribe("speech.wav")` or `(pcm, sample_rate=16000)`; `.text` and timed `.words` |
| tokens | `model.tokenizer` is a [`toknroll.Tokenizer`](https://pypi.org/project/toknroll/): `encode(text)` gives an `array('i')`, `decode(ids)`, `count_tokens`, `encode_batch`, `vocabulary_size`, `token(id)`, `id(token)`; `toknroll.load(ref)` gives one without the weights |
| token budgets | `len(model.tokenizer.encode(text))` against `model.context_capacity` (the state in use) and `model.max_context_length` (the model's bound) |
| the context size | `jinfer.load(ref, context_capacity=8192)`: the tokens of context the model's states hold, up to its maximum; the default is 4096 or the maximum when smaller, `-1` is the maximum, above it is refused |
| log lines | `logging.getLogger("jinfer")`: downloads, loads, kernels; silent until you configure logging |

Errors are `jinfer.JinferError` subclasses: `InvalidArgument` (also a `ValueError`), `NotFound` (also a `FileNotFoundError`), `Cancelled`; each has `.code`.
`jinfer.OpenAI(ref)` is a migration aid: code written for the OpenAI SDK's `chat.completions.create` and `embeddings.create` runs in-process, unchanged.

## Behaviour worth knowing

- A reply is already generating when `chat` returns. Iterating and the properties are two views of the same reply: read `.text` after a partial iteration and you get the whole reply.
- Closing a reply early cancels it, and the reply keeps what was produced: `.text` is the partial text, `.finish_reason` is `"cancelled"`, `.usage` counts what ran. In a conversation the partial text joins the history; a turn cancelled before its first word leaves no trace.
- A conversation turn with functions among the tools streams too: the calls run as you read the reply, between the words, and the text you iterate is the model's answer after them. `reply.usage` is the whole turn's.
- Temperature 0 gives the same tokens every run. Two different kernels can still land on different sides of a near-tie between two tokens, so a test pins behaviour, not prose.
- A conversation's turns are served from the engine's kept state, so a turn costs only its new tokens; `reply.usage.cached_tokens` shows it. A fork shares the prefix; a rewind recomputes nothing and keeps the system prompt; `prepare()` computes the prefix before the user types. With `prompt_cache=path`, what a model prepared is written when it closes and read by the next one.
- Structured output is enforced while decoding, so malformed JSON cannot come back. Validate meaning if you need to; there is no format to retry.
- Silent in programs: nothing is printed once logging is configured, `logging.basicConfig(level=logging.INFO)` shows downloads and loads as log lines. At a terminal, a download shows jinfer's progress board. Ctrl-C during a download ends the process; the download resumes next time. `repr(model)` names the kernels and threads that actually run.
- Any thread may call. A model may be shared between threads; a reply, speech or conversation belongs to one thread at a time. `async for` on a reply or speech steps in a worker thread; wrap one-shot calls in `asyncio.to_thread`.
- A model's weights stay alive while any reply or speech made from it lives; `close()` drops your reference. Context managers close; so does garbage collection.
- Embeddings and audio are float32 `array.array` buffers, scores are floats: the standard library, whoever installs the package. `np.asarray(vector)` and `torch.frombuffer(speech.audio, dtype=torch.float32)` are zero-copy.
- `jinfer.configure("-Djinfer.threads=8", "-Djinfer.cache.sessions=16", "-Djinfer.cache.budget=8g", "-Djinfer.offline=true", "-XX:MaxHeapSize=4g")` passes the runtime the options a java command line would, once and before the first load; the engine's `jinfer.*` and `jam.*` properties are all reachable this way. A later call is refused, and a rejected option fails this one. `JINFER_OPTIONS="-Djinfer.threads=4"` in the environment holds the same options and overrides them, so the box a program runs on can retune it without a code change.
- `model.tokenizer` runs inside libjinfer, so `jinfer.configure` is what configures it. `toknroll.load(...)` in the same process loads libtoknroll as well, a second runtime with its own `toknroll.configure`, about 60 MB more; a program that only needs a model's tokenizer never loads it.
- A scalar schema (`int`, `str`, `bool`, `Literal`, `Enum`) travels as `{"value": ...}` on the wire, so the model's first token is anchored; `reply.parsed` is the value, `reply.text` the JSON. Objects and lists go as they are.

## Install

```sh
pip install jinfer            # the wheel carries the library: Linux x86_64 with AVX2 (glibc 2.34 or later) today
```

From a source checkout, `make python` in `jinfer/jinfer-native` stages the freshly built library into the package, then `pip install python/`.
`JINFER_LIBRARY=/path/to/libjinfer.so` points the package at any build.
`pydantic` is optional.

## Examples

`examples/`: `chat.py`, `stream.py`, `conversation.py`, `structured.py`, `tools.py`, `openai_client.py`, `embed.py`, `rerank.py`, `speak.py`, `transcribe.py`, `vision.py`.
