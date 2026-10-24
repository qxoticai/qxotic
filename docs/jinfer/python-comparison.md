---
sidebar_position: 5
---

# jinfer for Python, compared

**How the jinfer Python API reads next to ollama, transformers and llama-cpp-python, use case by use case.**
This is a usability comparison: how many lines a common task takes, how many concepts it demands, how the result comes back, and what goes wrong when it goes wrong.
It is not a benchmark.

Versions compared, as of 2026-10-25:

| Library | Version | What it is |
|---|---|---|
| jinfer | 0.1.0, unpublished | an in-process library over `libjinfer`, one C API, GGUF models |
| ollama | Python client 0.6.3, server 0.33.3 | an HTTP client for a separately installed and running server |
| transformers | 5.20.2 | the PyTorch model zoo, with `pipeline`, `generate` and chat templates |
| llama-cpp-python | 0.3.36 | ctypes bindings over llama.cpp, with an OpenAI-shaped high-level API |

Every jinfer snippet below was run end to end on this machine against the current library, and all 23 checks passed.
The other three libraries' snippets are taken from their official documentation at the versions above and were not executed here.
Each one is as short as the documentation allows for the task.

## The shape of each library

**jinfer** has one entry point, `jinfer.load(ref)`, which returns the class the model file calls for: `LanguageModel`, `Embedder`, `Reranker`, `Speaker` or `Transcriber`.
Every call returns a typed object, every option is a keyword, and the stream and the result are the same object.
The engine keeps conversation state, runs tools, enforces schemas while decoding, and resolves and caches models from the hub.

**ollama** is a thin client over a server.
Functions mirror the HTTP routes (`chat`, `generate`, `embed`, `pull`, `list`, ...), requests and responses are Pydantic models with dict access, and the server owns the models, the cache and the concurrency.
Nothing works until `ollama serve` runs and `ollama pull` has fetched the model.

**transformers** is a model library first and an inference API second.
The short path is `pipeline("text-generation")`, the real path is `AutoTokenizer` plus `AutoModelForCausalLM`, `apply_chat_template`, `generate` and `batch_decode`, with tensors, devices and dtypes in view the whole way.
It covers every modality through separate pipelines and every model on the Hub, at the cost of the most concepts.

**llama-cpp-python** wraps llama.cpp in one class, `Llama`, whose constructor takes 45 parameters.
`create_chat_completion` returns the OpenAI response as a plain dict, and tools, JSON schemas and images use the OpenAI wire shapes verbatim.
Building the wheel compiles llama.cpp unless a prebuilt index is given, and the hardware backend is chosen at install time.

## Use cases, side by side

### A first reply

jinfer, 3 lines:

```python
import jinfer

model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")      # downloads once, picks the class
print(model.chat("Why is the sky blue?", max_tokens=200).text)
```

ollama, 3 lines plus two commands outside Python (`ollama serve`, `ollama pull gemma4`):

```python
from ollama import chat

response = chat(model="gemma4", messages=[{"role": "user", "content": "Why is the sky blue?"}])
print(response.message.content)
```

transformers, 4 lines, with `torch` installed and a model that fits in RAM at its saved dtype:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto")
out = pipe([{"role": "user", "content": "Why is the sky blue?"}], max_new_tokens=200)
print(out[0]["generated_text"][-1]["content"])
```

llama-cpp-python, 4 lines, with `huggingface-hub` installed for `from_pretrained`:

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(repo_id="LiquidAI/LFM2.5-350M-GGUF", filename="*Q8_0.gguf", n_ctx=4096, verbose=False)
out = llm.create_chat_completion(messages=[{"role": "user", "content": "Why is the sky blue?"}], max_tokens=200)
print(out["choices"][0]["message"]["content"])
```

The text lives at a different depth in each library: `.text`, `.message.content`, `[0]["generated_text"][-1]["content"]`, `["choices"][0]["message"]["content"]`.
jinfer and ollama accept a string or a messages list; transformers and llama-cpp-python need the messages list.
Two defaults bite in the others: transformers generates 20 tokens unless `max_new_tokens` is set, and llama-cpp-python opens a 512-token context unless `n_ctx` is set, and prints its load log unless `verbose=False`.
jinfer takes the model file's defaults for everything, and `max_tokens` left out means "up to the context".

### Streaming, and stopping early

jinfer:

```python
with model.chat("Write a poem about rivers.", max_tokens=200) as reply:
    for i, text in enumerate(reply):
        print(text, end="", flush=True)
        if i == 20:
            break                        # closing cancels the generation
print(reply.finish_reason, reply.usage)  # "cancelled", and what ran; reply.text is the partial poem
```

ollama:

```python
for chunk in chat(model="gemma4", messages=messages, stream=True):
    print(chunk.message.content, end="", flush=True)
```

Breaking out closes the HTTP response; the partial text is whatever you accumulated, and there is no usage for a cancelled stream.

transformers:

```python
from threading import Thread
from transformers import TextIteratorStreamer

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
Thread(target=model.generate, kwargs=dict(inputs, streamer=streamer, max_new_tokens=200)).start()
for text in streamer:
    print(text, end="", flush=True)
```

You run the generation in your own thread and read a queue.
Stopping early needs a custom `StoppingCriteria`; breaking out of the loop leaves the thread generating to the end.

llama-cpp-python:

```python
for chunk in llm.create_chat_completion(messages=messages, stream=True):
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

The first chunk carries a `role` and no `content`, so every consumer writes the `if`.
Breaking out stops the generator; there is no usage and no finish reason for what ran.

Only jinfer keeps stream and result together: iterate as far as you like, then read `.text`, `.usage` and `.finish_reason` from the same object.

### A conversation

jinfer:

```python
conversation = model.conversation(system="Be terse.", max_tokens=32)
conversation.chat("My name is Zoltan.")
reply = conversation.chat("What is my name?")
print(reply.text, reply.usage.cached_tokens, "of", reply.usage.prompt_tokens, "prompt tokens were cached")
branch = conversation.fork()      # same history, a different future, nothing recomputed
conversation.rewind()             # forgets the last exchange, keeps the system prompt
```

In the run above 29 of 44 prompt tokens came from the engine's kept state.

ollama:

```python
messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "My name is Zoltan."}]
messages.append(chat(model="gemma4", messages=messages).message)
messages.append({"role": "user", "content": "What is my name?"})
print(chat(model="gemma4", messages=messages).message.content)
```

You own the list and re-send it every turn; the server reuses the prefix it has.
A fork is `copy.deepcopy(messages)`, a rewind is `del messages[-2:]`.

transformers:

```python
chat = out[0]["generated_text"]                               # the history, the reply included
chat.append({"role": "user", "content": "What is my name?"})
out = pipe(chat, max_new_tokens=32)
print(out[0]["generated_text"][-1]["content"])
```

Every turn recomputes the whole history from scratch unless you manage `past_key_values` yourself.

llama-cpp-python: the same list management as ollama, with `out["choices"][0]["message"]` appended.
The `Llama` object reuses the longest common prefix of the previous prompt, so a linear conversation is cheap, but a fork invalidates it.

jinfer is the only one of the four where the conversation is an object with memory, a cache you can see in `usage.cached_tokens`, and `fork` and `rewind` as operations.

### Structured output

jinfer, with nothing installed beyond the standard library:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Person:
    name: str
    born: int
    field: Literal["mathematics", "physics", "literature"]

person = model.chat("Ada Lovelace, born 1815, wrote the first program.", schema=Person).parsed
# Person(name='Ada Lovelace', born=1815, field='mathematics')
planets = model.chat("Name three planets.", schema=list[str]).parsed   # ['Jupiter', 'Saturn', 'Uranus']
wet = model.chat("Is water wet?", schema=bool).parsed                  # True
```

The grammar is enforced while decoding, so there is no parse error to catch and no retry.
TypedDicts, Pydantic classes, Enums and raw JSON schemas work the same way.

ollama, with Pydantic:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    born: int

response = chat(model="gemma4", messages=messages, format=Person.model_json_schema(), options={"temperature": 0})
person = Person.model_validate_json(response.message.content)
```

Two steps, schema out and validation back in, and the documentation also recommends repeating the schema in the prompt.
A `list[str]` or a `bool` on its own is not a form the API takes.

transformers: nothing built in.
The documentation points to Outlines or SynCode as separate libraries; without them you prompt for JSON, `json.loads` the text and retry.

llama-cpp-python:

```python
out = llm.create_chat_completion(
    messages=messages,
    response_format={"type": "json_object", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "born": {"type": "integer"}}, "required": ["name", "born"]}},
)
person = json.loads(out["choices"][0]["message"]["content"])
```

The schema is enforced by a grammar, as in jinfer, but you write it by hand (or export it from Pydantic) and get a dict back, not an object.

### Tool calling

jinfer:

```python
def weather(city: str) -> dict:
    """Current weather for a city."""
    return {"celsius": 18, "sky": "clear"}

conversation = model.conversation(tools=[weather])
print(conversation.chat("What's the weather in Zurich?").text)   # weather("Zurich") ran; this is the answer

reply = model.chat("Weather in Zurich?", tools=[weather], tool_choice="required")   # the single-shot form
reply.tool_calls   # [ToolCall(id='call_...', name='weather', arguments={'city': 'Zurich'})]
```

The signature is the schema and the docstring the description; the conversation runs the loop, the single-shot form hands you parsed arguments.

ollama, from its `tools.py` example, trimmed:

```python
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a (int): The first number
      b (int): The second number
    """
    return int(a) + int(b)   # "returned tool call arguments don't always conform exactly to schema"

available = {"add_two_numbers": add_two_numbers}
response = chat("llama3.1", messages=messages, tools=[add_two_numbers])
if response.message.tool_calls:
    messages.append(response.message)
    for call in response.message.tool_calls:
        output = available[call.function.name](**call.function.arguments)
        messages.append({"role": "tool", "content": str(output), "tool_name": call.function.name})
    print(chat("llama3.1", messages=messages).message.content)
```

Functions are tools here too, but the loop, the dispatch table and the result message are yours, and the example itself casts its arguments because they arrive untyped.

transformers, from its tool-use guide, trimmed:

```python
inputs = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True, return_dict=True, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=128)
text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):])
call = tokenizer.parse_response(text, prefix=inputs["input_ids"][0])["tool_calls"][0]   # when the tokenizer has a response template
messages.append({"role": "assistant", "tool_calls": [call]})
messages.append({"role": "tool", "content": "22"})                                     # "the returned content is always a string"
inputs = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs.to(model.device), max_new_tokens=128)
```

Docstrings must be Google-style, the tool call comes back as text that `parse_response` decodes only for models that ship a response template, and the guide warns that the `tool_calls` dict shape differs from OpenAI's JSON string.

llama-cpp-python:

```python
out = llm.create_chat_completion(
    messages=messages,
    tools=[{"type": "function", "function": {"name": "weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    tool_choice={"type": "function", "function": {"name": "weather"}},
)
call = out["choices"][0]["message"]["tool_calls"][0]["function"]
args = json.loads(call["arguments"])
```

The tool schema is hand-written, the arguments are a JSON string, and the README's example needs `chat_format="chatml-function-calling"` on the model.

### Vision

jinfer:

```python
vl = jinfer.load("LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0",
                 companions={"media": "LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"})
print(vl.chat([Path("photo.jpg"), "What is this? One sentence."]).text)   # a Path or bytes, next to the question
```

ollama:

```python
response = chat(model="gemma4", messages=[{"role": "user", "content": "What is in this image?", "images": ["photo.jpg"]}])
```

transformers:

```python
pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
out = pipe([{"role": "user", "content": [{"type": "image", "url": "photo.jpg"}, {"type": "text", "text": "What is in this image?"}]}], max_new_tokens=60)
```

llama-cpp-python:

```python
from llama_cpp.llama_chat_format import Llava15ChatHandler

llm = Llama(model_path="llava.gguf", chat_handler=Llava15ChatHandler(clip_model_path="mmproj.bin"), n_ctx=4096)
data_url = "data:image/jpeg;base64," + base64.b64encode(Path("photo.jpg").read_bytes()).decode()
out = llm.create_chat_completion(messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": "What is in this image?"}]}])
```

The image itself is easiest in jinfer and ollama: a path or bytes, no encoding, no content-part dicts.
ollama and transformers bundle the projector with the model; jinfer and llama-cpp-python name it at load.
In jinfer that is the design, not an omission: a companion you did not name is a capability that is off, so a text-only load never pays for vision, and sending an image to such a load fails with the message that names the companion to attach.

### Embeddings

jinfer:

```python
embedder = jinfer.load("LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0")
vectors = embedder.embed(["AI on the JVM", "the JVM thinks now", "bananas are yellow"])   # (3, 1024) float32
query = embedder.embed("Java machine learning")[0]
scores = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
```

ollama:

```python
vectors = ollama.embed(model="embeddinggemma", input=["AI on the JVM", "the JVM thinks now"]).embeddings   # list of lists
```

transformers, with the model's pooling done by hand (or two lines with the separate `sentence-transformers` package):

```python
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    hidden = enc(**batch).last_hidden_state
mask = batch["attention_mask"].unsqueeze(-1)
vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
```

llama-cpp-python:

```python
llm = Llama(model_path="embedding.gguf", embedding=True)
vectors = llm.embed(["AI on the JVM", "the JVM thinks now"])   # list of lists
```

A whole text is one input everywhere, but only jinfer returns an array shaped `(n, dimension)` ready for linear algebra; the others return nested Python lists, and llama-cpp-python needs the model opened in a separate embedding mode.

### Reranking

jinfer:

```python
reranker = jinfer.load("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
scores = reranker.rerank("capital of France", ["Paris is the capital of France.", "Bananas are yellow.", "The Seine flows through Paris."])
# [0.999, 0.000, 0.009]
```

ollama has no reranking call in its API.
llama-cpp-python has no reranking call in its high-level API.
transformers runs a cross-encoder through `AutoModelForSequenceClassification` on query-document pairs; for a model like Qwen3-Reranker you also reproduce its prompt and read the yes/no logits yourself, a dozen lines.

### Speech synthesis

jinfer:

```python
speaker = jinfer.load("simonfxr/kokoro.cpp-GGUF:Q8_0", companions={"voice": "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf"})
speech = speaker.speak("Matrix multiplications can talk.", speed=1.1)
for chunk in speech:          # float32 PCM as it is synthesized
    play(chunk)
speech.save("speech.wav")     # or speech.audio, the whole thing
```

transformers:

```python
tts = pipeline("text-to-speech", model="suno/bark-small")
out = tts("Matrix multiplications can talk.")      # {"audio": ndarray (channels, samples), "sampling_rate": int}
scipy.io.wavfile.write("speech.wav", out["sampling_rate"], out["audio"].T)
```

No streaming, and writing the file is yours.
ollama and llama-cpp-python have no speech synthesis.

### Transcription

jinfer:

```python
asr = jinfer.load("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
transcript = asr.transcribe("speech.wav")          # or (pcm, sample_rate=16000); any rate, resampled
transcript.text, transcript.words[0]              # Word(text='Matrix', start=0, end=0.8, confidence=0.99)
```

The run above round-tripped the Kokoro output through Parakeet and got the sentence back word for word.

transformers:

```python
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
out = asr("speech.mp3", return_timestamps="word")   # {"text": ..., "chunks": [{"text": "hi ", "timestamp": (0.5, 0.9)}, ...]}
```

Any file ffmpeg reads.
jinfer's `transcribe` takes WAV or PCM, while an audio file in a `chat` content list goes through the engine's ffmpeg decoder in any container; see the note at the end.
Word timestamps are limited to CTC models and Whisper.
ollama and llama-cpp-python have no transcription.

### Code written for the OpenAI SDK

jinfer runs it in-process:

```python
client = jinfer.OpenAI("LiquidAI/LFM2.5-350M-GGUF:Q8_0")     # instead of OpenAI(api_key=...)
response = client.chat.completions.create(messages=messages, max_tokens=40)
response.choices[0].message.content
```

The other three need a second process: `ollama serve` (its `/v1` routes), `python -m llama_cpp.server`, or `transformers serve`, and the real `openai` package pointed at `base_url`.

### Async

jinfer: `async for text in model.chat(...)`, each step in a worker thread, the loop stays free.
ollama: `AsyncClient`, native, the cleanest of the four because the work is over the network anyway.
transformers: `AsyncTextIteratorStreamer` plus your own `Thread` running `generate`.
llama-cpp-python: nothing; wrap calls in `asyncio.to_thread`.

### When it goes wrong

jinfer raises `JinferError` subclasses that are also the standard exceptions a caller expects: `NotFound` is a `FileNotFoundError`, `InvalidArgument` is a `ValueError`, and a wrong option type or a typo in an option name is a `TypeError` at the call.

```python
jinfer.load("/no/such/model.gguf")        # NotFound: /no/such/model.gguf: no such file
model.chat("x", max_tokens=-5)            # InvalidArgument: max_tokens must be -1 (context-bounded) or at least 1
model.chat("x", max_tokens="8")           # TypeError: max_tokens: got str
```

ollama raises `ResponseError` with an HTTP `status_code`; a 404 means the model is not pulled, and a connection error means the server is not running.
transformers surfaces PyTorch's errors: CUDA out of memory, shape mismatches, and silent quality problems from the wrong padding side or a missing chat template that the documentation lists as pitfalls.
llama-cpp-python raises `ValueError` for bad arguments, but a context overflow or a bad GGUF reaches a C-level assertion that aborts the Python process.

## Scorecard

| | jinfer | ollama | transformers | llama-cpp-python |
|---|---|---|---|---|
| Install | `pip install` plus numpy | the Ollama app, a running server, `pip install ollama` | `transformers` plus `torch`, gigabytes | compiles llama.cpp, or a per-backend wheel index |
| Lines to a first reply | 3 | 3, plus two shell commands | 4 | 4 |
| Where the text is | `.text`, `print(reply)` | `.message.content` | `[0]["generated_text"][-1]["content"]` | `["choices"][0]["message"]["content"]` |
| A string as the prompt | yes | no, a messages list | no | no |
| Streaming | iterate the reply | `stream=True`, iterate chunks | streamer object plus your own thread | `stream=True`, filter deltas |
| Stop early | close; partial text, usage and finish reason stay | break; nothing stays | custom stopping criteria | break; nothing stays |
| Conversation state | an object; engine cache visible; fork, rewind | your list, re-sent each turn | your list, recomputed each turn | your list, last prefix reused |
| Structured output | any Python type, parsed object back, enforced | JSON schema in, validate yourself, Pydantic | none built in | JSON schema in, dict back, enforced |
| Tools | functions; the conversation runs the loop | functions; you run the loop | functions (Google docstrings); you parse and loop | hand-written JSON; you parse and loop |
| Images in | a Path or bytes; projector named at load | a path; nothing else | content-part dicts | base64 data URL; projector and handler named |
| Audio in and out | transcribe with words, speak with streaming | neither | both, no streaming, file I/O yours | neither |
| Embeddings | `(n, dim)` float32 array | list of lists | pooling by hand, or another package | list of lists, embedding mode at load |
| Reranking | one call | none | cross-encoder by hand | none |
| Context size | `load(context_capacity=)`, 4096 or the model's maximum by default, refused above it | `options={"num_ctx": ...}`, the server's default | the model's full length, the memory is yours | `n_ctx`, 512 by default |
| Options | 8, keyword-only, typed, model defaults | `options={...}` dict, server defaults | `GenerationConfig`, 20-token default, `do_sample` off | 28 keyword arguments, 512-token context default |
| Errors | typed, standard base classes | HTTP status codes | PyTorch exceptions, silent pitfalls | `ValueError`, or a process abort |
| Async | `async for` | `AsyncClient` | async streamer plus a thread | none |
| OpenAI SDK code | in-process shim | server `/v1` | `transformers serve` | `llama_cpp.server` |
| Extra process | none | required | none | none |
| GPU | no | yes | yes | yes, chosen at install |
| Model breadth | the GGUF architectures jinfer ports | the Ollama library | everything on the Hub | every GGUF llama.cpp loads |

## Where jinfer trails

An honest comparison names what the others do better.

- **No GPU.** The other three run on CUDA, Metal or ROCm; jinfer runs on the CPU with its jam kernels.
- **Model breadth.** transformers loads anything on the Hub and llama-cpp-python anything llama.cpp supports; jinfer runs the architectures it has ported.
- **Audio files into `transcribe`.** WAV or PCM only; the transformers pipeline takes anything ffmpeg reads.
- **Model management.** ollama lists, pulls, shows and deletes models from Python; jinfer loads, and `offline=True` is the only hub control.
- **Fewer knobs.** `top_k`, `min_p`, repetition penalties, logprobs, logit bias and raw completion exist in llama-cpp-python and transformers; jinfer's Python exposes temperature, top_p, seed, stop, max_tokens, tools, tool_choice and reasoning_effort, and leaves the rest to the model file on purpose.
- **Not published yet.** `pip install jinfer` is the plan; the others are on PyPI today.

## What the comparison says about the jinfer API

The comparison confirms the mantras the API was built on: one object is the stream and the result, options are keywords, functions are tools, media are values, errors are typed, and the engine keeps the state.
Each of those is a place where one of the other three libraries hands the work to the caller: accumulating chunks, managing the messages list, writing JSON schemas, dispatching tool calls, base64-encoding images, catching parse errors.
In every use case above the jinfer snippet is the shortest or tied, and it is the only one whose result is a typed object rather than a nested dict.

One difference above is usability rather than scope, and it is a public API decision:

1. **One meaning for a Path.** An mp3 is a value in a `chat` content list, decoded by the engine's ffmpeg door, and an error in `transcribe`, which takes WAV or PCM because the C entry point takes samples and the Python side reads only WAV itself.
   Closing it means a bytes-in transcription entry in the C API, since the decoder lives in the library; keeping it means documenting `transcribe` as the raw-samples door.
   jinfer is an AI library, not a media converter, so either is defensible; the inconsistency between the two calls is the only part worth deciding.

Explicit companions stay: a capability is on when its companion is named and off when it is not, which is a property worth more than one saved argument.

The context size is now the caller's, as in ollama and llama-cpp-python: `jinfer.load(ref, context_capacity=8192)` sizes the states the model runs in, up to `model.max_context_length`, which the model file sets and nobody raises.
Left out, it is 4096 or the maximum when smaller; `-1` is the maximum; above it the load is refused with the bound in the message.

One defect surfaced while writing the vision snippets and is fixed: a generation that failed (an image sent to a model loaded without its projector) raised "the stream has not finished" instead of the engine's own message, because reading the result of a failed stream overwrote the thread's last error before Python read it.
The reply now raises the engine's message, "image/audio/video input is not supported by this model: media on a text-only load - attach the model's mmproj as the 'media' companion", and the Python battery pins it.

## Sources

- ollama: [ollama-python README](https://github.com/ollama/ollama-python), [tools example](https://github.com/ollama/ollama-python/blob/main/examples/tools.py), [structured outputs example](https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py), [multimodal example](https://github.com/ollama/ollama-python/blob/main/examples/multimodal-chat.py), [structured outputs guide](https://docs.ollama.com/capabilities/structured-outputs), [tool calling guide](https://docs.ollama.com/capabilities/tool-calling)
- transformers: [chat basics](https://huggingface.co/docs/transformers/main/en/conversations), [tool use](https://huggingface.co/docs/transformers/main/en/chat_extras), [response parsing](https://huggingface.co/docs/transformers/main/en/chat_response_parsing), [text generation and pitfalls](https://huggingface.co/docs/transformers/main/en/llm_tutorial), [generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies), [streamers](https://huggingface.co/docs/transformers/main/en/internal/generation_utils), [pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)
- llama-cpp-python: [README](https://github.com/abetlen/llama-cpp-python), [documentation](https://llama-cpp-python.readthedocs.io/en/latest/), [`Llama.__init__` and `create_chat_completion` signatures](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py)
- jinfer: `jinfer/jinfer-native/python/README.md`, the snippets in this page as run on 2026-10-25
