# toknroll for Python

Token-perfect LLM tokenization from Python over `libtoknroll`, Tok'n'Roll's shared library with a C API.
Byte-exact with the reference tokenizers, no Python ML stack, no dependencies.

```python
import toknroll

tokenizer = toknroll.load("Qwen/Qwen3-0.6B")             # tokenizer.json from the hub, metadata only
ids = tokenizer.encode("Hello, world!")                  # array('i', [9707, 11, 1879, 0])
tokenizer.decode(ids)                                    # 'Hello, world!'
tokenizer.count_tokens(text)                             # the budget, without the list
tokenizer.encode_batch(texts)                            # on threads: the library releases the GIL
tokenizer.encode("<|im_end|>", specials=True)            # special tokens spelled in the text stay whole
tokenizer.decode_bytes(ids)                              # the tokens' bytes, even half a character
tokenizer.vocabulary_size, tokenizer.token(0), tokenizer.id("<|im_end|>")
```

| You want | Call |
|---|---|
| ids | `encode(text)`, an `array('i')`: four bytes an id, nothing boxed, `numpy.frombuffer(ids, dtype=numpy.int32)` without a copy; `specials=True` keeps spelled special tokens whole, by default they are text |
| text | `decode(ids)` from any int32 buffer or iterable of int, read in place; a partial character decodes per `errors=`, `"replace"` by default as in tiktoken; `decode_bytes(ids)` for the raw bytes |
| a token budget | `count_tokens(text)`, no list built |
| many texts | `encode_batch(texts)`: one call, the library spreads the texts over the cores; `decode_batch(batches)` |
| the vocabulary | `vocabulary_size`, `token(id)`, `id(token)`, `token_flags(id)` (a `TokenFlags` mask: `NORMAL`, `CONTROL`, `USER_DEFINED`, `BYTE`, ...), `None` when absent |
| text without the specials | `decode(i for i in ids if tokenizer.token_flags(i) & TokenFlags.NORMAL)` |
| another process | a `Tokenizer` pickles as its `source`, so `multiprocessing` and `concurrent.futures` just work |
| the runtime | `toknroll.configure("-Dtoknroll.gguf.pre.my-model=alias:qwen2", "-XX:MaxHeapSize=512m")` once, before first use: a new GGUF pre-tokenizer name, the hub cache root, a heap cap, any `toknroll.*` property the Java library reads; `TOKNROLL_OPTIONS` in the environment holds the same and overrides them |

| Source | Example |
|---|---|
| a `tokenizer.json`, or the directory holding one | `toknroll.load("./my-model/")` |
| a GGUF file | `toknroll.load("model-Q8_0.gguf")`, the tokenizer read from its header |
| a hub repository | `toknroll.load("Qwen/Qwen3-0.6B")`: its `tokenizer.json`; `modelscope.cn/Qwen/Qwen3-0.6B` for ModelScope, `@rev` pins a revision |
| a GGUF on a hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")` or `.../LFM2.5-350M-Q8_0.gguf`: the header only, never the weights |
| a file in a repository | `toknroll.load("org/repo/tokenizers/v2")`: a folder's `tokenizer.json`, or the file a path names |
| a GGUF on a hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")`, the header fetched, never the weights |

A `Tokenizer` never changes, so it may be shared between threads; `encode` releases the GIL.
Errors are `toknroll.ToknrollError` subclasses: `InvalidArgument` (also a `ValueError`), `NotFound` (also a `FileNotFoundError`); each has `.code`.
Remote metadata is fetched with `curl`, which must be on the PATH for hub sources only, and cached under `TOKNROLL_CACHE_ROOT`, the platform cache directory by default.

## With jinfer

[jinfer](https://pypi.org/project/jinfer/)'s `model.tokenizer` is this class, served by libjinfer, which carries its own toknroll; the jinfer wheel pins the exact toknroll it embeds, and the class checks the library's version when it binds.
`toknroll.configure` and `TOKNROLL_OPTIONS` configure libtoknroll only: a model's tokenizer takes the same `-Dtoknroll.*` options through `jinfer.configure` and `JINFER_OPTIONS`.
Both wheels in one process are two runtimes that never interact, in any order.

## Install

```sh
pip install toknroll            # the wheel carries the library: Linux x86_64 (glibc 2.17 or later), macOS 14 or later on Apple silicon, Windows 10 or later on x86_64
```

From a source checkout, `make python` in `toknroll/toknroll-native` stages the freshly built library into the package, then `pip install python/`.
`TOKNROLL_LIBRARY=/path/to/libtoknroll.so` points the package at any build.
