# toknroll for Python

Token-perfect LLM tokenization from Python over `libtoknroll`, Tok'n'Roll's shared library with a C API.
Byte-exact with the reference tokenizers, no Python ML stack, no dependencies.

```python
import toknroll

tokenizer = toknroll.load("Qwen/Qwen3-0.6B")             # tokenizer.json from the hub, metadata only
ids = tokenizer.encode("Hello, world!")                  # array('i', [9707, 11, 1879, 0])
tokenizer.decode(ids)                                    # 'Hello, world!'
tokenizer.count_tokens(text)                             # the budget, without the list
tokenizer.encode_batch(texts)                            # one call, the texts spread over every core
tokenizer.encode("<|im_end|>", specials=True)            # special tokens spelled in the text stay whole
tokenizer.decode_bytes(ids)                              # the tokens' bytes, even half a character
tokenizer.vocabulary_size, tokenizer.token(0), tokenizer.id("<|im_end|>")
```

| You want | Call |
|---|---|
| ids | `encode(text)`, an `array('i')`: four bytes an id, nothing boxed, `numpy.frombuffer(ids, dtype=numpy.int32)` without a copy; `specials=True` keeps spelled special tokens whole |
| text | `decode(ids)`: an int32 buffer (`array('i')`, numpy) read in place, any other iterable of int copied once; `decode_bytes(ids)` for the raw bytes |
| a token budget | `count_tokens(text)`, no list built |
| many texts | `encode_batch(texts)`: one call, the library spreads the texts over the cores |
| the vocabulary | `vocabulary_size`, `token(id)`, `id(token)`, `token_flags(id)` (a `TokenFlags` mask: `NORMAL`, `CONTROL`, `USER_DEFINED`, `BYTE`, ...), `None` when absent |
| text without the specials | `decode(i for i in ids if not tokenizer.token_flags(i) & SPECIAL)` with `SPECIAL = TokenFlags.CONTROL \| TokenFlags.USER_DEFINED \| TokenFlags.UNKNOWN`: the special kinds only, a `BYTE` token is part of a character |
| another process | a `Tokenizer` pickles as its `source`, so `multiprocessing` and `concurrent.futures` just work |
| the runtime | `toknroll.configure("-Dtoknroll.gguf.pre.my-model=alias:qwen2", "-XX:MaxHeapSize=512m", "-XX:ActiveProcessorCount=4")` once, before first use: a GGUF pre-tokenizer name, a heap cap, a cap on the threads `encode_batch` spreads over |

| Source | Example |
|---|---|
| a `tokenizer.json`, or the directory holding one | `toknroll.load("./my-model/")` |
| a GGUF file | `toknroll.load("model-Q8_0.gguf")`, the tokenizer read from its header |
| a tiktoken checkout | `toknroll.load("./Kimi-K3/")`: a folder holding `tiktoken.model` and the `tokenizer_config.json` with its regex, local or on a hub |
| a hub repository | `toknroll.load("Qwen/Qwen3-0.6B")`: its `tokenizer.json`; `modelscope.cn/Qwen/Qwen3-0.6B` for ModelScope, `@rev` pins a revision |
| a GGUF on a hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")` or `.../LFM2.5-350M-Q8_0.gguf`: the header only, never the weights |
| a file in a repository | `toknroll.load("org/repo/tokenizers/v2")`: a folder's `tokenizer.json`, or the file a path names |

`encode` takes raw text, as tiktoken's `encode_ordinary` does: no BOS, no EOS, no template, those belong to the chat template that writes the prompt.
`specials=True` keeps every special token spelled in the text whole, the `CONTROL`, `USER_DEFINED` and `UNKNOWN` tokens, `<think>` among them on the models that add it; by default they are text.
`decode` spells every id, special tokens included, as tiktoken does, where HF's `decode` skips them by default; the recipe above drops them.
A token holding part of a character decodes per `errors=`, `"replace"` by default as in tiktoken.
`decode(encode(text))` is the text after the model's own normalization, NFC where its `tokenizer.json` says so, spaces through `▁` in a SentencePiece model, exactly as the reference tokenizer does; `decode_bytes` gives the bytes.
A token's spelling is its source's: `▁<` from a `tokenizer.json` as HF spells it, ` <` from a GGUF as llama.cpp does, the same ids and text either way; an added token is spelled as the text that matches it, so one whose content is `▁▁` reads as two spaces.
A hub source is `[host/]owner/repo[@rev][/path][:quant]`: a bare word is a local path, never a repository, and `@rev` comes before the file path, `owner/repo@main/model-Q8_0.gguf`.
BPE tokenizers only, the families the [main README](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md) lists and others built from the same components: a `tokenizer.json` that uses a component outside that set is refused at load, as `Unsupported` naming the field, never loaded approximately.
A `Tokenizer` never changes, so it may be shared between threads, `close` being the one call to make when no other thread is using it; every call releases the GIL.
Errors are `toknroll.ToknrollError` subclasses: `InvalidArgument` (also a `ValueError`), `NotFound` (also a `FileNotFoundError`), `Unsupported` (a valid tokenizer toknroll does not implement, also a `ValueError`); each has `.code`.
An id that is negative or past int32 is an `OverflowError` and a non-int a `TypeError`, as in HF `tokenizers` and tiktoken.
Remote metadata is fetched with `curl`, which must be on the PATH for hub sources only.
It is cached where `toknroll.configure("-Dtoknroll.cache.root=/srv/cache")` says, else `TOKNROLL_CACHE_ROOT`, else in jinfer's store, the platform cache directory under `jinfer`.
The HF hub cache (`HF_HUB_CACHE`, `HF_HOME`) is read too, so a file `transformers` fetched serves here; with the default cache root, a file fetched here is published there as well, so `transformers` finds it.
`-Dtoknroll.offline=true` serves the cache and never touches the network; set to anything but `false` it is offline, a misspelled true never means online.
`TOKNROLL_OPTIONS` is split on whitespace, with no quoting and no `~`: a value holding a space goes through `configure`.
The library's log lines go to `logging.getLogger("toknroll")`, as many as `TOKNROLL_LOG` lets through: `off`, `error`, `warn` (the default), `info` (downloads) or `debug`.
Python's logging then decides what shows; it shows warnings and errors until the application configures it.
So `TOKNROLL_LOG=info` shows the downloads once the application logs at INFO, `logging.basicConfig(level=logging.INFO)` for one; `TOKNROLL_LOG=off` silences the library.

## With jinfer

[jinfer](https://pypi.org/project/jinfer/)'s `model.tokenizer` is this class, served by libjinfer, which carries its own toknroll; the jinfer wheel pins the exact toknroll it embeds, and the class checks the library's version when it binds.
`toknroll.configure` and `TOKNROLL_OPTIONS` configure libtoknroll only: a model's tokenizer takes the same `-Dtoknroll.*` options through `jinfer.configure` and `JINFER_OPTIONS`.
Both wheels in one process are two runtimes that never interact, in any order.

## Install

```sh
pip install toknroll            # the wheel carries the library: Linux x86_64 and aarch64 (glibc 2.28 or later), macOS 14 or later on Apple silicon, Windows 10 or later on x86_64; x86_64 means x86-64-v3, a CPU with AVX2
```

From a source checkout, `make python` in `toknroll/toknroll-native` stages the freshly built library into the package, then `pip install python/`.
`TOKNROLL_LIBRARY=/path/to/libtoknroll.so` points the package at any build, and at nothing else: a path that does not exist is an error, never a fallback.
