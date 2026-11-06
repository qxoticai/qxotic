# toknroll

**Token-perfect LLM tokenization for Python. One wheel, no dependencies.**

[![PyPI](https://img.shields.io/pypi/v/toknroll)](https://pypi.org/project/toknroll/)
[![Python](https://img.shields.io/pypi/pyversions/toknroll)](https://pypi.org/project/toknroll/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/qxoticai/qxotic/blob/main/LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)

`toknroll` is [Tok'n'Roll](https://github.com/qxoticai/qxotic/tree/main/toknroll) for Python: the same ids as each model's own tokenizer, from a `tokenizer.json`, a GGUF file, a tiktoken checkout or a hub reference, through `libtoknroll`, a shared library with a C API that the wheel carries.
No Python ML stack behind it, nothing to compile, nothing else to install.

```sh
pip install toknroll
```

```python
import toknroll

tokenizer = toknroll.load("Qwen/Qwen3-0.6B")             # the repository's tokenizer.json, fetched once, never the weights
ids = tokenizer.encode("Hello, world!")                  # array('i', [9707, 11, 1879, 0])
tokenizer.decode(ids)                                    # 'Hello, world!'
tokenizer.count_tokens("Hello, world!")                  # 4: the budget, without the list
tokenizer.encode_batch(["Hello", "world"])               # one call, the texts spread over every core
tokenizer.encode("<|im_end|>", specials=True)            # a special token spelled in the text stays whole
tokenizer.decode_bytes(ids)                              # the tokens' bytes, even half a character
tokenizer.vocabulary_size, tokenizer.token(0), tokenizer.id("<|im_end|>")
```

## Why toknroll

- **Token-perfect, and verified.** Byte-exact parity with HF `tokenizers`, tiktoken and llama.cpp, tested on the tokenizers of 18 model vendors: OpenAI, Google, Alibaba, Liquid AI, Moonshot AI, DeepSeek, Mistral AI, IBM, Meta, NVIDIA, Z.ai, MiniMax, Xiaomi, Poolside, JetBrains and more, the [full list](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md#tested-implementations) with the [known divergences](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md#known-divergences).
  A tokenizer it cannot reproduce exactly is refused at load, with the reason, never loaded approximately.
- **One loader for every file the models ship.** A `tokenizer.json` or its folder, a GGUF read for its header only, a tiktoken checkout, and any of them by hub reference, Hugging Face or ModelScope, metadata fetched and cached, weights never.
  No other package reads all three formats.
- **Fast, and out of the GIL.** Hundreds of MB/s of text on one core, `encode_batch` over every core, every call releasing the GIL; ids come back as an `array('i')`, four bytes each, that numpy and torch read without a copy.
- **Nothing to carry.** A 6 MB wheel with the library inside, no `torch`, no `transformers`, no `huggingface_hub`; a model's tokenizer in a 20 MB container.
- **Honest about its scope.** Raw text in, ids out, as tiktoken's `encode_ordinary`: no BOS, no EOS, no padding, no truncation, no offsets.
  Those belong to the chat template and the model, and leaving them out is what keeps the ids exact.
- **The same engine everywhere.** Tok'n'Roll is a Java library, `libtoknroll` is its native build, and [`toknroll.h`](https://github.com/qxoticai/qxotic/blob/main/toknroll/toknroll-native/include/toknroll.h) is thirteen functions any language calls; [jinfer](https://pypi.org/project/jinfer/) serves its models' tokenizers through this very class.

| | toknroll | HF `tokenizers` | tiktoken | llama.cpp bindings |
|---|---|---|---|---|
| `tokenizer.json` | yes | yes | no | no |
| GGUF | yes, the header only | no | no | yes, with the whole model loaded |
| tiktoken files | yes | no | yes | no |
| Hub references | yes, metadata only | with `huggingface_hub` | OpenAI's own downloads | with `huggingface_hub`, the whole model |
| Parity with the other two references | tested | its own format | its own format | its own format |
| Extra packages | none | `huggingface-hub`, `numpy` | `regex`, `requests` | builds llama.cpp at install, or a per-backend wheel index |

## Requirements

- Python 3.10 or later, and no other package.
- The wheel carries the library: Linux x86-64 and aarch64 with glibc 2.28 or later, macOS 14 or later on Apple silicon, Windows 10 or later on x86-64.
  On x86-64 the library is built for x86-64-v3, a CPU with AVX2.
- `curl` on the PATH, for hub sources only; a local file needs nothing.

## The API

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

A `Tokenizer` comes from `toknroll.load` only; it is closed by `close()`, a `with` block or garbage collection, and `source` says what it was loaded from.
It never changes, so it may be shared between threads, `close` being the one call to make when no other thread is using it; every call releases the GIL.

## Sources

| Source | Example |
|---|---|
| a `tokenizer.json`, or the directory holding one | `toknroll.load("./my-model/")` |
| a GGUF file | `toknroll.load("model-Q8_0.gguf")`, the tokenizer read from its header |
| a tiktoken checkout | `toknroll.load("./Kimi-K3/")`: a folder holding `tiktoken.model` and the `tokenizer_config.json` with its regex, local or on a hub |
| a hub repository | `toknroll.load("Qwen/Qwen3-0.6B")`: its `tokenizer.json`; `modelscope.cn/Qwen/Qwen3-0.6B` for ModelScope, `@rev` pins a revision |
| a GGUF on a hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")` or `.../LFM2.5-350M-Q8_0.gguf`: the header only, never the weights |
| a file in a repository | `toknroll.load("org/repo/tokenizers/v2")`: a folder's `tokenizer.json`, or the file a path names |

A hub source is `[host/]owner/repo[@rev][/path][:quant]`: a bare word is a local path, never a repository, and `@rev` comes before the file path, `owner/repo@main/model-Q8_0.gguf`.
A pasted URL is refused with the spelling to use instead.
BPE tokenizers only, the families the [main README](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md) lists and others built from the same components: a `tokenizer.json` that uses a component outside that set is refused at load, as `Unsupported` naming the field, never loaded approximately.
The main README has the table of common models that do not load, and why.
A `tokenizer.json` is read as it is: a token that only `tokenizer_config.json` adds is not a token here, as it is not one for HF `tokenizers`; most repositories write their added tokens into both files.

## What encode and decode do

`encode` takes raw text, as tiktoken's `encode_ordinary` does: no BOS, no EOS, no template, those belong to the chat template that writes the prompt.
`specials=True` keeps every special token spelled in the text whole, the `CONTROL`, `USER_DEFINED` and `UNKNOWN` tokens, `<think>` among them on the models that add it; by default they are text.
`decode` spells every id, special tokens included, as tiktoken does, where HF's `decode` skips them by default; the recipe in the table above drops them.
A token holding part of a character decodes per `errors=`, `"replace"` by default as in tiktoken.
`decode(encode(text))` is the text after the model's own normalization, NFC where its `tokenizer.json` says so, spaces through `▁` in a SentencePiece model, exactly as the reference tokenizer does; `decode_bytes` gives the bytes.
A token's spelling is its source's: `▁<` from a `tokenizer.json` as HF spells it, ` <` from a GGUF as llama.cpp does, the same ids and text either way; an added token is spelled as the text that matches it, so one whose content is `▁▁` reads as two spaces.

## Errors

Errors are `toknroll.ToknrollError` subclasses, each with `.code`:

| Error | Also a | When |
|---|---|---|
| `InvalidArgument` | `ValueError` | a malformed reference or file, an id no token has, an unknown option |
| `NotFound` | `FileNotFoundError` | a missing file, repository or revision, or one not cached while offline |
| `Unsupported` | `ValueError` | a valid tokenizer toknroll does not implement, the field named |

An id that is negative or past int32 is an `OverflowError` and a non-int a `TypeError`, as in HF `tokenizers` and tiktoken; a wrong argument type is a `TypeError` naming the call.

## Cache, offline, options

Remote metadata is fetched with `curl` and cached where `toknroll.configure("-Dtoknroll.cache.root=/srv/cache")` says, else `TOKNROLL_CACHE_ROOT`, else in jinfer's store, the platform cache directory under `jinfer`.
The HF hub cache (`HF_HUB_CACHE`, `HF_HOME`) is read too, so a file `transformers` fetched serves here; with the default cache root, a file fetched here is published there as well, so `transformers` finds it.
`-Dtoknroll.offline=true` serves the cache and never touches the network; `false`, `0`, `off`, `no` and an empty value mean online, and anything else offline, so a misspelled true never means online.
`toknroll.configure` takes the runtime's options once, before the library's first use; a later call is refused.
`TOKNROLL_OPTIONS` in the environment holds the same options, applied after `configure`'s so they override them: the program sets its options, whoever runs it has the last word.
It is split on whitespace, with no quoting and no `~`: a value holding a space goes through `configure`.

## Logging

The library's log lines go to `logging.getLogger("toknroll")`, as many as `TOKNROLL_LOG` lets through: `off`, `error`, `warn` (the default), `info` (downloads) or `debug`.
Python's logging then decides what shows; it shows warnings and errors until the application configures it.
A download shows its progress on stderr, toknroll's live board on a terminal and a line at each tenth elsewhere; `TOKNROLL_PROGRESS=off`, or `-Dtoknroll.progress=off` through `configure`, turns it off entirely.
With progress off, `TOKNROLL_LOG=info` logs each download instead, a start and a done record, shown once the application logs at INFO, `logging.basicConfig(level=logging.INFO)` for one; `TOKNROLL_LOG=off` silences the library.

## From C

The wheel is the C library too: `toknroll.h`, `libtoknroll` and a pkg-config file live in the installed package, and the package says where.

```sh
cc tokenize.c $(python -m toknroll --cflags) $(python -m toknroll --libs)         # the include flag, the library and an rpath into the venv
PKG_CONFIG_PATH=$(python -m toknroll --pkgconfig) pkg-config --cflags --libs toknroll
```

On Windows the flags are `cl`'s, `/I` and `/LIBPATH` with `toknroll.lib`, one per line, and `toknroll.dll` sits beside the import library.
The C API is [`toknroll.h`](https://github.com/qxoticai/qxotic/blob/main/toknroll/toknroll-native/include/toknroll.h); nothing else is a tool here, `jbang toknroll@qxoticai` is.

## With jinfer

[jinfer](https://pypi.org/project/jinfer/)'s `model.tokenizer` is this class, served by libjinfer, which carries its own toknroll; the jinfer wheel pins the exact toknroll it embeds, and the class checks the library's version when it binds.
`toknroll.configure` and `TOKNROLL_OPTIONS` configure libtoknroll only: a model's tokenizer takes the same `-Dtoknroll.*` options through `jinfer.configure` and `JINFER_OPTIONS`.
Both wheels in one process are two runtimes that never interact, in any order.

## From a checkout

`make python` in `toknroll/toknroll-native` stages the freshly built library into the package, then `pip install python/`.
`TOKNROLL_LIBRARY=/path/to/libtoknroll.so` points the package at any build, and at nothing else: a path that does not exist is an error, never a fallback.

## Links

- [Tok'n'Roll](https://github.com/qxoticai/qxotic/tree/main/toknroll): the Java library, the parity tests, the tested models and the known divergences.
- [libtoknroll](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native): the shared library, its C API [`toknroll.h`](https://github.com/qxoticai/qxotic/blob/main/toknroll/toknroll-native/include/toknroll.h) and this package.
- [Documentation](https://qxotic.ai/toknroll), and [issues](https://github.com/qxoticai/qxotic/issues).
- Apache 2.0, by [Quixotic AI](https://qxotic.ai).
