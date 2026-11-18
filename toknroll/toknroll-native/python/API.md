# toknroll Python API

The full reference of the `toknroll` package.
The [README](README.md) is the short introduction; the [Tok'n'Roll README](../../README.md) has the parity tests, the tested models and the known divergences.

## Loading a tokenizer

`toknroll.load(source)` returns a `Tokenizer`, and `source` says what it was loaded from:

| Source | Example |
|---|---|
| a `tokenizer.json`, or the folder holding one | `toknroll.load("./my-model/")` |
| a GGUF file | `toknroll.load("model-Q8_0.gguf")`, the tokenizer read from its header |
| a tiktoken checkout | `toknroll.load("./Kimi-K3/")`: `tiktoken.model` and the `tokenizer_config.json` beside it |
| a Hub repository | `toknroll.load("Qwen/Qwen3-0.6B")`; `modelscope.cn/Qwen/Qwen3-0.6B` for ModelScope, `@rev` for a revision |
| a GGUF on the Hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")`: the header only, never the weights |
| a file in a repository | `toknroll.load("org/repo/tokenizers/v2")`: a folder's `tokenizer.json`, or the file the path names |

- A Hub source is `[host/]owner/repo[@rev][/path][:quant]`; a bare word is a local path, and a pasted URL is refused with the spelling to use.
- A `tokenizer.json` is read as it is: a token only `tokenizer_config.json` adds is not a token here, as it is not for HF `tokenizers`.
- A `tokenizer.json` that uses a component toknroll lacks (WordPiece, Unigram, some normalizers) is refused as `Unsupported`, naming the field; the [tested implementations](../../README.md#tested-implementations) list the common models that do not load, and why.

## The Tokenizer

| You want | Call |
|---|---|
| ids | `encode(text)`, an `array('i')`; `specials=True` keeps special tokens spelled in the text whole |
| text | `decode(ids)`, or `decode_bytes(ids)` for the exact bytes |
| a token count | `count_tokens(text)`, without building the list |
| many texts | `encode_batch(texts)`, spread over every core |
| the vocabulary | `vocabulary_size`, `token(id)`, `id(token)`, `token_flags(id)`, `None` when absent |
| text without special tokens | `decode(i for i in ids if not tokenizer.token_flags(i) & SPECIAL)`, with `SPECIAL = TokenFlags.CONTROL \| TokenFlags.USER_DEFINED \| TokenFlags.UNKNOWN` |
| numpy | `numpy.frombuffer(ids, dtype=numpy.int32)`, no copy; `decode` reads a writable, contiguous int32 array in place |
| another process | a `Tokenizer` pickles as its `source`, so `multiprocessing` and `concurrent.futures` just work |

- A `Tokenizer` never changes, so threads may share it, and every call releases the GIL.
- `close()`, a `with` block or garbage collection frees it, once no other thread is using it.
- Its first encode builds and checks its fast encoder, once: about 10-60 ms, after which a short text takes microseconds.
- Memory: about 90-330 MB of RSS once loaded (gpt2 to Gemma 4), up to 130 MB more while a large text encodes.

## Encode and decode

- `encode` takes raw text, as tiktoken's `encode_ordinary` does: no BOS, no EOS, no template, which belong to the chat template that writes the prompt.
- `specials=True` keeps every special token spelled in the text whole, `<think>` among them on the models that add it; by default they are text.
- `decode` spells every id, special tokens included, as tiktoken does; a token holding part of a character decodes per `errors=`, `"replace"` by default.
- `decode(encode(text))` is the text after the model's own normalization, NFC where its `tokenizer.json` says so; `decode_bytes` gives the exact bytes.

## Errors

| Error | Also a | When |
|---|---|---|
| `toknroll.InvalidArgument` | `ValueError` | a malformed reference or file, an id no token has, an unknown option |
| `toknroll.NotFound` | `FileNotFoundError` | a missing file, repository or revision, or one not cached while offline |
| `toknroll.Unsupported` | `ValueError` | a valid tokenizer toknroll does not implement, the field named |

Each is a `toknroll.ToknrollError` with a `.code`; an id past int32 is an `OverflowError`, and a wrong argument type a `TypeError` naming the call.

## Cache, offline and options

- Fetched files are cached under `-Dtoknroll.cache.root=<dir>` or `TOKNROLL_CACHE_ROOT`, else in the store toknroll shares with jinfer: `~/.cache/jinfer` on Linux, `~/Library/Caches/jinfer` on macOS, `%LOCALAPPDATA%\jinfer` on Windows.
- The Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME`) is read in place, and with the default cache root a file fetched here is published there too, so `transformers` finds it.
- `TOKNROLL_OPTIONS="-Dtoknroll.offline=true"` serves the cache only.
- `toknroll.configure("-XX:MaxHeapSize=512m", "-XX:ActiveProcessorCount=4")` sets the runtime's options once, before first use: a heap cap, the cores `encode_batch` uses; `TOKNROLL_OPTIONS` holds the same options and wins.

## Logging

- Log lines go to `logging.getLogger("toknroll")`, as many as `TOKNROLL_LOG` lets through: `off`, `error`, `warn` (the default), `info` or `debug`.
- A download shows its progress on stderr; `TOKNROLL_PROGRESS=off` hides it.

## From C

The wheel is the C library too: `toknroll.h`, `libtoknroll` and a pkg-config file live in the installed package.
`toknroll-config --cflags --libs` prints the compiler flags, one argument per line, and `--pkgconfigdir` the directory for `PKG_CONFIG_PATH`.
The [libtoknroll README](../README.md) covers Windows, static linking and the thirteen functions of [`toknroll.h`](../include/toknroll.h).

## With jinfer

[jinfer](../../../jinfer/jinfer-native/python/API.md)'s `model.tokenizer` is this same class, served by the toknroll inside libjinfer and configured through `jinfer.configure`.
Both wheels can share a process; their libraries never interact.
