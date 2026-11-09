# toknroll

**Token-perfect LLM tokenization for Python: the model's own tokens, faster, with no dependencies.**

[![PyPI](https://img.shields.io/pypi/v/toknroll)](https://pypi.org/project/toknroll/)
[![Python](https://img.shields.io/pypi/pyversions/toknroll)](https://pypi.org/project/toknroll/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/qxoticai/qxotic/blob/main/LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#requirements)

`toknroll` reproduces each model's own tokenizer, id for id, from Hugging Face `tokenizer.json`, GGUF and tiktoken files, on disk or straight from the Hugging Face Hub.
One 6 MB wheel, no ML stack, and faster than Hugging Face `tokenizers` and tiktoken on every model in [the benchmark](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native/python/benchmarks):

![Encoding speed on six models: toknroll ahead of Hugging Face tokenizers 1.0 rc4 and 0.23.3 and of tiktoken](https://raw.githubusercontent.com/qxoticai/qxotic/main/toknroll/toknroll-native/python/benchmarks/encode.png)

```sh
pip install toknroll
```

```python
import toknroll

tokenizer = toknroll.load("Qwen/Qwen3-0.6B")    # fetched once, never the weights
ids = tokenizer.encode("Hello, world!")          # array('i', [9707, 11, 1879, 0])
tokenizer.decode(ids)                            # 'Hello, world!'
tokenizer.count_tokens("Hello, world!")          # 4
tokenizer.encode_batch(["Hello", "world"])       # spread over every core
tokenizer.encode("<|im_end|>", specials=True)    # array('i', [151645])
tokenizer.vocabulary_size, tokenizer.token(0)    # (151669, '!')
```

## Why toknroll

- **Token-perfect.** The same ids as each model's own tokenizer, byte for byte, verified by parity tests against Hugging Face `tokenizers`, llama.cpp and tiktoken; the few [known divergences](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md#known-divergences) are listed.
- **Fast.** 125-195 MB/s of new text on one core for Qwen, Llama, gpt-oss and DeepSeek, and 1.5-2 times Hugging Face on SentencePiece models such as Gemma and Mistral; `encode_batch` uses every core, and every call releases the GIL.
- **Reads what models ship.** A `tokenizer.json` or its folder, a GGUF file (its header only, never the weights) or a tiktoken checkout, from disk or by reference to a Hugging Face or ModelScope repository, fetched once and cached.
- **Popular models, tested; others may just work.** Parity is tested on the tokenizers of 18 vendors' models, Qwen, Llama, Gemma 4, gpt-oss, DeepSeek, Mistral, Granite, Kimi and [more](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md#tested-implementations).
  Another BPE tokenizer built from the same parts usually loads as it is; one toknroll cannot reproduce exactly is refused at load, with the reason, never approximated.
- **Nothing else to install.** No `torch`, no `transformers`, no compiler: the engine is compiled ahead of time with GraalVM Native Image into a plain shared library, so no JVM either.
- **Plain Python.** Ids come back as an `array('i')` that numpy and torch read without a copy; errors are typed; a tokenizer pickles, so `multiprocessing` just works.

| | toknroll | HF `tokenizers` | tiktoken | llama.cpp bindings |
|---|---|---|---|---|
| `tokenizer.json` | yes | yes | no | no |
| GGUF | yes, the header only | no | no | yes, the whole model loaded |
| tiktoken files | yes | no | yes | no |
| Hub references | built in, metadata only | with `huggingface-hub` | OpenAI's encodings only | with `huggingface-hub`, the whole model |
| Extra packages | none | `huggingface-hub` | `regex`, `requests` | `numpy`, `jinja2`, `diskcache`, and a llama.cpp build |

## Requirements

- Python 3.10 or later, and no other package.
- Linux x86-64 and aarch64 with glibc 2.28 or later, macOS 14 or later on Apple silicon, Windows 10 or later on x86-64: the wheel carries the library.
- On x86-64, a CPU with AVX2 (Intel Haswell, AMD Excavator or later); an older CPU, or a virtual machine that hides AVX2, is refused at load, naming what it lacks.
- `curl` on the PATH, for Hub sources only; a local file needs nothing.
- Memory: about 90-330 MB of RSS once loaded (gpt2 to Gemma 4), up to 130 MB more while a large text encodes.

## Usage

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
| runtime options | `toknroll.configure("-XX:MaxHeapSize=512m", "-XX:ActiveProcessorCount=4")`, once, before first use: a heap cap, the cores `encode_batch` uses |

A `Tokenizer` comes from `toknroll.load` only, and `source` says what it was loaded from.
It never changes, so threads may share it; `close()`, a `with` block or garbage collection frees it, once no other thread is using it.
Its first encode builds and checks its fast encoder, once: about 10-60 ms, after which a short text takes microseconds.

## Sources

| Source | Example |
|---|---|
| a `tokenizer.json`, or the folder holding one | `toknroll.load("./my-model/")` |
| a GGUF file | `toknroll.load("model-Q8_0.gguf")`, the tokenizer read from its header |
| a tiktoken checkout | `toknroll.load("./Kimi-K3/")`: `tiktoken.model` and the `tokenizer_config.json` beside it |
| a Hub repository | `toknroll.load("Qwen/Qwen3-0.6B")`; `modelscope.cn/Qwen/Qwen3-0.6B` for ModelScope, `@rev` for a revision |
| a GGUF on the Hub | `toknroll.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")`: the header only, never the weights |
| a file in a repository | `toknroll.load("org/repo/tokenizers/v2")`: a folder's `tokenizer.json`, or the file the path names |

A Hub source is `[host/]owner/repo[@rev][/path][:quant]`; a bare word is a local path, and a pasted URL is refused with the spelling to use.
A `tokenizer.json` is read as it is: a token only `tokenizer_config.json` adds is not a token here, as it is not for HF `tokenizers`.
A `tokenizer.json` that uses a component toknroll lacks (WordPiece, Unigram, some normalizers) is refused as `Unsupported`, naming the field; the main README has [the common models that do not load](https://github.com/qxoticai/qxotic/blob/main/toknroll/README.md#tested-implementations), and why.

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

## Cache, offline and logging

- Fetched files are cached under `-Dtoknroll.cache.root=<dir>` or `TOKNROLL_CACHE_ROOT`, else in the store toknroll shares with jinfer: `~/.cache/jinfer` on Linux, `~/Library/Caches/jinfer` on macOS, `%LOCALAPPDATA%\jinfer` on Windows.
- The Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME`) is read in place, and with the default cache root a file fetched here is published there too, so `transformers` finds it.
- `TOKNROLL_OPTIONS="-Dtoknroll.offline=true"` serves the cache only; the same options go through `toknroll.configure`, and the variable wins.
- Log lines go to `logging.getLogger("toknroll")`, as many as `TOKNROLL_LOG` lets through: `off`, `error`, `warn` (the default), `info` or `debug`.
- A download shows its progress on stderr; `TOKNROLL_PROGRESS=off` hides it.

## From C

The wheel is the C library too: `toknroll.h`, `libtoknroll` and a pkg-config file live in the installed package.
`toknroll-config --cflags --libs` prints the compiler flags, one argument per line, and `--pkgconfigdir` the directory for `PKG_CONFIG_PATH`.
The [libtoknroll README](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native) covers Windows, static linking and the thirteen functions of [`toknroll.h`](https://github.com/qxoticai/qxotic/blob/main/toknroll/toknroll-native/include/toknroll.h).

## With jinfer

[jinfer](https://pypi.org/project/jinfer/)'s `model.tokenizer` is this same class, served by the toknroll inside libjinfer and configured through `jinfer.configure`.
Both wheels can share a process; their libraries never interact.

## Links

- [Tok'n'Roll](https://github.com/qxoticai/qxotic/tree/main/toknroll): the library, its parity tests, the tested models and the known divergences.
- [libtoknroll](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native): the shared library and its C API.
- [The benchmark](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native/python/benchmarks) and the [issue tracker](https://github.com/qxoticai/qxotic/issues).

## License

Apache 2.0, by [Quixotic AI](https://qxotic.ai), for the package and the library's own code.
The library in the wheel is built with Oracle GraalVM and carries its Native Image runtime under the [GraalVM Free Terms and Conditions](https://www.oracle.com/downloads/licenses/graal-free-license.html), spelled out in the wheel's NOTICE.
Use it as you like, commercially too; a product sold for a fee that bundles a copy of it is not covered by those terms, and takes a build with GraalVM CE, see the [libtoknroll README](https://github.com/qxoticai/qxotic/tree/main/toknroll/toknroll-native#license).
