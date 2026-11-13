# Tok'n'Roll

**Token-perfect LLM tokenization**, against each model's own tokenizer. Pure Java. Zero dependencies.

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Tok'n'Roll matches the reference Python tokenizers token for token, verified by parity tests, with the [known divergences](#known-divergences) below, while staying competitive with native implementations.
No C extensions, no Rust bindings, no JNI.
Java 11+, and it compiles to GraalVM native images out of the box.

## Tokenize in three lines

```java
Tokenizer t = HuggingFaceTokenizerLoader.fromHuggingFace("google", "gemma-4-e2b-it");

int[] tokens = t.encodeToArray("Hello, world!");
String text  = t.decode(tokens);
int count    = t.countTokens("How many tokens is this?");
```

Remote loading fetches tokenizer metadata only, never model weights, and caches it on disk.

## Or from the command line

A [JBang CLI](https://github.com/qxoticai/qxotic/blob/main/toknroll/scripts/toknroll.java) encodes,
decodes and counts from any source: Hugging Face, ModelScope, GGUF or local files.

```bash
jbang toknroll@qxoticai --source google/gemma-4-e2b-it --input "Hello, Tok'n'Roll 🎸"
echo 'Hello, World!' | jbang toknroll@qxoticai --count --source Qwen/Qwen3.6-35B-A3B
```

## Or from C and Python

`libtoknroll` is Tok'n'Roll as a shared library with a C API, [`toknroll.h`](toknroll-native/include/toknroll.h): thirteen functions, the same sources, built with GraalVM Native Image.
The `toknroll` Python package binds it, with no Python ML stack behind it.

```python
import toknroll

tokenizer = toknroll.load("Qwen/Qwen3-0.6B")
ids = tokenizer.encode("Hello, world!")
tokenizer.decode(ids)
```

See [toknroll-native](toknroll-native/).

## Why Tok'n'Roll

- **Token-perfect.** Byte-exact parity with each model's own tokenizer, for [18 model vendors](#tested-implementations) and with the [known divergences](#known-divergences), not "close enough".
- **Fast.** Guaranteed worst-case `O(n log n)` BPE merging, optimized fast paths per model family,
  and zero-allocation, zero-copy APIs (`encodeInto`, `decodeBytesInto`) for hot loops.
- **Loads existing files.** Hugging Face `tokenizer.json`, ModelScope, and GGUF model files.
- **Composable.** Assemble tokenizers from reusable parts (vocabulary, splitter, model), or build
  one from scratch.

## Install

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>    <!-- tokenizer.json loading (HF / ModelScope) -->
  <version>0.4.0-rc1</version>
</dependency>
```

`toknroll-core` is the zero-dependency API and BPE engine. `toknroll-hf` loads `tokenizer.json`
files and `toknroll-gguf` loads tokenizers straight out of llama.cpp model files. Per-module
details and advanced usage: [toknroll-hf](toknroll-hf/), [toknroll-gguf](toknroll-gguf/).

`toknroll-vector` is optional and needs Java 25, and it is where the speed is.
Add it as a runtime dependency and run with `--add-modules jdk.incubator.vector`, and the byte-level BPE tokenizers whose pre-tokenization core recognizes encode through it.
The tiktoken family (GPT-2, GPT-4, Llama 3, Qwen2, GPT-4o, Tekken) runs in it entirely: a pre-tokenization split written as bitstream programs over the Java Vector API, a word cache shared by all threads, and merge loops over flat arrays.
For the other patterns core knows, such as Qwen 3.5 and DeepSeek, core splits the text and the engine merges the words.
Core's own encoder is the reference it is checked against when it loads: same API, same token ids, several times slower.
Details in [toknroll-vector](toknroll-vector/).

## Build a tokenizer from scratch

```java
Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Toknroll.tiktokenModel(vocab, mergeRules);
Splitter splitter = Splitter.regex(Pattern.compile(/* e.g. the cl100k_base pattern */));

Tokenizer tokenizer = Toknroll.pipeline(splitter, model);
```

## Tested implementations

Token-perfect against each model's own tokenizer, with the [known divergences](#known-divergences) below, backed by parity tests against the Hugging Face and llama.cpp reference tokenizers:

- **OpenAI**: tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o), gpt-oss
- **Google**: Gemma 4
- **Alibaba**: Qwen 3.5+
- **Liquid AI**: LFM 2.5, LFM 3
- **OpenBMB**: MiniCPM5
- **Moonshot AI**: Kimi 2.5+, Kimi K3
- **DeepSeek**: DeepSeek 3.2, DeepSeek 4
- **Mistral AI**: Tekken
- **IBM**: Granite 4+
- **Meta**: Llama 3+
- **Hugging Face**: SmolLM3
- **NVIDIA**: Nemotron 3
- **Z.ai**: GLM 5.1
- **MiniMax**: M2.7
- **Xiaomi**: MiMo V2
- **Poolside**: Laguna XS 2.1, Laguna S 2.1
- **inclusionAI**: Ling 3.0, Ling 3.1
- **JetBrains**: Mellum 2

Another BPE tokenizer built from the same components loads too; it is just not parity-tested.
A `tokenizer.json` that uses a component outside that set is refused at load as unsupported, the field named: a tokenizer is never loaded approximately.
Common models that do not load, and why:

| Model | Refused for |
|---|---|
| phi-4, RoBERTa | added tokens with `lstrip` or `rstrip` |
| TinyLlama | the `Prepend` normalizer |
| SmolLM2, StarCoder2, DeepSeek-LLM | the `Digits` pre-tokenizer |
| Falcon | the `Punctuation` pre-tokenizer |
| Pythia, OLMo, BLOOM | a vocabulary without every single-byte token |
| Yi | a decoder sequence the loader does not know |
| BERT, XLM-R, ALBERT | WordPiece, not BPE |
| T5 | Unigram, not BPE |
| OPT, ChatGLM, Baichuan, InternLM | no `tokenizer.json`: `vocab.json` plus `merges.txt`, or `tokenizer.model`, are not read |

## Raw text and special tokens

`encode` takes its input as raw text: no added token is matched, special or not, as with tiktoken's `encode_ordinary`.
`Specials.of(tokenizer).encode(tokenizer, text)` matches every added token in the text and substitutes its id, as Hugging Face's `encode` and llama.cpp with `parse_special` do.
Use it for text you wrote, such as a rendered chat template, never for a user's text.
Each mode is parity-tested against its own reference: `encode` against the reference with its added tokens switched off, `Specials` against the reference as it is.
[docs/SPECIALS.md](docs/SPECIALS.md) has the usage and the behavioral notes.

## Known divergences

Where llama.cpp differs from the model's own tokenizer, Tok'n'Roll follows the model's tokenizer:

- llama.cpp's case-rewriting regexes for gpt-oss, Nemotron and Kimi split words at non-ASCII capitals.
- llama.cpp's Laguna regex cuts `\r` from `\n`.
- llama.cpp matches gpt-oss's Harmony control tokens even in raw text.
- llama.cpp retypes a normal token named like an end-of-generation marker (Qwen2's `</s>`) as a control token, so `parse_special` matches it.

One difference is the file's own: a Qwen GGUF carries no normalizer, so it encodes decomposed accents (`e` + U+0301) as llama.cpp does, where the same model's `tokenizer.json` first composes them (NFC).
Another is the Unicode version: Tok'n'Roll knows the characters of Java 25's Unicode 16, Hugging Face tokenizers 1.0 those of Unicode 17, so a character added in 17 (2025), unassigned here, can split or normalize differently, such as the space before U+1F6D8.

## Benchmarks

Single-threaded encode and decode against the usual suspects. Competitive with native tokenizers,
with a guaranteed `O(n log n)` worst case where others degrade:

<img width="1424" height="536" alt="Single-threaded encode benchmark" src="https://github.com/user-attachments/assets/1ef13e40-1bee-4cb3-9c88-48e9b05b15f5" />

<img width="1424" height="536" alt="Single-threaded decode benchmark" src="https://github.com/user-attachments/assets/29ff3107-8d81-465f-ad93-f3bd3bca275b" />

Parallelize by batching, which this API makes trivial. Multi-threaded tokenization is supported but
rarely the right answer:

<img width="1424" height="451" alt="Multi-threaded encode benchmark" src="https://github.com/user-attachments/assets/69d543d8-be25-4c1f-8163-b159d3daadd8" />

<img width="1425" height="451" alt="Multi-threaded decode benchmark" src="https://github.com/user-attachments/assets/753543f6-146e-454e-94d7-8221b2f7a736" />

To reproduce them: `make toknroll-fixtures` fetches the enwik corpora explicitly, never as a side
effect, then the drivers in [`toknroll-benchmarks`](toknroll-benchmarks/) write to `bench-output/`
under the shared cache root.

Part of [Quixotic](../README.md), an open stack for local AI on the JVM.

## License

Apache 2.0
