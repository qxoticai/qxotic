# Tok'n'Roll

**Token-perfect LLM tokenization.** Pure Java. Zero dependencies.

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Tok'n'Roll matches the reference Python tokenizers token for token, verified by parity tests, while
staying competitive with native implementations. No C extensions, no Rust bindings, no JNI. Java
11+, and it compiles to GraalVM native images out of the box.

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
decodes and counts from any source: HuggingFace, ModelScope, GGUF or local files.

```bash
jbang toknroll@qxoticai --source google/gemma-4-e2b-it --input "Hello, Tok'n'Roll 🎸"
echo "Hello\!" | jbang toknroll@qxoticai --count --source Qwen/Qwen3.6-35B-A3B
```

## Why Tok'n'Roll

- **Token-perfect.** Byte-exact parity with the reference tokenizers for
  [15 model families](#tested-implementations), not "close enough".
- **Fast.** Guaranteed worst-case `O(n log n)` BPE merging, optimized fast paths per model family,
  and zero-allocation, zero-copy APIs (`encodeInto`, `decodeBytesInto`) for hot loops.
- **Loads existing files.** HuggingFace `tokenizer.json`, ModelScope, and GGUF model files.
- **Composable.** Assemble tokenizers from reusable parts (vocabulary, splitter, model), or build
  one from scratch.

## Install

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>    <!-- tokenizer.json loading (HF / ModelScope) -->
  <version>0.2.0</version>
</dependency>
```

`toknroll-core` is the zero-dependency API and BPE engine. `toknroll-hf` loads `tokenizer.json`
files and `toknroll-gguf` loads tokenizers straight out of llama.cpp model files. Per-module
details and advanced usage: [toknroll-hf](toknroll-hf/), [toknroll-gguf](toknroll-gguf/).

## Build a tokenizer from scratch

```java
Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Toknroll.tiktokenModel(vocab, mergeRules);
Splitter splitter = Splitter.regex(Pattern.compile(/* e.g. the cl100k_base pattern */));

Tokenizer tokenizer = Toknroll.pipeline(splitter, model);
```

## Tested implementations

Token-perfect, backed by parity tests against the reference Python tokenizers:

- **OpenAI**: tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o), gpt-oss
- **Google**: Gemma 4
- **Alibaba**: Qwen 3.5+
- **Moonshot AI**: Kimi 2.5+
- **DeepSeek**: DeepSeek 3.2, DeepSeek 4
- **Mistral AI**: Tekken
- **IBM**: Granite 4+
- **Meta**: Llama 3+
- **Microsoft**: Phi 4+
- **HuggingFace**: SmolLM3
- **NVIDIA**: Nemotron 3
- **Z.ai**: GLM 5.1
- **MiniMax**: M2.7
- **Xiaomi**: MiMo V2
- **Poolside**: Laguna XS 2.1

Other BPE-based tokenizers likely work. They just are not parity-tested.

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
effect, then the drivers in `toknroll-benchmarks` write to `bench-output/` under the shared cache
root. Details in the [documentation](https://qxotic.ai/toknroll).

Part of [Quixotic](../README.md), an open stack for local AI on the JVM.

## License

Apache 2.0
