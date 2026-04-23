# Tok'n'Roll

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A clean, extensible API with optimized BPE implementations for LLM tokenizers. Pure Java, zero native dependencies.

## Why Tok'n'Roll?

- **Fast.** Optimized fast paths for common model families. Competitive with native tokenizers.
- **Pure Java, zero dependencies.** No C extensions, no Rust bindings, no JNI. The core library has no external dependencies.
- **Clean, composable API.** Build tokenizers from sound, reusable components. Pristine by default, with pragmatic escape hatches when you need them.

## Benchmarks

#### Single-thread
These benchmartks show the standard encode/decode paths (string-to-tokens) using a single thread. The BPE merge engine ensures guaranteed worst-case `O(n log n)` complexity.  
**Note:** Zero-allocation, zero-copy APIs are available for even higher throughput.

<img width="1424" height="536" alt="Image" src="https://github.com/user-attachments/assets/1ef13e40-1bee-4cb3-9c88-48e9b05b15f5" />

<img width="1424" height="536" alt="Image" src="https://github.com/user-attachments/assets/29ff3107-8d81-465f-ad93-f3bd3bca275b" />


#### Multi-thread
The recommended way to parallelize Tok'n'Roll tokenizers is via batching, which is trivial to implement.  
While discouraged, the Tok'n'Roll API also supports multi-threaded implementations (no batching), as some other libraries do.

<img width="1424" height="451" alt="Image" src="https://github.com/user-attachments/assets/69d543d8-be25-4c1f-8163-b159d3daadd8" />

<img width="1425" height="451" alt="Image" src="https://github.com/user-attachments/assets/753543f6-146e-454e-94d7-8221b2f7a736" />


## Quick Start

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll</artifactId>
  <version>0.1.0</version>
</dependency>
```

```java
// Build a tokenizer from your model files
Vocabulary vocab = Tokenizers.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Tokenizers.tikTokenModel(vocab, mergeRules);

Tokenizer tokenizer = Tokenizers.pipeline(model)
    .splitter(FastSplitters.cl100k())  // or r50k(), o200k(), llama3(), qwen35(), ...
    .build();

// Use it
int[] tokens = tokenizer.encodeToArray("Hello world");
String text = tokenizer.decode(tokens);
```

## Tested Implementations

The test suite includes verified, token-perfect implementations backed by comprehensive parity tests against the Python implementations for:

- **OpenAI** – tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o)
- **Google** – Gemma 3 & 4
- **Alibaba** – Qwen 3.5+
- **Moonshot AI** – Kimi 2.5+
- **DeepSeek** – DeepSeek 3.2
- **Mistral AI** – pre-Tekken and Tekken tokenizers
- **IBM** – Granite 4+
- **Meta** – Llama 3+
- **Microsoft** – Phi 4+
- **HuggingFace** – SmolLM3

