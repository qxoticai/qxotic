# Tok'n'Roll

[![Java](https://img.shields.io/badge/Java-17+-blue)](https://openjdk.org/projects/jdk/17/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Tok'n'Roll is a fast, pure Java tokenizer library for LLMs. No native dependencies, no JNI, no Python runtime. Just a JAR that works everywhere Java 17+ runs.

## Why Tok'n'Roll?

- **Fast.** Hand-optimized fast paths for common model families. Competitive with native tokenizers.
- **Pure Java, zero dependencies.** No C extensions, no Rust bindings, no JNI. The core library has no external dependencies.
- **Easy to extend.** Add a new tokenizer in minutes: plug in your split regex, vocabulary, and merge rules. The library handles the rest.
- **Clean, composable API.** Build tokenizers from reusable components. Start simple, stay clean, but keep the flexibility to handle edge cases when you need it.


## Benchmarks

Tok'n'Roll API accomodates for zero-allocation, zero-copy implementations. Fast paths avoid regex overhead for common ASCII patterns, and the BPE merge engine is optimized for both small and large inputs guaranteed worst-case O(n lg n) complexity.  
Single-thread benchmarks:

<img width="1423" height="559" alt="Image" src="https://github.com/user-attachments/assets/d92e330b-5555-406d-add6-6880ed4a4438" />

<img width="1423" height="559" alt="Image" src="https://github.com/user-attachments/assets/bcf0024a-2ea3-456f-8da6-cd599fc61f55" />

Note about multi-threading: Tok'n'Roll tokenizers are trivial to parallelize via batching and the strongly recommended way to do it.  
While discouraged, the Tok'n'Roll API perfectly supports multi-threaded implementations (no batching), as some other libraries do.

<img width="1423" height="476" alt="Image" src="https://github.com/user-attachments/assets/08592366-6dd0-4b1a-aa43-2a4fec2ab191" />

<img width="1423" height="476" alt="Image" src="https://github.com/user-attachments/assets/284666d1-41d9-4673-bfa7-24f22ddab274" />


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

