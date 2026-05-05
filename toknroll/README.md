# Tok'n'Roll

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Token-perfect LLM tokenization. Pure Java. Zero dependencies.

## Why Tok'n'Roll?

- **Fast.** Optimized fast paths for common model families. Competitive with native tokenizers.
- **Pure Java, zero dependencies.** No C extensions, no Rust bindings, no JNI. The core library has no external dependencies.
- **Clean, composable API.** Build tokenizers from sound, reusable components.

## Benchmarks

#### Single-thread
These benchmarks show the standard encode/decode paths (string-to-tokens) using a single thread. The BPE merge engine ensures guaranteed worst-case `O(n log n)` complexity.  
**Note:** Zero-allocation, zero-copy APIs are available for even higher throughput.

<img width="1424" height="536" alt="Image" src="https://github.com/user-attachments/assets/1ef13e40-1bee-4cb3-9c88-48e9b05b15f5" />

<img width="1424" height="536" alt="Image" src="https://github.com/user-attachments/assets/29ff3107-8d81-465f-ad93-f3bd3bca275b" />


#### Multi-thread
The recommended way to parallelize Tok'n'Roll tokenizers is via batching, which is trivial to implement.  
While discouraged, the Tok'n'Roll API also supports multi-threaded implementations (no batching), as some other libraries do.

<img width="1424" height="451" alt="Image" src="https://github.com/user-attachments/assets/69d543d8-be25-4c1f-8163-b159d3daadd8" />

<img width="1425" height="451" alt="Image" src="https://github.com/user-attachments/assets/753543f6-146e-454e-94d7-8221b2f7a736" />


## Quick Start

### Maven

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-core</artifactId>
  <version>0.1.0</version>
</dependency>
```

`toknroll-core` is a high-level, generic tokenizer API with the core BPE algorithms and zero external dependencies. It does not include format-specific loading logic.

For loading tokenizers from HuggingFace, ModelScope, or GGUF files, add the loader modules:

```xml
<!-- HuggingFace tokenizer.json / tiktoken.model loading -->
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>
  <version>0.1.0</version>
</dependency>

<!-- GGUF (llama.cpp format) tokenizer loading -->
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-gguf</artifactId>
  <version>0.1.0</version>
</dependency>
```

### Build a tokenizer from scratch

```java
import java.util.regex.Pattern;
import com.qxotic.toknroll.*;

// Build a vocabulary from your model's ranked tokens
Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Toknroll.tiktokenModel(vocab, mergeRules);

// Choose a regex splitter (e.g. cl100k_base pattern for GPT-4)
Splitter splitter = Splitter.regex(Pattern.compile(
    "(?i:'(?:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)"));

// Assemble the pipeline
Tokenizer tokenizer = Toknroll.pipeline(splitter, model);

// Use it
int[] tokens = tokenizer.encodeToArray("Hello world");
String text = tokenizer.decode(tokens);
int count = tokenizer.countTokens("How many tokens is this?");
```


## Loading Tokenizers

Load tokenizers from HuggingFace, ModelScope, or local files.

```java
// HuggingFace tokenizer.json format
Tokenizer t = HuggingFaceTokenizerLoader
    .fromHuggingFace("google", "gemma-4-e2b-it");

// ModelScope
Tokenizer t = HuggingFaceTokenizerLoader
    .fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");

// GGUF format (llama.cpp)
GGUFTokenizerLoader ggufLoader = GGUFTokenizerLoader
    .createBuilderWithBuiltins()
    .build();

Tokenizer t = ggufLoader.fromHuggingFace(
    "unsloth", "Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q8_0.gguf");

// Local files
Tokenizer t = HuggingFaceTokenizerLoader
    .fromLocal(Path.of("/models/gemma-4-e2b-it"));

Tokenizer t = ggufLoader.fromLocal(
    Path.of("/models/model.gguf"));

// Encode / decode
int[] tokens = t.encodeToArray("Hello, world!");
String decoded = t.decode(tokens);
```

See [toknroll-hf](toknroll-hf/) and [toknroll-gguf](toknroll-gguf/) READMEs for per-model-family examples and advanced usage.

Remote loading fetches only tokenizer metadata, not model weights. Artifacts are cached on disk; cache location is configurable via `toknroll.cache.root` system property or `TOKNROLL_CACHE_ROOT` env var.


## Tested Implementations

The test suite includes verified, token-perfect implementations backed by comprehensive parity tests against the Python implementations for:

- **OpenAI** - tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o)
- **Google** - Gemma 3, Gemma 4
- **Alibaba** - Qwen 3.5+
- **Moonshot AI** - Kimi 2.5+
- **DeepSeek** - DeepSeek 3.2, DeepSeek 4
- **Mistral AI** - Tekken
- **IBM** - Granite 4+
- **Meta** - Llama 3+
- **Microsoft** - Phi 4+
- **HuggingFace** - SmolLM3

Other models using BPE-based tokenizers are likely to work but are not tested against reference Python tokenizers.
