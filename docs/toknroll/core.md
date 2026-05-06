---
sidebar_position: 2
---

# Core API

The core module (`toknroll-core`) is an **extensible tokenizer API** with efficient implementations of the two dominant BPE algorithms used by modern LLMs: **Tiktoken** (OpenAI GPT family) and **SentencePiece BPE** (Llama, Mistral, Gemma, DeepSeek, Qwen, etc.). Tiktoken is just an optimisation of BPE. Same foundation, different merge rule representation.

The API is extensible (you can implement your own `Splitter`, `Normalizer`, or `TokenizationModel`) but ships with these two implementations. No BERT, WordPiece, Unigram, or other NLP-focused tokenization. Targets LLM inference workloads.

Zero external dependencies. Java 11+.

## Philosophy

Tok'n'Roll takes a **composable** and **spartan** approach to tokenization:

**Composable.** Every tokenizer is built from reusable, single-purpose components. A `Normalizer` preprocesses text, a `Splitter` partitions it into chunks, and a `TokenizationModel` encodes each chunk. No hidden coupling. Swap any component independently.

**Spartan.** Tokenizers do one thing: encode text into token IDs and decode token IDs back into text. They do **not** inject control tokens, modify input, or alter output in arbitrary ways. Control token handling is explicit and opt-in via the `Specials` API, separate from the core encode/decode path.

**Zero-allocation.** Hot paths avoid object creation. `encodeInto(CharSequence, int start, int end, IntSequence.Builder)` encodes a slice of a `CharSequence` without copying. `Splitter` emits ranges via `SplitConsumer` callbacks rather than allocating `String` objects. The returned `IntSequence` is an `int`-backed read-only view, not a boxed `List<Integer>`.

## Tokenizer

The central interface. All tokenizer instances implement `Tokenizer`.

### Encode / Decode

```java
// Encoding. Returns IntSequence (lightweight int[] view, not boxed List)
IntSequence seq = tokenizer.encode("Hello, world!");
int[] tokens = tokenizer.encodeToArray("Hello, world!");

// Decoding
String text = tokenizer.decode(tokens);
byte[] bytes = tokenizer.decodeBytes(tokens);

// Counting
int count = tokenizer.countTokens("How many tokens?");
int byteCount = tokenizer.countBytes(tokens);

// Preallocation hint. Estimate tokens per input char to size your builder
int expected = tokenizer.expectedTokensPerChar();
```

### Zero-allocation encoding

Encode a slice of a `CharSequence` into a reusable `IntSequence.Builder`. No string copies, no allocation on the hot path.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="zero-alloc-encode"
```

### Zero-allocation decoding

Decode token IDs directly into a pre-allocated `ByteBuffer`.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="zero-alloc-decode"
```

## Toknroll

Static factory for building tokenizer components. Every tokenizer starts here.

### Vocabulary

`Vocabulary` is a bidirectional token-ID mapping with insertion-order iteration.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="vocabulary-iterate"
```

```java
// Lookups
int id = vocab.id("<|endoftext|>");       // -1 if not found
String token = vocab.token(50256);         // null if not found
boolean has = vocab.contains("<|endoftext|>");
int size = vocab.size();
```

### Models

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="model-create"
```

A `TokenizationModel` extends `Tokenizer`. It accepts pre-split, pre-normalized text chunks. Guarantees `decode(encode(chunk)).equals(chunk)` for ordinary text.

### Pipeline

Pipelines compose normalization, splitting, and model into a single `Tokenizer`. Components can be introspected at runtime.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="pipeline-compose"
```

```java
TokenizationPipeline p1 = Toknroll.pipeline(splitter, model);
TokenizationPipeline p2 = Toknroll.pipeline(normalizer, model);
```

## Splitter

A `@FunctionalInterface` that partitions input text into half-open ranges. Each range is delivered via a `SplitConsumer(int start, int end)` callback. Zero object allocation.

### Regex splitter

The most common pattern. Used by GPT-family tokenizers.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="splitter-regex"
```

### Composing splitters

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="splitter-compose"
```

```java
// Identity. No splitting. Full input is one chunk
Splitter identity = Splitter.identity();
```

## Normalizer

Preprocessing transformation applied before splitting and encoding. Normalizers are intentional and explicit. Tokenizers don't silently modify input.

### Common normalizers

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="normalizer-variants"
```

### Composing normalizers

Applied left-to-right.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="normalizer-compose"
```

## IntSequence

Read-only sequence of `int` values (token IDs). A lightweight alternative to `List<Integer>` with zero boxing overhead.

### Building

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="intsequence-builder"
```

### Access and manipulation

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="intsequence-ops"
```

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="intsequence-create"
```

### Iteration

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="intsequence-iterate"
```

```java
// Zero-allocation: copy into existing array
seq.copyTo(targetArray, offset);
```

## Specials

Special-token matcher for **explicit, opt-in** control-token injection. Tok'n'Roll's core `Tokenizer` does not inject special tokens. Use `Specials` when you need them.

### Compile and encode

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="specials-encode"
```

### Zero-allocation encoding

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="specials-encode-into"
```

### No specials (passthrough)

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="specials-none"
```

- Matches raw input **before** normalizer/splitter preprocessing
- Non-special spans still go through normal tokenizer encoding
- Separate from `Tokenizer` by design. Call sites stay auditable
- `tokenizer.encode(text)` = tiktoken `encode_ordinary`, `specials.encode(tokenizer, text)` = tiktoken `encode(allowed_special="all")`

## MergeRule

A BPE merge rule: a pair of token IDs with a rank. Lower rank = higher merge priority.

```java
MergeRule rule = MergeRule.of(leftId, rightId, rank);
```

## ByteLevel

GPT-2 byte-level encoding. Maps each byte (0–255) to a distinct printable Unicode character, producing a lossless, human-safe string from arbitrary binary data. Used by GPT-family tokenizers to represent text before BPE merges.

The encoding is reversible: `ByteLevel.decode(ByteLevel.encode(bytes))` always produces the original bytes.

```snippet path="toknroll/toknroll-core/src/test/java/com/qxotic/toknroll/Snippets.java" tag="bytelevel"
```

```java
// Validate without decoding
boolean valid = ByteLevel.isValidEncoding(symbols);

// Single byte/char convenience
String sym = ByteLevel.encodeSingle((byte) 65);  // "A"
byte b = ByteLevel.decodeSingle('A');              // 65
```
