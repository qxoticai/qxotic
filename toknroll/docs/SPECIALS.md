# Specials API

`Specials` provides an explicit special-token path for inference workloads: the tokens a vocabulary types control, user-defined or unknown, matched whole.

## Why it is separate from Tokenizer

- `Tokenizer` remains ordinary-text oriented and focused on round-trip integrity.
- Control-token handling stays explicit and opt-in instead of being hidden in default encode behavior.
- Call sites remain auditable: plain text encoding and special-aware encoding are distinct operations.

## The references' terms

- `tokenizer.encode(text)` is raw text: tiktoken's `encode_ordinary`, Hugging Face's tokenizer without its added tokens, llama.cpp with no added token matched.
- `specials.encode(tokenizer, text)` matches every added token: tiktoken's `encode(..., allowed_special="all")`, Hugging Face's `encode`, llama.cpp's `parse_special`.
- The parity tests check each mode against these references.

## Usage

```java
Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokens);
Tokenizer tokenizer = Toknroll.pipeline(splitter, Toknroll.tiktokenModel(vocab, mergeRules));

// Build once and reuse.
Specials specials = Specials.of(tokenizer); // the vocabulary's special tokens, by their text

IntSequence ids = specials.encode(tokenizer, "hello <|endoftext|>");
```

```java
IntSequence.Builder out = IntSequence.newBuilder();
specials.encodeInto(tokenizer, "prefix <|endoftext|> suffix", out);
IntSequence ids = out.build();
```

```java
// No specials configured: equivalent to tokenizer.encode(...)
Specials none = Specials.none();
IntSequence ids = none.encode(tokenizer, "just text");
```

## Behavioral notes

- Build `Specials` with `Specials.of(tokenizer)` from the same tokenizer used at encode time.
- Matching runs on raw input before tokenizer preprocessing (normalizer/splitter).
- Non-special spans still go through tokenizer preprocessing and encoding.
- If a `Specials` instance is used with an incompatible tokenizer vocabulary mapping, behavior is undefined.
