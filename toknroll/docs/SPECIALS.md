# Specials API

`Specials` provides an explicit control-token path for inference workloads.

## Why it is separate from Tokenizer

- `Tokenizer` remains ordinary-text oriented and focused on round-trip integrity.
- Control-token handling stays explicit and opt-in instead of being hidden in default encode behavior.
- Call sites remain auditable: plain text encoding and special-aware encoding are distinct operations.

## Tiktoken mapping

- `tokenizer.encode(text)` is analogous to Tiktoken ordinary-text encoding (`encode_ordinary(...)`).
- `specials.encode(tokenizer, text)` is analogous to Tiktoken special-aware encoding (`encode(..., allowed_special="all")`).

## Usage

```java
Tokenizer tokenizer = Tokenizers.fastBpe(mergeableRanks, specialTokens, splitPatternRegex);

// Compile once and reuse.
Specials specials = Specials.compile(tokenizer.vocabulary(), specialTokens.keySet());

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

- Compile `Specials` with the same vocabulary exposed by the tokenizer used at encode time.
- Matching runs on raw input before tokenizer preprocessing (normalizer/splitter).
- Non-special spans still go through tokenizer preprocessing and encoding.
- If a `Specials` instance is used with an incompatible tokenizer vocabulary mapping, behavior is undefined.
