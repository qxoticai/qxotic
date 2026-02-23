# Tokenizers GGUF Module

Tokenizer loading support for GGUF (GGML Universal File Format) models.

Default behavior is strict: no lossy normalization transforms are applied unless you explicitly add them in advanced composition paths.

## Quick Start

```java
import com.qxotic.tokenizers.gguf.GGUFTokenizers;
import com.qxotic.tokenizers.Tokenizer;

// Simple one-liner
Tokenizer tokenizer = GGUFTokenizers.fromFile("/path/to/model.gguf");

// Or from String path
Tokenizer tokenizer = GGUFTokenizers.fromFile("model.gguf");
```

## GGUF Tokenizer Metadata

This module implements the [GGUF tokenizer specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#tokenizer).

### GGML Tokenizer Keys

The following keys describe the tokenizer in GGUF files:

#### tokenizer.ggml.model (string)
The name of the tokenizer model:
- `gpt2`: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF tokenizer.json)
- `llama`: Llama style SentencePiece (tokens and scores extracted from HF tokenizer.model)
- `replit`: Replit style SentencePiece (tokens and scores extracted from HF spiece.model)
- `rwkv`: RWKV tokenizer

#### tokenizer.ggml.tokens (array[string])
A list of tokens indexed by the token ID used by the model.

#### tokenizer.ggml.scores (array[float32]) - Optional
The score/probability of each token. If not present, all tokens are assumed to have equal probability. Must have the same length as tokens.

#### tokenizer.ggml.token_type (array[int32]) - Optional
The token type:
- `1`: Normal token
- `2`: Unknown token
- `3`: Control token
- `4`: User defined
- `5`: Unused
- `6`: Byte token

Must have the same length as tokens.

#### tokenizer.ggml.merges (array[string]) - Optional
The merges of the tokenizer (for BPE tokenizers). If not present, the tokens are assumed to be atomic.

#### tokenizer.ggml.added_tokens (array[string]) - Optional
Tokens that were added after training.

### Special Token IDs

- **tokenizer.ggml.bos_token_id** (uint32): Beginning of sequence marker
- **tokenizer.ggml.eos_token_id** (uint32): End of sequence marker
- **tokenizer.ggml.unknown_token_id** (uint32): Unknown token
- **tokenizer.ggml.separator_token_id** (uint32): Separator token
- **tokenizer.ggml.padding_token_id** (uint32): Padding token

### Pre-tokenizer

- **tokenizer.ggml.pre** (string): The pre-tokenizer name (e.g., `llama`, `qwen2`, `smollm`, `tekken`)

## Supported Pre-tokenizers

This module supports the following pre-tokenizer patterns:

- `llama`, `llama-bpe`, `mistral`, `mixtral`, `phi`, `phi3`, `dbrx`
- `qwen`, `qwen2`, `qwen3`
- `smollm`, `smollm2`
- `tekken`
- `refact`, `granite`
- `default`
- `gemma`, `gemma2`, `gemma3` (identity)

## Advanced Usage

### Custom Registries

```java
import com.qxotic.tokenizers.gguf.*;
import com.qxotic.tokenizers.advanced.Splitter;
import java.util.regex.Pattern;

// Get default registries
GGUFPreTokenizerRegistry preTokenizers = GGUFTokenizers.preTokenizers();
GGUFTokenizerRegistry tokenizers = GGUFTokenizers.tokenizers();

// Register custom pre-tokenizer
preTokenizers.register("my-pattern", Splitter.regex(Pattern.compile("\\p{L}+|\\p{N}+")));

// Optional shorthand when you already have a regex string
preTokenizers.register("my-pattern-2", Splitter.regex("\\p{L}+|\\p{N}+"));

// Register custom tokenizer factory
tokenizers.register("custom", (gguf, splitter) -> {
    // Create tokenizer from GGUF metadata
    return new MyTokenizer(gguf, splitter);
});

// Use custom registries
Tokenizer tokenizer = GGUFTokenizers.fromFile(
    path,
    preTokenizers,
    tokenizers
);
```

### Checking Support

```java
// Check if a pre-tokenizer + tokenizer model combination is supported
boolean supported = GGUFTokenizers.isRegistered("qwen2", "gpt2");
```

## Official Specification

For the complete GGUF specification, see:
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#tokenizer

## Note on Tokenization Accuracy

As noted in the GGUF specification:

> GGML supports an embedded vocabulary that enables inference of the model, but 
> implementations of tokenization using this vocabulary (i.e. llama.cpp's tokenizer) 
> may have lower accuracy than the original tokenizer used for the model. When a 
> more accurate tokenizer is available and supported, it should be used instead.

For best results, use the HuggingFace tokenizer when available via `tokenizers-hf`.
