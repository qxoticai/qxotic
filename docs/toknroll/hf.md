---
sidebar_position: 4
---

# HuggingFace Loader

Loads a Tok'n'Roll `Tokenizer` from a HuggingFace `tokenizer.json`.

Unsupported features (e.g. `post_processor`, custom `decoders`) fail fast instead of introducing silent incompatibilities.

## Quick Start

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.nio.file.Path;

// From local files (directory or tokenizer.json path)
Tokenizer t = HuggingFaceTokenizerLoader
    .fromLocal(Path.of("/models/gemma-4-e2b-it"));

// From HuggingFace
Tokenizer t2 = HuggingFaceTokenizerLoader
    .fromHuggingFace("google", "gemma-4-e2b-it");

// From ModelScope
Tokenizer t3 = HuggingFaceTokenizerLoader
    .fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");

// Encode and decode
int[] tokens = t.encodeToArray("Hello, world!");
String decoded = t.decode(tokens);
```

## Sources

Local filesystem, [HuggingFace](https://huggingface.co), or [ModelScope](https://modelscope.cn).

Remote loading fetches the `tokenizer.json` (and optional `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`). If `tokenizer.json` is not found, the loader falls back to `tiktoken.model` reconstruction.

Artifacts are cached on disk and reused across runs.

## Supported Features

The loader is intentionally strict. Unsupported features fail fast.

**BPE models only** (`type: "BPE"`).

**Normalizers:**
- `NFC`, `NFD`, `NFKC`, `NFKD`
- `Lowercase`
- `Replace`
- `Sequence` (composed normalizers)

**Pre-tokenizers:**
- `Split` (Regex / String pattern)
- `Sequence` (composed pre-tokenizers)
- `ByteLevel` (GPT-2 byte-level encoding)
- `Metaspace` (SentencePiece ▁ replacement)

**Special tokens** from `added_tokens` with `special: true`.

## Tiktoken Fallback

When `tokenizer.json` returns 404, the loader reconstructs the tokenizer from:

- `tiktoken.model`: base64-encoded BPE model
- `tokenizer_config.json`: pre-tokenizer pattern (`pat_str`)
- Auto-resolves `pat_str` from Python module via `auto_map` (parses `pat_str = "|".join([...])` from `.py` files)

```java
// GPT-4o tokenizer (tiktoken format)
Tokenizer gpt4o = HuggingFaceTokenizerLoader
    .fromHuggingFace("Xenova", "gpt-4o");
```

## SentencePiece Detection

Automatic SentencePiece BPE detection when:

- A `Replace` normalizer replaces spaces with ▁ (U+2581)
- A `Metaspace` pre-tokenizer is configured

```java
// Llama tokenizer via HuggingFace (SentencePiece BPE)
Tokenizer llama = HuggingFaceTokenizerLoader
    .fromHuggingFace("meta-llama", "Llama-3.2-1B-Instruct");
```

## Configuration

| Key | Purpose |
|-----|---------|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory for downloaded artifacts |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace authentication token |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope authentication token |
