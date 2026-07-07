# jinfer architecture

One page: the module graph, the chat-to-tokens flow, and the prompt-cache design.

## Modules

```
jinfer-core     types + seams: Batch, FloatTensor, Model/LanguageModel/RuntimeState,
                GgufTokenizer, chat/ (layout), cache/ (prompt caching), media/ (codecs)
jinfer-jinja    Jinja template engine (CompiledTemplate) - used by the server fallback
                and as the OFFLINE oracle for hand-written templates; never on the hot path
jinfer-kernels  ModelLoader, FlashAttention, GEMM dispatch (JAM native / Vector API);
                test-jar ships the shared testkit (Harness + scenario batteries)
jinfer-<model>  one module per curated model (lfm2, gemma4, llama, gptoss, qwen35, ...):
                the arch port + its TurnTemplate + its StateCodec
jinfer-server   OpenAI-compatible HTTP server on top
```

Dependencies flow strictly downward (core &larr; jinja &larr; kernels &larr; models &larr; server); no cycles.

## Chat flow: conversation to tokens to KV

```
Message(role, [Text|Blob(Media)])                    high-level, portable
  -> TurnTemplate.encodeTurn / generationPrompt      per-turn, deterministic, turn-stable
  -> List<Batch>  (Tokens | Embeddings)              text plain-encoded, media via MultiModal
  -> Batch.prepare(batches, batchCapacity)           merge/split, bidirectional blocks whole
  -> model.ingest(state, batch)                      KV grows at state.position()
```

Templates are hand-written per model (curated set) and validated **token-exact** against the
GGUF's own `tokenizer.chat_template` by the oracle tests. Two tokenization domains: scaffolding
is trusted special ids; conversation text is plain-encoded and can never mint control tokens.
Models without a hand-written template fall back to the server's Jinja render path.

## Prompt cache: policy over two seams

The model provides a `StateCodec` (resume-state for a position span &harr; opaque bytes: per-position
K/V rows for attention layers; fixed-size checkpoints for short-conv, SSM, or sliding-window
layers - restored at true ring slots since RoPE is position-baked). Storage provides a
`CacheStore`. `PromptCache` owns the rest:

- **Chained crypto keys**: block key = SHA-256(parent key, span fingerprints); fingerprints are
  token ids (media: content hash). The chain names the whole prefix - trusted as identity.
- **Per-model seed**: the chain ROOT is seeded with the GGUF's identity
  (`PromptCache.modelSeed`: length + head/tail SHA-256), so caches are per-model by construction
  and a wrong cache file fails with a descriptive error, never a wrong restore.
- **Complete blocks only**: checkpoints exist only at boundaries, so a block matches whole or
  not at all. Prefill chunks are large blocks; each decode step is a single-token block
  (`CachedSession` commits at every ingestion boundary via the O(span) `Cursor`).
- **Pure optimization**: any miss, eviction, or verification failure degrades to recompute.

Serving artifacts: `SealedPrompt` (ONE compiled prompt: mmap, memcmp fingerprint verify, instant
restore - the prompt-compiler path for native images) and frozen `PromptCache` files
(`freeze()`/`open()`: read-only, lazily mapped, several prompts with shared-prefix dedup).
Measured on CPU: resume is near-constant in history length (~20-50 ms) vs linear prefill -
84-404x on multi-thousand-token histories.

## Testing

Main-method harnesses per model, parameterized by the shared testkit: oracle battery
(token-exact vs the Jinja render, injection-inert), cache battery (byte-identical restored
state - the sound gate; reply equality strict for deterministic decodes, informational for MoE
whose threaded reductions are not byte-deterministic), sealed/frozen batteries, benchmarks.
