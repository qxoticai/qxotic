/**
 * Prompt caching: pure policy over two narrow seams.
 *
 * <p>The model contributes one {@link com.qxotic.jinfer.cache.KvCodec} - serialize/restore the
 * resume-state for a span of positions to opaque bytes (per-position KV rows for attention,
 * fixed-size checkpoints for recurrent/windowed layers). Storage contributes one
 * {@link com.qxotic.jinfer.CacheStore}. {@link com.qxotic.jinfer.cache.PromptCache} owns
 * everything else: blocks are content-addressed by a CHAINED SHA-256 key over per-position
 * fingerprints (trusted as identity, git/IPFS regime), seeded with the model's identity so caches
 * are per-model by construction; blocks match completely or not at all (checkpoints exist only at
 * boundaries); every miss degrades to recompute, never to a wrong answer.
 *
 * <p>{@link com.qxotic.jinfer.cache.CachedSession} binds the dual representation - the exact
 * fingerprint stream alongside the KV - and commits at every ingestion boundary (large blocks for
 * prefill chunks, single-token blocks for decode steps). Deployment artifacts:
 * {@link com.qxotic.jinfer.cache.SealedPrompt} (one compiled prompt, memcmp verify, instant
 * restore) and frozen {@code PromptCache} files ({@code freeze}/{@code open}: read-only,
 * lazily mapped, multi-prompt with shared-prefix dedup).
 */
package com.qxotic.jinfer.cache;
