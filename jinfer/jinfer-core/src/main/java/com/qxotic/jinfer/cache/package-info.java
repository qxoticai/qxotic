/**
 * jinfer's KV cache. Two layers, one law, one front door.
 *
 * <p>{@link com.qxotic.jinfer.cache.PromptCache} is the front door - the only class production
 * callers need. HOT: the last N conversations stay live as ready-to-continue states; a prompt that
 * strictly extends one continues in place (any model, codec or not). BLOCKS: everything computed is
 * kept as content-keyed KV blocks under a byte budget, optionally on a catalog file that survives
 * restarts. A request is served from the hottest thing that matches; THE LAW: resume always stops
 * one position short, so logits are always fresh - and a cached answer is byte-identical to a cold
 * one. Every miss degrades to recompute, never to a wrong answer.
 *
 * <p>The low-level layer underneath (public for the testkit, benches and speculative decoding): the
 * model contributes one {@link com.qxotic.jinfer.cache.StateCodec} - serialize/restore the
 * resume-state for a span of positions to opaque bytes (per-position KV rows for attention, a
 * fixed-size residue trailer for recurrent state); storage contributes one {@link
 * com.qxotic.jinfer.cache.CacheStore}. {@link com.qxotic.jinfer.cache.BlockTree} owns the block
 * physics: content addressing by a CHAINED SHA-256 key over per-position fingerprints (trusted as
 * identity, git/IPFS regime), seeded with the model's identity so caches are per-model by
 * construction; blocks match completely or not at all (self-contained; every boundary is a resume
 * point); budget/LRU-leaf eviction. {@link com.qxotic.jinfer.cache.CachedSession} binds the dual
 * representation - the exact fingerprint stream alongside the KV - committing large blocks for
 * prefill chunks and single-token blocks for the decode tail. {@link
 * com.qxotic.jinfer.cache.FrozenBlocks} is the on-disk artifact format (read-only, lazily mapped,
 * multi-prompt with shared-prefix dedup) behind the facade's catalog, save and export.
 */
package com.qxotic.jinfer.cache;
