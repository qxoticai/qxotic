/**
 * Context caching behind one production entry point: {@link com.qxotic.jinfer.cache.PromptCache}.
 *
 * <p>Retained sessions keep a bounded number of live conversation states for append-only reuse. The
 * optional block layer stores content-addressed checkpoints under a byte budget and can mount or
 * grow one catalog file. Requests use the longest compatible prefix and otherwise recompute;
 * incompatible model identities never match.
 *
 * <p>Every restore stops one position short and re-ingests the final position, so logits are fresh
 * and cached generation is equivalent to cold generation. {@link
 * com.qxotic.jinfer.cache.CachedSession}, {@link com.qxotic.jinfer.cache.BlockTree} and {@link
 * com.qxotic.jinfer.cache.FrozenBlocks} are low-level building blocks for tests, benchmarks and
 * advanced runtimes; ordinary callers should stay on {@code PromptCache}.
 */
package com.qxotic.jinfer.cache;
