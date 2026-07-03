package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;

import java.util.ArrayDeque;

/** Tier-1 caching: the last N live conversations, kept as {@link CachedSession}s with their
 *  states resident (the llama.cpp-slot equivalent). A pooled session whose whole fingerprint
 *  stream is a strict prefix of the incoming conversation continues APPEND-ONLY on its live
 *  state — zero restore cost, only the delta is ingested. A mid-stream divergence cannot be
 *  reused (recurrent state cannot rewind), so it falls through to the shared {@link PromptCache}
 *  (tier 2); pooled sessions commit their blocks there as they go, so pool eviction loses
 *  nothing but the zero-copy continuation.
 *
 *  <p>Memory bound: N live states (each holds KV for its full context) — size the pool like
 *  llama.cpp slots. Single-threaded by design (the generation worker), like the cache. */
public final class SessionPool<S extends RuntimeState> {

    private final int capacity;
    private final ArrayDeque<CachedSession<S>> pool = new ArrayDeque<>();

    /** @param capacity live sessions retained; 0 disables the pool (every request is tier 2). */
    public SessionPool(int capacity) {
        this.capacity = Math.max(0, capacity);
    }

    /** The pooled session with the LONGEST stream that is a strict prefix of
     *  {@code fingerprints[0..len)} (append-only reuse: at least one position remains to ingest,
     *  so the logits are refreshed) and whose state can hold {@code len} positions. The session
     *  is removed from the pool while in use — {@link #release} returns it. Null = no tier-1
     *  match; resume from the block cache instead. */
    public CachedSession<S> acquire(long[] fingerprints, int len) {
        CachedSession<S> best = null;
        for (CachedSession<S> s : pool) {
            if (s.streamIsStrictPrefixOf(fingerprints, len)
                    && len <= s.state().contextCapacity()
                    && (best == null || s.length() > best.length())) {
                best = s;
            }
        }
        if (best != null) pool.remove(best);
        return best;
    }

    /** Returns a session to the pool as most-recent; past capacity the least-recent is dropped
     *  (its state is freed; its blocks remain in the shared {@link PromptCache}). */
    public void release(CachedSession<S> session) {
        if (capacity == 0) return;
        pool.addLast(session);
        while (pool.size() > capacity) pool.removeFirst();
    }

    public int size() {
        return pool.size();
    }
}
