package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Tier-1 caching: the last N live conversations, kept as {@link CachedSession}s with their states
 * resident (the llama.cpp-slot equivalent). A pooled session whose whole fingerprint stream is a
 * strict prefix of the incoming conversation continues APPEND-ONLY on its live state — zero restore
 * cost, only the delta is ingested. A mid-stream divergence cannot be reused (recurrent state
 * cannot rewind), so it falls through to the shared {@link PromptCache} (tier 2); pooled sessions
 * commit their blocks there as they go, so pool eviction loses nothing but the zero-copy
 * continuation.
 *
 * <p>Memory bound: N live states (each holds KV for its full context) — size the pool like
 * llama.cpp slots. Single-threaded by design (the generation worker), like the cache.
 */
public final class SessionPool<S extends RuntimeState> {

    private final int capacity;
    private final ArrayDeque<CachedSession<S>> pool = new ArrayDeque<>();

    /**
     * @param capacity live sessions retained; 0 disables the pool (every request is tier 2).
     */
    public SessionPool(int capacity) {
        this.capacity = Math.max(0, capacity);
    }

    /**
     * The pooled session with the LONGEST stream that is a strict prefix of {@code
     * fingerprints[0..len)} (append-only reuse: at least one position remains to ingest, so the
     * logits are refreshed) and whose state can hold {@code len} positions. The session is removed
     * from the pool while in use — {@link #release} returns it. Null = no tier-1 match; resume from
     * the block cache instead.
     */
    CachedSession<S> acquire(long[] fingerprints, int len) {
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

    /**
     * Returns a session to the pool as most-recent; past capacity the least-recent is dropped (its
     * state is freed; its blocks remain in the shared {@link PromptCache}).
     */
    void release(CachedSession<S> session) {
        if (capacity == 0) {
            close(session); // pooling disabled: free the state now, not at GC
            return;
        }
        pool.addLast(session);
        while (pool.size() > capacity) close(pool.removeFirst());
    }

    /** An owned state must be freed deterministically - a dropped one only degrades to GC. */
    private static void close(CachedSession<?> session) {
        if (session.state() instanceof com.qxotic.jinfer.BaseState base) base.close();
    }

    public int size() {
        return pool.size();
    }

    /**
     * The tier-1/tier-2 arbitration protocol, owned here so callers cannot mis-sequence it: acquire
     * a pooled session (tier 1, append-only) or resume a fresh state from the block cache (tier 2,
     * at most {@code resumeLimit} positions so the caller re-ingests the final block and gets fresh
     * logits), run {@code work}, and return the session to the pool ON SUCCESS ONLY. If {@code
     * work} throws, the session is discarded - a possibly-inconsistent state must never serve
     * future requests; its committed blocks remain in the shared cache, so nothing durable is lost.
     */
    public <R> R withSession(
            Model<?, ?, S> model,
            PromptCache<S> cache,
            Supplier<S> freshState,
            List<List<Batch>> groups,
            Work<S, R> work) {
        // flatten to the internal content stream; tier-2 resumes at most up to the FINAL group
        // (the generation prompt), so a whole-prompt hit still re-ingests it for fresh logits
        List<Batch> flat = new ArrayList<>();
        for (List<Batch> group : groups) flat.addAll(group);
        long[] fingerprints = CachedSession.fingerprints(flat);
        int lastGroup =
                groups.isEmpty()
                        ? 0
                        : groups.get(groups.size() - 1).stream().mapToInt(Batch::count).sum();
        int resumeLimit = fingerprints.length - lastGroup;
        CachedSession<S> session = acquire(fingerprints, fingerprints.length);
        boolean tier1 = session != null;
        if (!tier1) {
            session =
                    CachedSession.resume(model, cache, freshState.get(), fingerprints, resumeLimit);
        }
        R result;
        try {
            result = work.run(session, tier1);
        } catch (RuntimeException | Error e) {
            close(session); // a torn state must never serve again - and must not wait for GC
            throw e;
        }
        release(session);
        return result;
    }

    /** Body run against the acquired-or-resumed session; {@code tier1} = pooled append-only. */
    @FunctionalInterface
    public interface Work<S extends RuntimeState, R> {
        R run(CachedSession<S> session, boolean tier1);
    }
}
