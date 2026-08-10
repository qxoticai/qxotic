package com.qxotic.jinfer.x.boundary;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.ConcurrentModificationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The shared cursor of every model State: how far the sequence has been ingested and what the last
 * ingest retained. Owns the {@link #advance}/{@link #resumeAt} lifecycle so each model's State
 * carries no cursor boilerplate. Fields are public for the model's own forward code (hot-path reads
 * like {@code s.position} across packages); mutation belongs to the two lifecycle methods only.
 *
 * <p>Also owns the state's MEMORY lifetime. Every buffer of a state comes from its {@link #arena};
 * who provided the arena owns it: {@code newState(ctx, batch)} builds an internal owned arena that
 * {@link #close()} frees deterministically (with a {@link Cleaner} backstop, so a dropped unclosed
 * state degrades to GC-eventually rather than leaking); {@code newState(ctx, batch, arena)} borrows
 * the caller's arena and {@link #close()} never touches it - close YOUR arena only after your last
 * call returns (closing it first sequentially is caught fail-fast by {@link #enter}'s canary;
 * closing it DURING a computation is a data race the kernels' raw reads can turn into a crash);
 * {@code newState(ctx, batch, arena, true)} ADOPTS the caller's arena, fusing its lifetime into the
 * state's (close frees it, co-tenants like weights included).
 *
 * <p>One lock carries the three run-time laws: entry points {@code tryLock} so two concurrent
 * computations fail fast with {@link ConcurrentModificationException} (the single-serial-pipeline
 * contract - never queued, that would hide the bug); {@link #close()} {@code lock()}s so it BLOCKS
 * until the in-flight call returns and is therefore the caller's quiescence certificate; entries
 * after close fail with {@link IllegalStateException} before any kernel touches freed memory.
 */
public abstract class BaseState implements RuntimeState, AutoCloseable {

    private static final Cleaner CLEANER = Cleaner.create();

    /** Every buffer of this state allocates from here; ownership per the class contract. */
    public final Arena arena;

    private final ReentrantLock lock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean();
    private Cleaner.Cleanable owned; // non-null iff this state owns its arena

    protected BaseState(Arena arena) {
        if (arena == null) throw new IllegalArgumentException("null arena");
        this.arena = arena;
    }

    /**
     * Marks this state as owning {@link #arena} - called only by the adopting {@code newState}
     * flavors. A non-closeable arena (ofAuto/global) may be adopted: owning it just means there is
     * nothing to free eagerly, and {@link #close()} stays a valid no-op on the memory.
     */
    final void adoptArena() {
        // at-most-once: a second registration would mean two Cleanables racing to close one arena.
        // Package-private and called from exactly one place today, so this is a tripwire for a
        // future caller, not a live bug
        if (owned != null) throw new IllegalStateException("this state already owns its arena");
        // the cleanup action must not capture the state itself - only the arena, the closed
        // flag (its own object), and the optional LeakWatch site
        Arena a = arena;
        AtomicBoolean closedFlag = closed;
        Throwable site = LeakWatch.site("owned state arena");
        owned =
                CLEANER.register(
                        this,
                        () -> {
                            if (site != null && !closedFlag.get()) LeakWatch.report(site);
                            try {
                                a.close();
                            } catch (UnsupportedOperationException ignored) {
                                // a non-closeable arena (ofAuto/global) already manages itself;
                                // owning it just means there is nothing to free eagerly
                            }
                        });
    }

    /** What {@link #enter()} says when a borrowed arena was closed under this state. */
    static final String FREED_MESSAGE =
            "the state's buffers have been freed - the arena passed to newState must outlive the"
                    + " state (close your arena LAST). This canary catches the sequential mistake;"
                    + " freeing the arena DURING a computation is a data race and can still crash"
                    + " the VM.";

    /**
     * Claims this state for a computation on the current thread (reentrant - a generation may hold
     * it across many forwards). Fails fast: another thread computing -> {@link
     * ConcurrentModificationException}; closed -> {@link IllegalStateException}; a BORROWED arena
     * the caller already closed -> {@link IllegalStateException} too (the state-side safety canary:
     * kernels read raw addresses, so this entry check is the only liveness on offer, and a
     * concurrent free mid-computation remains a data race).
     */
    public final void enter() {
        if (closed.get()) throw new IllegalStateException("state is closed");
        if (!arena.scope().isAlive()) throw new IllegalStateException(FREED_MESSAGE);
        if (!lock.tryLock()) {
            // the holder is either another computation (contract violation) or the winning
            // closer draining us (shutdown); `closed` tells which
            if (closed.get()) throw new IllegalStateException("state is closed");
            throw new ConcurrentModificationException(
                    "model state is a single serial pipeline (one computation at a time) - for"
                            + " parallel pipelines create separate model instances/states");
        }
        if (closed.get()) {
            // barged the non-fair lock ahead of the draining closer: hand it straight over -
            // this recheck makes "no computation begins once close is called" strict, not just
            // "freed memory is never touched" (that one the lock alone guarantees)
            lock.unlock();
            throw new IllegalStateException("state is closed");
        }
    }

    /** Releases one {@link #enter} claim. */
    public final void exit() {
        lock.unlock();
    }

    /**
     * Idempotent, BLOCKING close: returns only after the in-flight computation (if any) has
     * finished, then frees the arena iff this state owns it. After close every entry fails with
     * {@link IllegalStateException}. Racing closers return immediately (the CAS winner waits);
     * closing from within this state's own computation throws instead of self-freeing.
     */
    @Override
    public final void close() {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("cannot close a state from within its own computation");
        }
        if (!closed.compareAndSet(false, true)) return;
        lock.lock();
        try {
            if (owned != null) owned.clean(); // at-most-once: frees the arena now, not at GC
        } finally {
            lock.unlock();
        }
    }

    /** True once {@link #close} has been called; entries then fail loudly. */
    public final boolean isClosed() {
        return closed.get();
    }

    public int position; // tokens ingested so far
    public int outputCount; // hidden states the last ingest retained (1 after LAST, n after ALL)
    public int lastChunkLen; // rows of the last ingested batch

    @Override
    public final int position() {
        return position;
    }

    @Override
    public final int outputCount() {
        return outputCount;
    }

    @Override
    public final void advance(int rows, Batch.Outputs outputs) {
        lastChunkLen = rows;
        outputCount = outputs == Batch.Outputs.ALL ? rows : 1;
        position += rows;
    }

    /**
     * MANDATORY for every generative state (unlike the {@link RuntimeState} default): recycling a
     * pooled allocation is only sound when the family has decided which of its buffers carry
     * information across positions - cursor rewind suffices for pure attention (stale KV rows
     * beyond the cursor are masked), recurrent carriers (conv rings, SSM state) must be zeroed.
     * Abstract so the decision is made at compile time, never defaulted into silent corruption or
     * silently-lost recycling.
     */
    @Override
    public abstract void reset();

    @Override
    public final void resumeAt(int p) {
        position = p;
        lastChunkLen = 0;
        outputCount = 0;
    }
}
