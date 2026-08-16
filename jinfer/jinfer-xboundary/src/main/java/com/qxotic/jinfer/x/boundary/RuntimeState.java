package com.qxotic.jinfer.x.boundary;

import java.util.ConcurrentModificationException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/** Mutable, caller-owned resources used by one inference pipeline. */
public abstract class RuntimeState implements AutoCloseable {

    private final ReentrantLock lock = new ReentrantLock();
    private final AtomicBoolean closing = new AtomicBoolean();
    private boolean released;
    private Throwable closeFailure;

    /** Runs {@code operation} with exclusive access to this state. Same-thread calls may nest. */
    public final void exclusively(Runnable operation) {
        Objects.requireNonNull(operation, "operation");
        enter();
        try {
            operation.run();
        } finally {
            lock.unlock();
        }
    }

    /** Runs {@code operation} with exclusive access to this state. Same-thread calls may nest. */
    public final <T> T exclusively(Supplier<? extends T> operation) {
        Objects.requireNonNull(operation, "operation");
        enter();
        try {
            return operation.get();
        } finally {
            lock.unlock();
        }
    }

    private void enter() {
        boolean nested = lock.isHeldByCurrentThread();
        if (!nested && closing.get()) throw new IllegalStateException("state is closed");
        if (!lock.tryLock()) {
            if (closing.get()) throw new IllegalStateException("state is closed");
            throw new ConcurrentModificationException(
                    "model state is a single serial pipeline (one computation at a time) - for"
                            + " parallel pipelines create separate states");
        }
        try {
            // A running operation may finish through nested calls while another thread closes.
            if (!nested && closing.get()) throw new IllegalStateException("state is closed");
            if (!nested) checkResourcesAlive();
        } catch (RuntimeException | Error failure) {
            lock.unlock();
            throw failure;
        }
    }

    /** Optional fail-fast canary for borrowed resources. */
    protected void checkResourcesAlive() {}

    /** Releases owned resources. Called exactly once, with exclusive access already held. */
    protected abstract void releaseResources();

    /** Whether this state still accepts operations. */
    public final boolean isAlive() {
        return !closing.get();
    }

    /**
     * Closes this state after any active operation finishes. All concurrent callers wait for the
     * same cleanup and observe the same failure, if cleanup fails.
     */
    @Override
    public final void close() {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("cannot close a state from within its own operation");
        }
        closing.set(true);
        Throwable failure;
        lock.lock();
        try {
            if (!released) {
                try {
                    releaseResources();
                } catch (RuntimeException | Error e) {
                    closeFailure = e;
                } finally {
                    released = true;
                }
            }
            failure = closeFailure;
        } finally {
            lock.unlock();
        }
        if (failure instanceof RuntimeException e) throw e;
        if (failure instanceof Error e) throw e;
    }
}
