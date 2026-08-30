package com.qxotic.jinfer;

import com.qxotic.jinfer.telemetry.PerformanceCliff;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/**
 * A pool of {@link #width()} participants, the calling thread being one of them. Jinfer runs on
 * {@link #shared()}, sized by {@link RuntimeFlags#THREADS}; the static {@link #forLoop} methods are
 * that instance's. A jam backend, a test or a second host can own its own pool from {@link #of};
 * its workers are daemon threads that live until {@link #close()}, so an own pool is closed like
 * any resource.
 *
 * <p>A loop is a region: participants claim contiguous chunks of it from a counter (dynamic, so a
 * slow chunk never holds the region), the caller works alongside the workers and returns when the
 * last chunk is done. A {@link Body} also learns its participant's {@code slot} in {@code [0,
 * width)}: the caller is slot 0, and within one loop no two iterations running at once share a
 * slot, so a slot indexes per-participant scratch owned by the loop's caller. A body that throws
 * ends the loop early: what has not started is skipped, and the first failure is rethrown to the
 * caller once every participant has stopped; that is the only way to end a loop early - an
 * interrupt is preserved for the caller to see afterwards, not acted on. Between regions the
 * workers spin {@link #SPIN_NANOS} - the many small regions of one decode token or one prefill
 * matmul are microseconds apart - and park when the pool is idle. Regions of one pool are
 * serialized: a second thread submitting waits for the running region, which is the fair share on a
 * machine the first submitter already fills. A loop submitted from inside a region of the same pool
 * runs inline on that thread; loops are independent and non-blocking, so that is always correct,
 * only less parallel. A loop on another pool from inside a region is a region there; two pools
 * whose regions submit to each other at the same time deadlock, as two locks taken in both orders
 * do.
 */
public final class Parallel implements AutoCloseable {

    /** Spin this long for the next region before parking. */
    private static final long SPIN_NANOS = 100_000L;

    /**
     * {@link #loop} claims chunks of {@code size / (width * CHUNKS_PER_PARTICIPANT)}: two per
     * participant, so each streams a long contiguous half-band and the second claims absorb a late
     * starter. Measured on a 2.6B Q4_K_M decode (16 threads, a memory-bound gemv over 2048 rows):
     * chunks of 1/64 of the region cost 3% against halves, one whole band per participant was
     * straggler-fragile, and guided (shrinking) claims sat in between with more variance. Coarse
     * items that must balance one by one (a gemm panel, a slice of a fan-out) use {@link #forEach}.
     */
    private static final int CHUNKS_PER_PARTICIPANT = 2;

    private static final class Shared {
        static final Parallel POOL = new Parallel(RuntimeFlags.THREADS, "jinfer");
    }

    private static final AtomicInteger IDS = new AtomicInteger();

    private final int width;
    private final String name;
    private final ReentrantLock lock = new ReentrantLock(true);

    /** The current region; workers act on every new object they see here. */
    private volatile Region current;

    private volatile Worker[] workers;
    private volatile boolean closed;

    private Parallel(int width, String name) {
        if (width < 1) throw new IllegalArgumentException("width < 1: " + width);
        this.width = width;
        this.name = name;
        this.current = new Region(0, 0, null, null, null, false);
    }

    /** A pool of {@code width} participants; its workers start on the first region. */
    public static Parallel of(int width) {
        return new Parallel(width, "jinfer-p" + IDS.incrementAndGet());
    }

    /** A loop body that also receives the slot of the participant running it. */
    @FunctionalInterface
    public interface Body {
        void run(int index, int slot);
    }

    /** The process-wide pool, {@link RuntimeFlags#THREADS} wide. */
    public static Parallel shared() {
        return Shared.POOL;
    }

    /** {@link #shared()}'s width: the compute thread budget. */
    public static int threads() {
        return shared().width();
    }

    /** {@link #shared()}'s {@link #loop(int, IntConsumer)}. */
    public static void forLoop(int count, IntConsumer body) {
        shared().loop(0, count, body);
    }

    /** {@link #shared()}'s {@link #loop(int, int, IntConsumer)}. */
    public static void forLoop(int start, int end, IntConsumer body) {
        shared().loop(start, end, body);
    }

    /** {@link #shared()}'s {@link #loop(int, Body)}. */
    public static void forLoop(int count, Body body) {
        shared().loop(count, body);
    }

    /** {@link #shared()}'s {@link #loop(int, int, Body)}. */
    public static void forLoop(int start, int end, Body body) {
        shared().loop(start, end, body);
    }

    /** {@link #shared()}'s {@link #each(int, Body)}. */
    public static void forEach(int count, Body body) {
        shared().each(count, body);
    }

    /** How many participants a region has, the caller included. */
    public int width() {
        return width;
    }

    /** Whether the calling thread is inside one of {@link #shared()}'s regions. */
    public static boolean inRegion() {
        return shared().inside();
    }

    /**
     * Whether the calling thread is inside one of this pool's regions (a worker, or the submitter).
     */
    public boolean inside() {
        Thread me = Thread.currentThread();
        if (me instanceof Worker w) return w.pool == this;
        Region running = current;
        return running.submitter == me && !running.done();
    }

    /** {@code body.accept(i)} for every {@code i < count}. */
    public void loop(int count, IntConsumer body) {
        loop(0, count, body);
    }

    /**
     * {@code body.accept(i)} for every {@code i} in {@code [start, end)}. Iterations may run
     * concurrently and in any order; they must be independent and non-blocking.
     */
    public void loop(int start, int end, IntConsumer body) {
        Objects.requireNonNull(body, "body");
        loop(start, end, body, null, false);
    }

    /** {@code body.run(i, slot)} for every {@code i < count}. */
    public void loop(int count, Body body) {
        loop(0, count, body);
    }

    /** As {@link #loop(int, int, IntConsumer)}, with the participant's slot. */
    public void loop(int start, int end, Body body) {
        Objects.requireNonNull(body, "body");
        loop(start, end, null, body, false);
    }

    /**
     * {@link #loop(int, Body)} claiming one index at a time: for coarse items (a gemm panel, a
     * slice of a native fan-out) where the pool must balance every item, not bands of them.
     */
    public void each(int count, Body body) {
        Objects.requireNonNull(body, "body");
        loop(0, count, null, body, true);
    }

    /** One of {@code simple} and {@code body} is set; the region calls it without a wrapper. */
    private void loop(int start, int end, IntConsumer simple, Body body, boolean each) {
        if (start >= end) return;
        Thread me = Thread.currentThread();
        Region running = current;
        boolean nested =
                (me instanceof Worker w && w.pool == this)
                        || (running.submitter == me && !running.done());
        if (end - start == 1 || width == 1 || nested) {
            if (nested && end - start > 1 && width > 1) PerformanceCliff.NESTED_REGION.report();
            int slot = me instanceof Worker w && w.pool == this ? w.slot : 0;
            if (simple != null) for (long i = start; i < end; i++) simple.accept((int) i);
            else for (long i = start; i < end; i++) body.run((int) i, slot);
            return;
        }
        Worker[] pool = pool();
        lock.lock();
        try {
            Region region = new Region(start, end, simple, body, me, each); // one allocation
            current = region;
            for (Worker w : pool) if (w.parked) LockSupport.unpark(w);
            region.work(0);
            while (!region.done()) Thread.onSpinWait();
            region.simple = null; // nobody reads them after done(): drop what the lambdas captured
            region.body = null;
            Throwable failure = region.failure;
            if (failure != null) throw unchecked(failure);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public String toString() {
        Worker[] pool = workers;
        return name
                + "[width="
                + width
                + ", workers="
                + (pool == null ? 0 : pool.length)
                + (closed ? ", closed" : "")
                + "]";
    }

    /** Stops the workers; a closed pool runs every later loop inline. */
    @Override
    public void close() {
        synchronized (
                this) { // with pool(): a worker started before `workers` is set is unparked too
            closed = true;
            Worker[] pool = workers;
            if (pool != null) for (Worker w : pool) LockSupport.unpark(w);
        }
    }

    private Worker[] pool() {
        Worker[] p = workers;
        if (p == null) {
            synchronized (this) {
                p = workers;
                if (p == null) {
                    p = new Worker[closed ? 0 : width - 1];
                    for (int i = 0; i < p.length; i++) {
                        p[i] = new Worker(this, name + "-" + i, i);
                        p[i].start();
                    }
                    workers = p;
                }
            }
        }
        return p;
    }

    /** One loop: chunks are claimed from {@code next}; {@code finished} counts done indices. */
    private final class Region {
        private static final VarHandle NEXT, FINISHED, FAILURE;

        static {
            try {
                MethodHandles.Lookup l = MethodHandles.lookup();
                NEXT = l.findVarHandle(Region.class, "next", long.class);
                FINISHED = l.findVarHandle(Region.class, "finished", long.class);
                FAILURE = l.findVarHandle(Region.class, "failure", Throwable.class);
            } catch (ReflectiveOperationException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        final int start;
        final long size; // offsets from start: int arithmetic would wrap near MAX_VALUE
        final long chunk; // indices per claim
        final Thread submitter;
        volatile IntConsumer simple; // one of the two is the loop body
        volatile Body body;
        volatile long next;
        volatile long finished;
        volatile Throwable failure;

        Region(int start, int end, IntConsumer simple, Body body, Thread submitter, boolean each) {
            this.start = start;
            this.size = Math.max(0, (long) end - start);
            this.simple = simple;
            this.body = body;
            this.submitter = submitter;
            this.chunk = each ? 1 : Math.max(1, size / ((long) width * CHUNKS_PER_PARTICIPANT));
        }

        boolean done() {
            return finished >= size;
        }

        void work(int slot) {
            IntConsumer simple = this.simple;
            Body body = this.body;
            for (long lo; (lo = (long) NEXT.getAndAdd(this, chunk)) < size; ) {
                long hi = Math.min(size, lo + chunk);
                try {
                    for (long off = lo; off < hi && failure == null; off++) {
                        try {
                            if (simple != null) simple.accept((int) (start + off));
                            else body.run((int) (start + off), slot);
                        } catch (Throwable thrown) {
                            FAILURE.compareAndSet(
                                    this, null, thrown); // fail fast: the rest is skipped
                        }
                    }
                } finally {
                    FINISHED.getAndAdd(this, hi - lo); // whatever escaped above, the region can end
                }
            }
        }
    }

    private static final class Worker extends Thread {
        final Parallel pool;
        final int slot; // the caller is slot 0
        volatile boolean parked;

        Worker(Parallel pool, String name, int index) {
            super(name);
            this.pool = pool;
            this.slot = index + 1;
            setDaemon(true);
        }

        @Override
        public void run() {
            Region seen = null; // the region published while this worker started is new to it
            long idleSince = System.nanoTime();
            while (!pool.closed) {
                Region region = pool.current;
                if (region != seen) {
                    seen = region;
                    try {
                        region.work(slot);
                    } catch (Throwable escaped) { // not from a body: those are caught in work
                        Region.FAILURE.compareAndSet(region, null, escaped);
                    }
                    idleSince = System.nanoTime();
                } else if (System.nanoTime() - idleSince < SPIN_NANOS) {
                    Thread.onSpinWait();
                } else {
                    parked = true;
                    Thread.interrupted(); // a stray interrupt would make park return at once
                    if (pool.current == seen && !pool.closed) LockSupport.park(this);
                    parked = false;
                    idleSince = System.nanoTime();
                }
            }
        }
    }

    static RuntimeException unchecked(Throwable failure) {
        if (failure instanceof RuntimeException runtime) return runtime;
        if (failure instanceof Error error) throw error;
        return new RuntimeException(failure);
    }
}
