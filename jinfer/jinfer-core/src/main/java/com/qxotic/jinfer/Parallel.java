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
 * <p>The pool's one primitive is {@link #run}: {@code jobs} jobs, claimed one at a time from a
 * counter, the caller working alongside the workers and returning when the last is done. A job is
 * whatever the caller wants balanced - a gemm panel, a slice of a native fan-out, a head. For many
 * cheap uniform iterations, {@link #loop} (and the static {@link #forLoop}) is sugar that presents
 * the range as {@code 2 x width} contiguous bands, one job each: a participant then streams one
 * long band (a memory-bound gemv measured 3% faster than 64 pieces, 6% faster than 256) and the
 * second claims absorb a late starter (one band per participant measured straggler-fragile).
 * Nothing else in the pool decides granularity. A {@link Body} also learns its participant's {@code
 * slot} in {@code [0, width)}: the caller is slot 0, and within one loop no two iterations running
 * at once share a slot, so a slot indexes per-participant scratch owned by the loop's caller. A
 * body that throws ends the loop early: what has not started is skipped, and the first failure is
 * rethrown to the caller once every participant has stopped; that is the only way to end a loop
 * early - an interrupt is preserved for the caller to see afterwards, not acted on. Between regions
 * the workers spin {@link #SPIN_NANOS} - the many small regions of one decode token or one prefill
 * matmul are microseconds apart - and park when the pool is idle. Regions of one pool are
 * serialized: a second thread submitting waits for the running region, which is the fair share on a
 * machine the first submitter already fills. A loop submitted from inside a region of the same pool
 * runs inline on that thread; loops are independent and non-blocking, so that is always correct,
 * only less parallel. A loop on another pool from inside a region is a region there; two pools
 * whose regions submit to each other at the same time deadlock, as two locks taken in both orders
 * do.
 */
public final class Parallel implements AutoCloseable {

    /**
     * Spin this long for the next region before parking. A prefill's regions are separated by the
     * model's serial glue; at 100 us the workers parked in those gaps and paid the wake-up on every
     * region (16T pp512: 964 t/s at 100 us, 1025 at 1 ms, 1033 at 5 ms). {@code -Djinfer.spinNanos}
     * overrides for machines where spinning through the gaps costs more than the wake-ups - a
     * fan-limited laptop turns the spin into heat the long MoE prefill pays for.
     */
    private static final long SPIN_NANOS = Long.getLong("jinfer.spinNanos", 1_000_000L);

    /** Bands per participant for {@link #loop}: the one scheduling constant, see the class note. */
    private static final int BANDS_PER_PARTICIPANT = 2;

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
        this.current = new Region(0, 0, 0, null, null, null);
    }

    /** A pool of {@code width} participants; its workers start on the first region. */
    public static Parallel of(int width) {
        return new Parallel(width, "jinfer-p" + IDS.incrementAndGet());
    }

    /**
     * A job (or, under {@link #loop}, one iteration) and the slot of the participant running it.
     */
    @FunctionalInterface
    public interface Job {
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

    /** {@link #shared()}'s {@link #loop(int, Job)}. */
    public static void forLoop(int count, Job body) {
        shared().loop(count, body);
    }

    /** {@link #shared()}'s {@link #loop(int, int, Job)}. */
    public static void forLoop(int start, int end, Job body) {
        shared().loop(start, end, body);
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

    /**
     * The primitive: {@code body.run(j, slot)} for every job {@code j < jobs}, each claimed by one
     * participant; returns when all are done. Jobs may run concurrently and in any order; they must
     * be independent and non-blocking.
     */
    public void run(int jobs, Job body) {
        Objects.requireNonNull(body, "body");
        region(0, jobs, null, body, true);
    }

    /** Sugar over {@link #run}: {@code body.accept(i)} for every {@code i < count}, in bands. */
    public void loop(int count, IntConsumer body) {
        loop(0, count, body);
    }

    /**
     * Sugar over {@link #run}: {@code body.accept(i)} for every {@code i} in {@code [start, end)},
     * in bands.
     */
    public void loop(int start, int end, IntConsumer body) {
        Objects.requireNonNull(body, "body");
        region(start, end, body, null, false);
    }

    /** Sugar over {@link #run}: {@code body.run(i, slot)} for every {@code i < count}, in bands. */
    public void loop(int count, Job body) {
        loop(0, count, body);
    }

    /**
     * Sugar over {@link #run}: {@code body.run(i, slot)} for every {@code i} in {@code [start,
     * end)}, in bands.
     */
    public void loop(int start, int end, Job body) {
        Objects.requireNonNull(body, "body");
        region(start, end, null, body, false);
    }

    /**
     * One of {@code simple} and {@code body} is set; the region calls it without a wrapper. With
     * {@code jobsAreIndices} every index is its own job, else the range is cut into bands.
     */
    private void region(int start, int end, IntConsumer simple, Job body, boolean jobsAreIndices) {
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
            long size = (long) end - start;
            int jobs =
                    (int)
                            Math.min(
                                    size,
                                    jobsAreIndices ? size : (long) width * BANDS_PER_PARTICIPANT);
            Region region = new Region(start, size, jobs, simple, body, me); // one allocation
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

    /** One loop: jobs are claimed from {@code next}; {@code finished} counts done jobs. */
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
        final int jobs; // job j covers [size * j / jobs, size * (j + 1) / jobs)
        final Thread submitter;
        volatile IntConsumer simple; // one of the two is the loop body
        volatile Job body;
        volatile long next;
        volatile long finished;
        volatile Throwable failure;

        Region(int start, long size, int jobs, IntConsumer simple, Job body, Thread submitter) {
            this.start = start;
            this.size = size;
            this.jobs = jobs;
            this.simple = simple;
            this.body = body;
            this.submitter = submitter;
        }

        boolean done() {
            return finished >= jobs;
        }

        void work(int slot) {
            IntConsumer simple = this.simple;
            Job body = this.body;
            for (long j; (j = (long) NEXT.getAndAdd(this, 1L)) < jobs; ) {
                long lo = size * j / jobs, hi = size * (j + 1) / jobs;
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
                    FINISHED.getAndAdd(this, 1L); // whatever escaped above, the region can end
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
