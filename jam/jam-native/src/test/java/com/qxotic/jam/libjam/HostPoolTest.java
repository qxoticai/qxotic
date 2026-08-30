package com.qxotic.jam.libjam;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jam.JAM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * The host-executor bridge: a context created on a real pool fans every jam task back onto that
 * pool with the task's slot as its tid, matches the single-threaded result bit for bit, refuses a
 * tid it has no scratch for, and surfaces a task's exception instead of ending the VM.
 */
class HostPoolTest {

    /** A pool with unique slots: the caller is 0, each executor thread claims one on first use. */
    static final class Pool implements JAM.Parallel, AutoCloseable {
        final int width;
        final ExecutorService workers;
        final ThreadLocal<Integer> slotOf = new ThreadLocal<>();
        final AtomicInteger nextSlot = new AtomicInteger(1);
        final Set<Integer> slotsSeen = ConcurrentHashMap.newKeySet();

        Pool(int width) {
            this.width = width;
            this.workers = Executors.newFixedThreadPool(width - 1);
        }

        @Override
        public int width() {
            return width;
        }

        @Override
        public void run(int count, Job body) {
            int groups = Math.min(count, width);
            List<Future<?>> futures = new ArrayList<>();
            for (int g = 1; g < groups; g++) {
                int lo = (int) ((long) count * g / groups),
                        hi = (int) ((long) count * (g + 1) / groups);
                futures.add(
                        workers.submit(
                                () -> {
                                    Integer slot = slotOf.get();
                                    if (slot == null) slotOf.set(slot = nextSlot.getAndIncrement());
                                    slotsSeen.add(slot);
                                    for (int i = lo; i < hi; i++) body.run(i, slot);
                                }));
            }
            int hi0 = (int) ((long) count / groups);
            slotsSeen.add(0);
            for (int i = 0; i < hi0; i++) body.run(i, 0);
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException(e.getCause());
                }
            }
        }

        @Override
        public void close() {
            workers.shutdownNow();
        }
    }

    private static float[] run(JAM jam, Arena ar, float[] W, float[] A, int m, int n, int k) {
        MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
        MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
        MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
        MemorySegment.copy(W, 0, w, JAVA_FLOAT, 0, m * k);
        MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
        assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F32, m, n, k));
        float[] out = new float[m * n];
        MemorySegment.copy(c, JAVA_FLOAT, 0, out, 0, m * n);
        return out;
    }

    private static float[] random(int n, long seed) {
        Random r = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = r.nextFloat() - 0.5f;
        return v;
    }

    @Test
    void aPooledContextMatchesTheInlineOneAndUsesEverySlot() {
        int m = 256, n = 64, k = 512;
        float[] W = random(m * k, 1), A = random(n * k, 2);
        try (Arena ar = Arena.ofConfined();
                Pool pool = new Pool(4)) {
            JAM inline = NativeJAM.create(JAM.Parallel.INLINE);
            JAM pooled = NativeJAM.create(pool);
            float[] expected = run(inline, ar, W, A, m, n, k);
            for (int round = 0; round < 5; round++) {
                assertArrayEquals(expected, run(pooled, ar, W, A, m, n, k));
            }
            assertEquals(Set.of(0, 1, 2, 3), pool.slotsSeen, "every slot took part");
        }
    }

    @Test
    void decodeAndPrefillShapesAgreeAcrossPools() {
        int k = 1024;
        float[] W = random(2048 * k, 3);
        try (Arena ar = Arena.ofConfined();
                Pool two = new Pool(2);
                Pool eight = new Pool(8)) {
            JAM inline = NativeJAM.create(JAM.Parallel.INLINE);
            JAM p2 = NativeJAM.create(two);
            JAM p8 = NativeJAM.create(eight);
            for (int n : new int[] {1, 2, 7, 8, 33}) {
                float[] A = random(n * k, 100 + n);
                float[] expected = run(inline, ar, W, A, 2048, n, k);
                assertArrayEquals(expected, run(p2, ar, W, A, 2048, n, k), "n=" + n + " width 2");
                assertArrayEquals(expected, run(p8, ar, W, A, 2048, n, k), "n=" + n + " width 8");
            }
        }
    }

    @Test
    void aTidBeyondTheDeclaredWidthIsRefusedNotIndexed() {
        JAM.Parallel lying =
                new JAM.Parallel() {
                    @Override
                    public void run(int count, Job body) {
                        for (int i = 0; i < count; i++) body.run(i, 7); // width says 2
                    }

                    @Override
                    public int width() {
                        return 2;
                    }
                };
        try (Arena ar = Arena.ofConfined()) {
            JAM jam = NativeJAM.create(lying);
            int m = 64, n = 4, k = 64;
            MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
            MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
            MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
            assertEquals(JAM.EINVAL, jam.mm(w, a, c, JAM.F32, m, n, k));
        }
    }

    @Test
    void aTaskExceptionSurfacesFromMm() {
        RuntimeException boom = new IllegalStateException("task");
        JAM.Parallel throwing =
                new JAM.Parallel() {
                    @Override
                    public void run(int count, Job body) {
                        body.run(0, 0);
                        throw boom;
                    }

                    @Override
                    public int width() {
                        return 2;
                    }
                };
        try (Arena ar = Arena.ofConfined()) {
            JAM jam = NativeJAM.create(throwing);
            int m = 64, n = 4, k = 64;
            MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
            MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
            MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
            IllegalStateException thrown =
                    assertThrows(
                            IllegalStateException.class, () -> jam.mm(w, a, c, JAM.F32, m, n, k));
            assertSame(boom, thrown);
            // the context is usable again on a sane pool
            JAM sane = NativeJAM.create(JAM.Parallel.INLINE);
            assertEquals(JAM.OK, sane.mm(w, a, c, JAM.F32, m, n, k));
        }
    }

    @Test
    void instancesAreIndependent() {
        try (Arena ar = Arena.ofConfined();
                Pool a = new Pool(3);
                Pool b = new Pool(5)) {
            JAM ja = NativeJAM.create(a), jb = NativeJAM.create(b);
            int m = 128, n = 16, k = 256;
            float[] W = random(m * k, 9), A = random(n * k, 10);
            float[] ra = run(ja, ar, W, A, m, n, k), rb = run(jb, ar, W, A, m, n, k);
            assertArrayEquals(ra, rb);
            assertTrue(a.slotsSeen.stream().allMatch(s -> s < 3));
            assertTrue(b.slotsSeen.stream().allMatch(s -> s < 5));
        }
    }

    @Test
    void slicesAreBoundedAndCoverEveryRow() {
        List<int[]> tasks = new ArrayList<>();
        JAM.Parallel recording =
                new JAM.Parallel() {
                    @Override
                    public void run(int count, Job body) {
                        for (int i = 0; i < count; i++) {
                            tasks.add(new int[] {count, i});
                            body.run(i, i % 4);
                        }
                    }

                    @Override
                    public int width() {
                        return 4;
                    }
                };
        try (Arena ar = Arena.ofConfined();
                NativeJAM jam = NativeJAM.create(recording)) {
            int m = 4096, n = 8, k = 256;
            float[] W = random(m * k, 5), A = random(n * k, 6);
            float[] expected = run(NativeJAM.create(JAM.Parallel.INLINE), ar, W, A, m, n, k);
            assertArrayEquals(expected, run(jam, ar, W, A, m, n, k));
            assertTrue(!tasks.isEmpty(), "the fan-out reached the pool");
            for (int[] t : tasks)
                assertTrue(t[0] <= 16, "a fan-out had " + t[0] + " slices for width 4");
        }
    }

    @Test
    void closeFreesTheContextAndLaterCallsThrow() {
        try (Arena ar = Arena.ofConfined()) {
            NativeJAM jam = NativeJAM.create(JAM.Parallel.INLINE);
            int m = 16, n = 2, k = 32;
            MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
            MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
            MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
            assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F32, m, n, k));
            jam.close();
            jam.close();
            assertThrows(IllegalStateException.class, () -> jam.mm(w, a, c, JAM.F32, m, n, k));
            assertThrows(IllegalStateException.class, () -> jam.packSize(JAM.Q4_K, 256, 256));
            // the handle is reused, and the new instance is the one the upcall finds
            try (Pool pool = new Pool(3);
                    NativeJAM again = NativeJAM.create(pool)) {
                assertEquals(JAM.OK, again.mm(w, a, c, JAM.F32, m, n, k));
                assertTrue(pool.slotsSeen.contains(0));
            }
        }
    }

    @Test
    void manyCreateCloseCyclesLeaveNothingBehind() {
        try (Arena ar = Arena.ofConfined()) {
            int m = 64, n = 4, k = 64;
            MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
            MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
            MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
            for (int r = 0; r < 200; r++) {
                try (Pool pool = new Pool(2 + (r % 3));
                        NativeJAM jam = NativeJAM.create(pool)) {
                    assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F32, m, n, k));
                }
            }
        }
    }

    @Test
    void concurrentCallersOnOneInstanceSerialize() throws Exception {
        try (Arena ar = Arena.ofShared();
                Pool pool = new Pool(4);
                NativeJAM jam = NativeJAM.create(pool)) {
            int m = 512, n = 16, k = 256;
            float[] W = random(m * k, 7), A = random(n * k, 8);
            float[] expected = run(NativeJAM.create(JAM.Parallel.INLINE), ar, W, A, m, n, k);
            Thread[] threads = new Thread[6];
            java.util.concurrent.atomic.AtomicReference<Throwable> failure =
                    new java.util.concurrent.atomic.AtomicReference<>();
            for (int t = 0; t < threads.length; t++) {
                threads[t] =
                        new Thread(
                                () -> {
                                    try {
                                        for (int r = 0; r < 20; r++)
                                            assertArrayEquals(
                                                    expected, run(jam, ar, W, A, m, n, k));
                                    } catch (Throwable e) {
                                        failure.compareAndSet(null, e);
                                    }
                                });
                threads[t].start();
            }
            for (Thread t : threads) t.join(60_000);
            if (failure.get() != null) throw new AssertionError(failure.get());
        }
    }
}
