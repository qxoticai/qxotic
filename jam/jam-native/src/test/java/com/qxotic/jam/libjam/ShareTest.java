package com.qxotic.jam.libjam;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jam.JAM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

/**
 * {@link JAM#share} through the native backend: a hint, so shared and plain calls match bit for
 * bit; the share ends with the segment's scope; it never keeps an automatic arena alive. Where no
 * device maps host memory (anything but Apple Silicon on AUTO or metal) every share declines.
 */
class ShareTest {

    private static final boolean DEVICE =
            System.getProperty("os.name").startsWith("Mac")
                    && System.getProperty("os.arch").equals("aarch64")
                    && switch (NativeLoader.config("jam.native.isa", "").toLowerCase()) {
                        case "", "auto", "metal" -> true;
                        default -> false;
                    };

    private static final JAM jam = NativeJAM.create(JAM.Parallel.INLINE);

    private static MemorySegment filled(Arena arena, int count, int seed) {
        MemorySegment s = arena.allocate((long) count * Float.BYTES, 64);
        for (int i = 0; i < count; i++)
            s.setAtIndex(JAVA_FLOAT, i, (float) Math.sin(seed + i * 0.37));
        return s;
    }

    @Test
    void sharedCallsMatchPlainOnes() {
        int m = 64, n = 32, k = 64;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = filled(arena, m * k, 1), a = filled(arena, n * k, 2);
            MemorySegment plain = arena.allocate((long) n * m * Float.BYTES);
            MemorySegment shared = arena.allocate((long) n * m * Float.BYTES);
            assertEquals(JAM.OK, jam.mm(w, a, plain, JAM.F32, m, n, k));
            assertEquals(DEVICE, jam.share(w, true));
            assertEquals(DEVICE, jam.share(a, false));
            assertEquals(DEVICE, jam.share(shared, false));
            assertEquals(JAM.OK, jam.mm(w, a, shared, JAM.F32, m, n, k));
            assertEquals(-1, plain.mismatch(shared));
            // a mutable share: the next call sees rewritten activations
            MemorySegment.copy(filled(arena, n * k, 3), 0, a, 0, a.byteSize());
            assertEquals(JAM.OK, jam.mm(w, a, shared, JAM.F32, m, n, k));
            MemorySegment fresh = arena.allocate(a.byteSize(), 64);
            MemorySegment.copy(a, 0, fresh, 0, a.byteSize());
            assertEquals(JAM.OK, jam.mm(w, fresh, plain, JAM.F32, m, n, k));
            assertEquals(-1, plain.mismatch(shared));
        }
    }

    @Test
    void shareEndsWithItsScope() {
        try (Arena outer = Arena.ofShared()) {
            MemorySegment memory = outer.allocate(1 << 16, 4096);
            try (Arena inner = Arena.ofConfined()) {
                assertEquals(DEVICE, jam.share(memory.reinterpret(inner, null), false));
                assertFalse(jam.share(memory.asSlice(100, 10), false), "overlaps the live share");
            }
            assertEquals(DEVICE, jam.share(memory, true), "released when inner closed");
        }
    }

    @Test
    void shareDoesNotKeepAnAutomaticArenaAlive() throws InterruptedException {
        AtomicBoolean collected = new AtomicBoolean();
        shareAuto(collected);
        for (int i = 0; i < 200 && !collected.get(); i++) {
            System.gc();
            Thread.sleep(50);
        }
        assertTrue(collected.get(), "the automatic arena was never collected");
    }

    private static void shareAuto(AtomicBoolean collected) {
        MemorySegment s = Arena.ofAuto().allocate(1 << 16, 4096);
        assertEquals(DEVICE, jam.share(s, true));
        MemorySegment.ofAddress(s.address())
                .reinterpret(s.byteSize(), view(s.scope()), x -> collected.set(true));
    }

    /** An arena naming another arena's scope, as NativeJAM's own hook does. */
    private static Arena view(MemorySegment.Scope scope) {
        return new Arena() {
            @Override
            public MemorySegment.Scope scope() {
                return scope;
            }

            @Override
            public MemorySegment allocate(long byteSize, long byteAlignment) {
                throw new UnsupportedOperationException();
            }

            @Override
            public void close() {
                throw new UnsupportedOperationException();
            }
        };
    }

    @Test
    void declinesWhatNoDeviceCanUse() {
        assertFalse(jam.share(MemorySegment.ofArray(new float[16]), true), "heap");
        try (Arena arena = Arena.ofConfined()) {
            assertFalse(jam.share(arena.allocate(0), true), "empty");
        }
    }

    /**
     * The binding passes {@code size_t} at its full width: a size with only bit 32 set arrives
     * intact (a CPU context declines it) instead of cut to 0 (which jam would refuse as EINVAL).
     */
    @Test
    void sizesCrossAtFullWidth() {
        if (DEVICE) return; // a device would try to map 4 GB of address space
        try (Arena arena = Arena.ofConfined()) {
            long p = arena.allocate(64).address();
            assertEquals(
                    JAM.EUNSUPPORTED, NativeJAM.jamShare(((NativeJAM) jam).ctx(), p, 1L << 32, 0));
        }
    }

    @Test
    void unknownFlagsAreRefused() {
        try (Arena arena = Arena.ofConfined()) {
            long p = arena.allocate(64).address();
            assertEquals(JAM.EINVAL, NativeJAM.jamShare(((NativeJAM) jam).ctx(), p, 64, 0x2));
            assertFalse(NativeJAM.lastError().isEmpty(), "jam says why");
        }
    }

    @Test
    void misuseThrowsOnEveryPlatform() throws Exception {
        Arena closed = Arena.ofConfined();
        MemorySegment gone = closed.allocate(4096);
        closed.close();
        assertThrows(IllegalStateException.class, () -> jam.share(gone, false));
        try (Arena confined = Arena.ofConfined()) {
            MemorySegment mine = confined.allocate(4096);
            Throwable[] thrown = new Throwable[1];
            Thread other =
                    new Thread(
                            () -> {
                                try {
                                    jam.share(mine, false);
                                } catch (Throwable t) {
                                    thrown[0] = t;
                                }
                            });
            other.start();
            other.join();
            assertInstanceOf(WrongThreadException.class, thrown[0]);
        }
        NativeJAM shut = NativeJAM.create(JAM.Parallel.INLINE);
        shut.close();
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    IllegalStateException.class, () -> shut.share(arena.allocate(4096), false));
        }
    }

    @Test
    void scopeClosingAfterTheBackendIsHarmless() {
        NativeJAM j = NativeJAM.create(JAM.Parallel.INLINE);
        Arena arena = Arena.ofConfined();
        assertEquals(DEVICE, j.share(arena.allocate(1 << 16, 4096), false));
        j.close(); // releases the share with the context
        arena.close(); // the scope's cleanup finds the context gone
    }
}
