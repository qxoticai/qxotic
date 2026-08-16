package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.kernels.Oracles.assertClose;
import static com.qxotic.jinfer.x.kernels.Oracles.f32;
import static com.qxotic.jinfer.x.kernels.Oracles.f32View;
import static com.qxotic.jinfer.x.kernels.Oracles.q8;
import static com.qxotic.jinfer.x.kernels.Oracles.q8View;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Jam-rung parity through {@link MatMul}: this class runs in the {@code jam-parity} surefire
 * execution, where the pure-java backends (jam-vector, jam-scalar; jam-native stays excluded -
 * flaky in forked JVMs) ARE on the classpath, so {@code MatMul.mm}'s prefill rungs - and the
 * k-quant decode rungs via the C2 {@code slowDot} exception - actually fire. The routed result must
 * match {@code MatMul.mmFloor} (the no-jam x floor) within dot tolerance. For dtypes whose decode
 * stays on the floor (dense, Q4_0, MXFP4, NVFP4, Q1_0 - see {@code bytePackedDot}) the gemv arm is
 * floor-vs-floor by construction; the gemm arm is the real jam check.
 */
class MatMulJamParityTest {

    private final Arena arena = Arena.ofAuto();

    /**
     * jam-vs-floor tolerance: jam's kernels accumulate in a different order than the floor dot, so
     * the diff is fp noise, not ulp-level (observed 1.5e-4 abs on a 3.48 lane for Q6_K k=256).
     * jam's own backend parity allows 1e-3..1e-2; 1e-3 abs sits far above that noise and far below
     * any real block-math bug (those are O(1)).
     */
    private static final double JAM_ABS_TOL = 1e-3;

    @BeforeAll
    static void jamBackendPresent() {
        Assumptions.assumeTrue(
                JAM.providers().stream()
                        .anyMatch(p -> p.id().equals("vector") || p.id().equals("scalar")),
                "no pure-java jam backend on the test classpath");
    }

    private MemoryView<MemorySegment> f16View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.FP16, Shape.flat(n));
    }

    private MemoryView<MemorySegment> bf16View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.BF16, Shape.flat(n));
    }

    private MemorySegment f16(Arena arena, long n, long seed) {
        MemorySegment seg = arena.allocate(2L * n, 64);
        Random rng = new Random(seed);
        for (long i = 0; i < n; i++) {
            seg.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    2L * i,
                    Float.floatToFloat16(rng.nextFloat() * 4 - 2));
        }
        return seg;
    }

    private MemorySegment bf16(Arena arena, long n, long seed) {
        MemorySegment seg = arena.allocate(2L * n, 64);
        Random rng = new Random(seed);
        for (long i = 0; i < n; i++) {
            seg.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    2L * i,
                    BFloat16.fromFloat(rng.nextFloat() * 4 - 2));
        }
        return seg;
    }

    /** routed (jam) vs floor for one weight dtype, gemv (n=1) and gemm (n=4) shapes. */
    private void parity(
            String name, MemorySegment wSeg, MemoryView<MemorySegment> w, int m, int k) {
        for (int n : new int[] {1, 4}) {
            MemorySegment a = f32(arena, n * k, 2);
            MemorySegment outFloor = arena.allocate(4L * n * m, 64);
            MemorySegment outRouted = arena.allocate(4L * n * m, 64);
            MemoryView<MemorySegment> av = f32View(a, (long) n * k);
            MatMul.mmFloor(
                    w,
                    0,
                    k,
                    Raw.f32(av, "a"),
                    0,
                    k,
                    Raw.f32(f32View(outFloor, (long) n * m), "c"),
                    0,
                    m,
                    m,
                    n,
                    k,
                    w.dataType(),
                    false);
            MatMul.mm(w, 0, k, av, 0, k, f32View(outRouted, (long) n * m), 0, m, m, n, k);
            assertClose(
                    outFloor,
                    outRouted,
                    n * m,
                    name + " n=" + n + " m=" + m + " k=" + k,
                    JAM_ABS_TOL);
        }
    }

    @Test
    void denseParity() {
        int m = 64, k = 256;
        MemorySegment f32w = f32(arena, m * k, 1);
        parity("F32", f32w, f32View(f32w, (long) m * k), m, k);
        MemorySegment f16w = f16(arena, (long) m * k, 1);
        parity("F16", f16w, f16View(f16w, (long) m * k), m, k);
        MemorySegment bf16w = bf16(arena, (long) m * k, 1);
        parity("BF16", bf16w, bf16View(bf16w, (long) m * k), m, k);
    }

    @Test
    void quantParity() {
        int m = 64, k = 256; // k = QK_K: a multiple of every jam dtype's block
        MemorySegment q8w = q8(arena, m, k, 3);
        parity("Q8_0", q8w, q8View(q8w, (long) m * k), m, k);
        MemorySegment mx = Oracles.mxfp4(arena, m, k, 3);
        parity("MXFP4", mx, Oracles.mxfp4View(mx, (long) m * k), m, k);
        for (GGMLType type :
                new GGMLType[] {
                    GGMLType.Q4_0,
                    GGMLType.Q4_1,
                    GGMLType.Q5_1,
                    GGMLType.Q4_K,
                    GGMLType.Q5_K,
                    GGMLType.Q6_K,
                    GGMLType.NVFP4,
                    GGMLType.Q1_0
                }) {
            long elems = (long) m * k;
            MemorySegment w = Oracles.blockQuant(arena, type, elems, type.ordinal() + 3L);
            parity(type.name(), w, Oracles.blockQuantView(w, type, elems), m, k);
        }
    }
}
