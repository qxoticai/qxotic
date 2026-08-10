package com.qxotic.jinfer.bench;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Random;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Phase-0 gate for the FloatTensor -> MemoryView migration (see
 * qxotic/.opencode/plans/memoryview-migration.md). Compares, on identical buffers:
 *
 * <ol>
 *   <li>the current virtual dispatch: {@code Q8_0FloatTensor.dot} via {@link FloatTensor#create}
 *   <li>the migration target: a static, byte-addressed {@code dotQ8_0(MemorySegment, long,
 *       MemorySegment, long, int)} whose body is the same vector code
 *   <li>the same static dot driven through the jota extraction contract: {@code
 *       view.memory().base() + view.byteOffset()} (also the P0.3 correctness check)
 * </ol>
 *
 * Run: {@code java --add-modules jdk.incubator.vector com.qxotic.jinfer.bench.Q8DotBench [secs]}
 */
public final class Q8DotBench {

    private static final VectorSpecies<Float> FS = FloatVector.SPECIES_PREFERRED;
    private static final int BLOCK = GGMLType.Q8_0.getElementsPerBlock(); // 32
    private static final int BLOCK_BYTES = GGMLType.Q8_0.getBlockByteSize(); // 34

    private static volatile float sink;

    public static void main(String[] args) {
        int secs = args.length > 0 ? Integer.parseInt(args[0]) : 3;
        System.out.printf("species=%s (%d-bit)%n", FS, FS.vectorBitSize());

        int[][] shapes = {{2048, 2048}, {4096, 4096}, {8192, 2048}}; // (m rows, k cols)
        boolean ok = true;
        for (int[] mk : shapes) {
            ok &= run(mk[0], mk[1], secs);
        }
        if (!ok) {
            throw new AssertionError("differential check failed");
        }
    }

    private static boolean run(int m, int k, int secs) {
        System.out.printf(
                "%n=== gemv m=%d k=%d (weights %.1f MB)%n",
                m, k, (double) m * k / BLOCK * BLOCK_BYTES / 1e6);

        Arena arena = Arena.ofAuto();
        Random rng = new Random(42);
        // weights: m*k Q8_0 (scale ~2^-6 keeps products finite; q random)
        MemorySegment wseg = arena.allocate((long) m * k / BLOCK * BLOCK_BYTES, 64);
        for (long b = 0; b < wseg.byteSize() / BLOCK_BYTES; b++) {
            wseg.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    b * BLOCK_BYTES,
                    Float.floatToFloat16(0.015625f));
            for (int i = 0; i < BLOCK; i++) {
                wseg.set(ValueLayout.JAVA_BYTE, b * BLOCK_BYTES + 2 + i, (byte) rng.nextInt(256));
            }
        }
        // activation: k F32
        MemorySegment xseg = arena.allocate(4L * k, 64);
        for (int i = 0; i < k; i++) {
            xseg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i, rng.nextFloat() * 2 - 1);
        }

        // old world: virtual tensors
        FloatTensor wOld = FloatTensor.create(GGMLType.Q8_0, (long) m * k, wseg);
        F32FloatTensor xOld = (F32FloatTensor) FloatTensor.create(GGMLType.F32, k, xseg);

        // new world: jota view over the same segment (row 0; rows addressed by byte math).
        // Block-dtype convention: shape counts BLOCKS (one element-step = one 34-byte block).
        Memory<MemorySegment> wmem = MemoryFactory.ofMemorySegment(wseg);
        MemoryView<MemorySegment> wView =
                MemoryView.of(wmem, 0, DataType.Q8_0, Layout.rowMajor(Shape.of((long) k / BLOCK)));

        // --- differential check (P0.3 + kernel parity) ---
        boolean ok = true;
        for (int row : new int[] {0, 1, m / 2, m - 1}) {
            float a = wOld.dot((long) row * k, xOld, 0, k);
            float b = dotQ8_0(wseg, (long) row * k / BLOCK * BLOCK_BYTES, xseg, 0, k);
            MemoryView<MemorySegment> rowView =
                    MemoryView.of(
                            wmem,
                            (long) row * k / BLOCK * BLOCK_BYTES,
                            DataType.Q8_0,
                            Layout.rowMajor(Shape.of((long) k / BLOCK)));
            float c = dotQ8_0(rowView.memory().base(), rowView.byteOffset(), xseg, 0, k);
            if (a != b || a != c) {
                System.out.printf("  MISMATCH row=%d old=%s static=%s viaView=%s%n", row, a, b, c);
                ok = false;
            }
        }
        if (!ok) {
            return false;
        }
        System.out.println("  differential: old == static == via-view (bit-equal)");

        // --- throughput ---
        long rowBytes = (long) k / BLOCK * BLOCK_BYTES;
        double gbPerIter = (double) m * rowBytes / 1e9;
        bench(
                "old virtual w.dot ",
                secs,
                () -> {
                    float acc = 0;
                    for (int row = 0; row < m; row++) {
                        acc += wOld.dot((long) row * k, xOld, 0, k);
                    }
                    sink = acc;
                },
                gbPerIter);
        bench(
                "static dotQ8_0      ",
                secs,
                () -> {
                    float acc = 0;
                    for (int row = 0; row < m; row++) {
                        acc += dotQ8_0(wseg, (long) row * rowBytes, xseg, 0, k);
                    }
                    sink = acc;
                },
                gbPerIter);
        bench(
                "static via MemoryView",
                secs,
                () -> {
                    // extraction recipe: segment + base once per mm call, byte math per row
                    MemorySegment ws = wView.memory().base();
                    long wb = wView.byteOffset();
                    float acc = 0;
                    for (int row = 0; row < m; row++) {
                        acc += dotQ8_0(ws, wb + (long) row * rowBytes, xseg, 0, k);
                    }
                    sink = acc;
                },
                gbPerIter);
        return true;
    }

    private static void bench(String name, int secs, Runnable body, double gbPerIter) {
        long deadline = System.nanoTime() + 2_000_000_000L; // warmup 2s
        while (System.nanoTime() < deadline) {
            body.run();
        }
        long iters = 0, nanos;
        deadline = System.nanoTime() + secs * 1_000_000_000L;
        long start = System.nanoTime();
        do {
            body.run();
            iters++;
        } while (System.nanoTime() < deadline);
        nanos = System.nanoTime() - start;
        double s = nanos / 1e9;
        System.out.printf(
                "  %s %7.3f ms/iter  %7.1f GB/s%n", name, s / iters * 1e3, gbPerIter * iters / s);
    }

    // ------------------------------------------------------------------
    // Static, byte-addressed Q8_0·F32 dot. Bodies ported byte-for-byte from
    // Q8_0FloatTensor (Tensors.java), dropping the element-offset alignment
    // dance: weights are addressed block-aligned by byte offset, always.
    // ------------------------------------------------------------------

    static float dotQ8_0(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        if (FS.vectorBitSize() == 512) {
            return dot512(w, wByte, x, xByte, k);
        }
        return dotGeneric(w, wByte, x, xByte, k);
    }

    private static float dot512(MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        int j = 0;
        int upperBound = k / BLOCK * BLOCK;
        long b0 = wByte;
        FloatVector c0 = FloatVector.zero(FS);
        FloatVector c1 = FloatVector.zero(FS);
        for (; j + BLOCK < upperBound; j += 2 * BLOCK, b0 += 2L * BLOCK_BYTES) {
            var vd0 = FloatVector.broadcast(FS, readF16(w, b0));
            var vd1 = FloatVector.broadcast(FS, readF16(w, b0 + BLOCK_BYTES));
            var w00 = bytesAt(w, b0 + 2).mul(vd0);
            var w01 = bytesAt(w, b0 + 2 + 16).mul(vd0);
            var w10 = bytesAt(w, b0 + BLOCK_BYTES + 2).mul(vd1);
            var w11 = bytesAt(w, b0 + BLOCK_BYTES + 2 + 16).mul(vd1);
            c0 =
                    c0.add(
                            w01.fma(
                                    floatsAt(x, xByte + 4L * (j + 16)),
                                    w00.mul(floatsAt(x, xByte + 4L * j))));
            c1 =
                    c1.add(
                            w11.fma(
                                    floatsAt(x, xByte + 4L * (j + BLOCK + 16)),
                                    w10.mul(floatsAt(x, xByte + 4L * (j + BLOCK)))));
        }
        result += c0.reduceLanes(VectorOperators.ADD) + c1.reduceLanes(VectorOperators.ADD);
        for (; j < upperBound; j += BLOCK, b0 += BLOCK_BYTES) {
            var vd0 = FloatVector.broadcast(FS, readF16(w, b0));
            var w00 = bytesAt(w, b0 + 2).mul(vd0);
            var w01 = bytesAt(w, b0 + 2 + 16).mul(vd0);
            result +=
                    w01.fma(
                                    floatsAt(x, xByte + 4L * (j + 16)),
                                    w00.mul(floatsAt(x, xByte + 4L * j)))
                            .reduceLanes(VectorOperators.ADD);
        }
        if (j < k) {
            result += scalarDot(w, b0, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    private static float dotGeneric(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int k) {
        float result = 0f;
        int upperBound = k / BLOCK * BLOCK;
        FloatVector val = FloatVector.zero(FS);
        long bo = wByte;
        int j = 0;
        for (; j < upperBound; j += BLOCK, bo += BLOCK_BYTES) {
            val = blockFma(w, bo, x, xByte + 4L * j, val);
        }
        result += val.reduceLanes(VectorOperators.ADD);
        if (j < k) {
            result += scalarDot(w, bo, x, xByte + 4L * j, k - j);
        }
        return result;
    }

    private static FloatVector blockFma(
            MemorySegment w, long blockOffset, MemorySegment x, long xByte, FloatVector acc) {
        var wScale = FloatVector.broadcast(FS, readF16(w, blockOffset));
        return switch (FS.vectorBitSize()) {
            case 512 -> {
                var w0 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                w,
                                blockOffset + 2,
                                ByteOrder.LITTLE_ENDIAN);
                var w1 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                w,
                                blockOffset + 2 + 16,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = floatsAt(x, xByte).mul(w0.castShape(FS, 0));
                var s1 = floatsAt(x, xByte + 4L * FS.length()).mul(w1.castShape(FS, 0));
                yield s0.add(s1).fma(wScale, acc);
            }
            case 256 -> {
                var wBytes =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_256,
                                w,
                                blockOffset + 2,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = floatsAt(x, xByte).mul(wBytes.castShape(FS, 0));
                var s1 = floatsAt(x, xByte + 4L * 2 * FS.length()).mul(wBytes.castShape(FS, 2));
                s0 = floatsAt(x, xByte + 4L * FS.length()).fma(wBytes.castShape(FS, 1), s0);
                s1 = floatsAt(x, xByte + 4L * 3 * FS.length()).fma(wBytes.castShape(FS, 3), s1);
                yield s0.add(s1).fma(wScale, acc);
            }
            case 128 -> {
                FloatVector val = acc;
                for (int i = 0; i < 2; ++i) {
                    int off = i * 16;
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    w,
                                    blockOffset + 2 + i * ByteVector.SPECIES_128.vectorByteSize(),
                                    ByteOrder.LITTLE_ENDIAN);
                    var s0 = floatsAt(x, xByte + 4L * off).mul(wBytes.castShape(FS, 0));
                    var s1 =
                            floatsAt(x, xByte + 4L * (off + 2 * FS.length()))
                                    .mul(wBytes.castShape(FS, 2));
                    s0 =
                            floatsAt(x, xByte + 4L * (off + FS.length()))
                                    .fma(wBytes.castShape(FS, 1), s0);
                    s1 =
                            floatsAt(x, xByte + 4L * (off + 3 * FS.length()))
                                    .fma(wBytes.castShape(FS, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                yield val;
            }
            default -> throw new UnsupportedOperationException(FS.toString());
        };
    }

    private static float scalarDot(
            MemorySegment w, long wByte, MemorySegment x, long xByte, int n) {
        // element-wise dequant tail (never runs for block-multiple k, kept for honesty)
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            long idx = (wByte - 0) / BLOCK_BYTES * BLOCK + i; // element index within weights
            long blockOffset = idx / BLOCK * BLOCK_BYTES;
            float scale = readF16(w, blockOffset);
            byte q = w.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + idx % BLOCK);
            sum += q * scale * x.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xByte + 4L * i);
        }
        return sum;
    }

    // 512-path helper: 16 sign-extended bytes widened to a float vector (part-0 cast).
    private static FloatVector bytesAt(MemorySegment w, long off) {
        return (FloatVector)
                ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128, w, off, ByteOrder.LITTLE_ENDIAN)
                        .castShape(FS, 0);
    }

    private static FloatVector floatsAt(MemorySegment x, long byteOff) {
        return FloatVector.fromMemorySegment(FS, x, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    private static float readF16(MemorySegment seg, long off) {
        return Float.float16ToFloat(seg.get(ValueLayout.JAVA_SHORT_UNALIGNED, off));
    }

    private Q8DotBench() {}
}
