package com.llama4j;

import com.qxotic.format.gguf.GGMLType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/**
 * Kernel parity harness: every quantized dot/gemv/gemm and the rmsnorm vector path are checked
 * against a double-precision reference computed through the same {@code getFloat} decode. Weights
 * are synthetic blocks (random payload bytes, scale fields rewritten to finite values), so no
 * model file is needed and a run takes seconds. Exits non-zero on any mismatch.
 *
 * <p>Run via {@code make test}. The vector kernels under test are selected by the same dispatch
 * as inference (USE_VECTOR_API, species width, native lib if {@code -Dllama.nativeGemmLib} or
 * {@code -Dllama.staticGemm} is set), so the harness covers whatever configuration it runs under.
 */
public final class KernelParityTest {

    private static final long SEED = 42;
    private static int checks = 0;
    private static int failures = 0;

    public static void main(String[] args) {
        GGMLType[] quants = {GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_1, GGMLType.Q8_0,
                             GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.MXFP4,
                             GGMLType.F16, GGMLType.BF16, GGMLType.F32};
        for (GGMLType type : quants) {
            testDot(type);
        }
        testGemv(GGMLType.Q8_0);
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_0, GGMLType.Q4_1,
                                            GGMLType.MXFP4, GGMLType.BF16, GGMLType.F16, GGMLType.F32}) {
            testGemm(type);
        }
        testRmsnorm();

        System.out.printf("%d checks, %d failures%n", checks, failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    // ---- synthetic tensors ------------------------------------------------------------------

    /** f16 bits for a value in [0.5, 1): keeps every decoded weight finite and well-scaled. */
    private static void putScaleF16(MemorySegment seg, long offset, Random rng) {
        seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, (short) (0x3800 | (rng.nextInt() & 0x3FF)));
    }

    static FloatTensor makeQuant(GGMLType type, int numElements, Random rng) {
        if (type == GGMLType.F32) {
            float[] values = new float[numElements];
            for (int i = 0; i < numElements; i++) {
                values[i] = (float) rng.nextGaussian();
            }
            return F32FloatTensor.of(values);
        }
        if (type == GGMLType.F16) {
            F16FloatTensor t = F16FloatTensor.allocate(numElements);
            for (int i = 0; i < numElements; i++) {
                t.setFloat(i, (float) rng.nextGaussian());
            }
            return t;
        }
        if (type == GGMLType.BF16) {
            MemorySegment seg = Arena.ofAuto().allocate(numElements * 2L, 64);
            for (int i = 0; i < numElements; i++) {
                short bits = (short) (Float.floatToRawIntBits((float) rng.nextGaussian()) >>> 16);
                seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, i * 2L, bits);
            }
            return new BF16FloatTensor(numElements, seg);
        }

        int blockSize = type.getElementsPerBlock();
        int typeSize = type.getBlockByteSize();
        int blocks = numElements / blockSize;
        byte[] payload = new byte[blocks * typeSize];
        rng.nextBytes(payload);
        MemorySegment seg = Arena.ofAuto().allocate(payload.length, 64);
        MemorySegment.copy(payload, 0, seg, ValueLayout.JAVA_BYTE, 0, payload.length);
        for (int b = 0; b < blocks; b++) {
            long off = (long) b * typeSize;
            switch (type) {
                case Q4_0, Q8_0 -> putScaleF16(seg, off, rng);                                   // d
                case Q4_1, Q5_1, Q4_K, Q5_K -> { putScaleF16(seg, off, rng); putScaleF16(seg, off + 2, rng); } // d, m/dmin
                case Q6_K -> putScaleF16(seg, off + 208, rng);                                   // ql|qh|scales|d
                case MXFP4 -> seg.set(ValueLayout.JAVA_BYTE, off, (byte) (120 + rng.nextInt(10))); // e8m0 ~ 2^-4..2^1
                default -> throw new UnsupportedOperationException(type.toString());
            }
        }
        return switch (type) {
            case Q4_0 -> new Q4_0FloatTensor(numElements, seg);
            case Q4_1 -> new Q4_1FloatTensor(numElements, seg);
            case Q5_1 -> new Q5_1FloatTensor(numElements, seg);
            case Q8_0 -> new Q8_0FloatTensor(numElements, seg);
            case Q4_K -> new Q4_KFloatTensor(numElements, seg);
            case Q5_K -> new Q5_KFloatTensor(numElements, seg);
            case Q6_K -> new Q6_KFloatTensor(numElements, seg);
            case MXFP4 -> new MXFP4FloatTensor(numElements, seg);
            default -> throw new UnsupportedOperationException(type.toString());
        };
    }

    static F32FloatTensor makeF32(int n, Random rng) {
        return (F32FloatTensor) makeQuant(GGMLType.F32, n, rng);
    }

    // ---- reference + comparison -------------------------------------------------------------

    /** Double-precision dot through the scalar decode; also returns sum|a_i*b_i| for tolerance. */
    private static double[] refDot(FloatTensor a, int aOffset, FloatTensor b, int bOffset, int size) {
        double sum = 0, sumAbs = 0;
        for (int i = 0; i < size; i++) {
            double p = (double) a.getFloat(aOffset + i) * b.getFloat(bOffset + i);
            sum += p;
            sumAbs += Math.abs(p);
        }
        return new double[]{sum, sumAbs};
    }

    private static void check(String what, double actual, double expected, double sumAbs) {
        check(what, actual, expected, sumAbs, 1e-4);
    }

    private static void check(String what, double actual, double expected, double sumAbs, double relTol) {
        checks++;
        double tol = relTol * sumAbs + 1e-3;
        if (!(Math.abs(actual - expected) <= tol)) {
            failures++;
            System.err.printf("FAIL %s: got %.8g expected %.8g (tol %.3g)%n", what, actual, expected, tol);
        }
    }

    // ---- tests --------------------------------------------------------------------------------

    static void testDot(GGMLType type) {
        Random rng = new Random(SEED ^ type.ordinal());
        int blockSize = Math.max(type.getElementsPerBlock(), 1);
        int n = 16 * Math.max(blockSize, 64);
        FloatTensor w = makeQuant(type, n, rng);
        F32FloatTensor x = makeF32(n, rng);

        int[][] cases = {
                {0, n},                                  // full, aligned
                {blockSize, 4 * blockSize},              // aligned offset
                {0, 2 * blockSize + blockSize / 2 + 3},  // ragged tail
                {3 * blockSize, blockSize},              // single block
                {blockSize + 5, 2 * blockSize},          // unaligned head
        };
        for (int[] c : cases) {
            int off = c[0], size = Math.min(c[1], n - off);
            double[] ref = refDot(w, off, x, 0, size);
            check(type + ".dot(off=" + off + ",size=" + size + ")",
                    w.dot(off, x, 0, size), ref[0], ref[1]);
            // and through scalarDot (the non-vector fallback) for the same span
            check(type + ".scalarDot(off=" + off + ",size=" + size + ")",
                    FloatTensor.scalarDot(w, off, x, 0, size), ref[0], ref[1]);
        }
    }

    static void testGemv(GGMLType type) {
        Random rng = new Random(SEED ^ 0x9e3779b9L);
        // large enough for the blocked vector gemv path (dim0*dim1 > 1<<18), plus a small one
        for (int[] shape : new int[][]{{512, 1024}, {32, 512}}) {
            int dim0 = shape[0], dim1 = shape[1];
            FloatTensor w = makeQuant(type, dim0 * dim1, rng);
            F32FloatTensor x = makeF32(dim1, rng);
            F32FloatTensor out = F32FloatTensor.allocate(dim0);
            w.gemv(x, 0, out, 0, dim0, dim1, 0);
            for (int row = 0; row < dim0; row += Math.max(1, dim0 / 17)) {
                double[] ref = refDot(w, row * dim1, x, 0, dim1);
                check(type + ".gemv[" + dim0 + "x" + dim1 + "] row " + row, out.getFloat(row), ref[0], ref[1]);
            }
        }
    }

    static void testGemm(GGMLType type) {
        Random rng = new Random(SEED ^ 0x7f4a7c15L);
        // dim0 = 104 = 6x16 + 8: the trailing 8 rows hit the native bands' scalar-leftover path;
        // seq 7 exercises tile remainders below the VNNI threshold, 13 the 16x1 column
        // remainder, 16 the full 16x4 tiles. The VNNI paths' int8 activation quantization
        // needs the looser tolerance (~0.4% relative).
        int dim0 = 104, dim1 = 1024;
        for (int seqLen : new int[]{7, 13, 16}) {
            FloatTensor w = makeQuant(type, dim0 * dim1, rng);
            F32FloatTensor x = makeF32(seqLen * dim1, rng);
            F32FloatTensor out = F32FloatTensor.allocate(seqLen * dim0);
            w.gemm(x, dim1, out, dim0, seqLen, dim0, dim1);
            for (int s = 0; s < seqLen; s++) {
                for (int row = 0; row < dim0; row += 13) {
                    double[] ref = refDot(w, row * dim1, x, s * dim1, dim1);
                    check(type + ".gemm[seq=" + seqLen + "," + s + "," + row + "]",
                            out.getFloat(s * dim0 + row), ref[0], ref[1], 5e-3);
                }
            }
        }
    }

    static void testRmsnorm() {
        Random rng = new Random(SEED ^ 0x85ebca6bL);
        int size = 2048;
        float eps = 1e-5f;
        F32FloatTensor x = makeF32(size, rng);
        F32FloatTensor weight = makeF32(size, rng);
        F32FloatTensor out = F32FloatTensor.allocate(size);
        Llama.rmsnorm(out, 0, x, 0, weight, size, eps);

        double ss = 0;
        for (int i = 0; i < size; i++) {
            ss += (double) x.getFloat(i) * x.getFloat(i);
        }
        double scale = 1.0 / Math.sqrt(ss / size + eps);
        for (int i = 0; i < size; i += 37) {
            double expected = weight.getFloat(i) * scale * x.getFloat(i);
            check("rmsnorm[" + i + "]", out.getFloat(i), expected, Math.abs(expected) * 8);
        }
    }
}
