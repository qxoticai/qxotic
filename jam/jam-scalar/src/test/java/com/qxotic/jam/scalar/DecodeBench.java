package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/**
 * Throughput of the row decoders: {@code DecodeBench [dtype...]} prints Gelem/s per dtype for a
 * 1024-row x 2048 weight decoded 256 elements at a time (the gemv chunk) and 2048 at a time (a
 * prefill k-block). Not a test - a tuning tool; run under both JITs, one dtype per JVM (a
 * megamorphic run hides each decoder's own speed).
 */
public final class DecodeBench {

    static float sink;
    static final int REPS = Integer.getInteger("reps", 12);

    public static void main(String[] args) {
        String[] names =
                args.length > 0
                        ? args
                        : new String[] {
                            "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "MXFP4",
                            "NVFP4", "Q1_0"
                        };
        for (String name : names) {
            GGMLType t = GGMLType.valueOf(name);
            int m = 1024, k = 2048;
            MemorySegment w = Arena.ofAuto().allocate(t.rowBytes((long) m * k), 64);
            Random rng = new Random(1);
            for (long i = 0; i < w.byteSize(); i++) w.set(JAVA_BYTE, i, (byte) rng.nextInt(256));
            Decode row = new Decode();
            float[] out = new float[k];
            for (int span : new int[] {256, 2048}) {
                double best = 0;
                for (int rep = 0; rep < REPS; rep++) {
                    long t0 = System.nanoTime();
                    for (int i = 0; i < m; i++) {
                        for (int l = 0; l < k; l += span)
                            row.row(new Weight(w, 0, t, k), i, l, span, span, out, 0);
                        sink += out[7];
                    }
                    double g = (double) m * k / (System.nanoTime() - t0);
                    if (rep >= REPS / 3) best = Math.max(best, g);
                }
                System.out.printf("%-6s span=%-5d %6.1f Gelem/s%n", name, span, best);
            }
        }
        if (sink == 42f) System.out.println(JAM.OK);
    }

    private DecodeBench() {}
}
