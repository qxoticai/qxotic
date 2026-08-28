package com.qxotic.jam.vector;

import com.qxotic.jam.JAM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/** Dequant-only microbenchmark: elements/s of each dtype's dequantizeRow (single thread). */
final class DequantBench {
    public static void main(String[] args) {
        int m = 1024, k = 2048;
        Arena arena = Arena.ofShared();
        Random rng = new Random(1);
        record D(String name, int tag, BandGemm.RowDequant deq) {}
        D[] ds = {
            new D("Q8_0", JAM.Q8_0, Q8Kernel::dequantizeRow),
            new D("Q4_0", JAM.Q4_0, Q4Kernel::dequantizeRow),
            new D("Q4_K", JAM.Q4_K, Q4KKernel::dequantizeRow),
            new D("Q5_K", JAM.Q5_K, Q5KKernel::dequantizeRow),
            new D("Q6_K", JAM.Q6_K, Q6KKernel::dequantizeRow),
        };
        MemorySegment dst = arena.allocate((long) k * 4, 64);
        MemorySegment sv = VectorSupport.vectorSegment(dst);
        long sb = VectorSupport.vectorBase(dst);
        for (D d : ds) {
            if (args.length > 0 && !d.name().equals(args[0])) continue;
            QuantWeights.Weight w = QuantWeights.encode(d.tag(), m, k, arena, rng);
            double best = 0;
            for (int rep = 0; rep < 150; rep++) {
                long t0 = System.nanoTime();
                for (int i = 0; i < 20; i++)
                    for (int r = 0; r < m; r++)
                        d.deq().dequantize(w.seg(), (long) r * k, k, sv, sb);
                double dt = (System.nanoTime() - t0) / 1e9;
                double eps = 20.0 * m * k / dt / 1e9;
                if (eps > best) best = eps;
            }
            System.out.printf("%-5s %.2f Gelem/s%n", d.name(), best);
        }
    }
}
