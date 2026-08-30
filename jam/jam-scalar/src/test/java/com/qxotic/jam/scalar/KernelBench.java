package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.qxotic.jam.JAM;
import com.qxotic.jam.TestPool;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Throughput of {@link ScalarJAM#mm} on one shape: {@code KernelBench <dtype> <m> <n> <k>
 * [seconds]}. Warms up for that long, then reports the median and best rep of another such window;
 * {@code -Djam.bench.threads=1} isolates a core. Not a test - a tool for tuning the sweeps (run it
 * under both -XX:-UseJVMCICompiler and Graal).
 */
public final class KernelBench {

    public static void main(String[] args) {
        int tag = com.qxotic.jam.internal.GGMLType.valueOf(args[0]).code();
        int m = Integer.parseInt(args[1]),
                n = Integer.parseInt(args[2]),
                k = Integer.parseInt(args[3]);
        int seconds = args.length > 4 ? Integer.parseInt(args[4]) : 3;
        Random rng = new Random(1);
        Arena arena = Arena.ofAuto();
        MemorySegment w =
                arena.allocate(
                        com.qxotic.jam.internal.GGMLType.byCode(tag).rowBytes((long) m * k), 64);
        for (long i = 0; i < w.byteSize(); i++)
            w.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) rng.nextInt(256));
        MemorySegment a = arena.allocate((long) n * k * 4, 64),
                r = arena.allocate((long) n * m * 4, 64);
        for (long i = 0; i < (long) n * k; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4, rng.nextFloat());
        JAM jam =
                new ScalarJAM(
                        TestPool.of(
                                Integer.getInteger(
                                        "jam.bench.threads",
                                        Runtime.getRuntime().availableProcessors())));
        double macs = (double) m * n * k;
        long warm = System.nanoTime();
        while (System.nanoTime() - warm < seconds * 1_000_000_000L) jam.mm(w, a, r, tag, m, n, k);
        List<Double> reps = new ArrayList<>();
        long from = System.nanoTime();
        while (System.nanoTime() - from < seconds * 1_000_000_000L) {
            long t0 = System.nanoTime();
            jam.mm(w, a, r, tag, m, n, k);
            reps.add(macs / (System.nanoTime() - t0));
        }
        Collections.sort(reps);
        System.out.printf(
                "%s m=%d n=%d k=%d threads=%d  median %.1f  best %.1f GMAC/s (%d reps)%n",
                args[0],
                m,
                n,
                k,
                jam instanceof ScalarJAM
                        ? Integer.getInteger(
                                "jam.bench.threads", Runtime.getRuntime().availableProcessors())
                        : 1,
                reps.get(reps.size() / 2),
                reps.get(reps.size() - 1),
                reps.size());
    }

    private KernelBench() {}
}
