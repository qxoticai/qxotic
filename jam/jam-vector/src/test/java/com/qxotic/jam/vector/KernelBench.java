package com.qxotic.jam.vector;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.qxotic.jam.JAM;
import com.qxotic.jam.TestPool;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Kernel microbenchmark (not a JUnit test): times each dtype's gemm at prefill-like shapes and
 * reports GFLOP/s (2*m*n*k / t) plus effective weight-stream GB/s. Run:
 *
 * <pre>
 * java --add-modules jdk.incubator.vector \
 *   -cp jam-vector/target/classes:jam-vector/target/test-classes:&lt;deps&gt; \
 *   com.qxotic.jam.vector.KernelBench [dtypeFilter] [m] [k] [n,...] [warmup] [reps]
 * </pre>
 *
 * Runs on the calling thread (the provider's inline Parallel), tile via -Djam.vector.tile=...
 */
final class KernelBench {

    /** {@code -Djam.bench.threads=N} (default: every logical CPU). */
    private static final JAM.Parallel POOL =
            TestPool.of(
                    Integer.getInteger(
                            "jam.bench.threads", Runtime.getRuntime().availableProcessors()));

    @FunctionalInterface
    interface Gemm {
        void run(
                MemorySegment w,
                MemorySegment a,
                long aBase,
                MemorySegment o,
                long oBase,
                int aStride,
                int oStride,
                int n,
                int m,
                int k,
                long wOff);
    }

    private static Gemm withScratch(VectorKernelTestBand g) {
        Scratch s = new Scratch(POOL);
        return (w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff) ->
                g.run(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, s);
    }

    @FunctionalInterface
    interface VectorKernelTestBand {
        void run(
                MemorySegment w,
                MemorySegment a,
                long aBase,
                MemorySegment o,
                long oBase,
                int aStride,
                int oStride,
                int n,
                int m,
                int k,
                long wOff,
                Scratch scratch);
    }

    public static void main(String[] args) throws Exception {
        String dumpSecs = System.getProperty("jam.selfdump");
        if (dumpSecs != null) {
            Thread t =
                    new Thread(
                            () -> {
                                try {
                                    Thread.sleep(Long.parseLong(dumpSecs) * 1000);
                                    dumpCodeHeaps();
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            });
            t.setDaemon(true);
            t.start();
        }
        String filter = args.length > 0 ? args[0].toUpperCase() : "ALL";
        int m = args.length > 1 ? Integer.parseInt(args[1]) : 4096;
        int k = args.length > 2 ? Integer.parseInt(args[2]) : 4096;
        String[] ns = (args.length > 3 ? args[3] : "128,512").split(",");
        int warmup = args.length > 4 ? Integer.parseInt(args[4]) : 4;
        int reps = args.length > 5 ? Integer.parseInt(args[5]) : 8;

        System.out.printf(
                "vector=%dbit tile=%s(%d) band=%s wide=%s graalJit=%s threads=%d jit=%s%n",
                VectorSupport.F_SPECIES.vectorBitSize(),
                VectorSupport.TILE,
                VectorSupport.TILE_CODE,
                BandGemm.BAND,
                VectorSupport.WIDE_TILE,
                VectorSupport.GRAAL_JIT,
                POOL.width(),
                System.getProperty("java.vm.name"));

        record Case(String name, int tag, Gemm g) {}
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("Q8_0", JAM.Q8_0, withScratch(Q8Kernel::gemm)));
        cases.add(new Case("Q4_0", JAM.Q4_0, withScratch(Q4Kernel::gemm)));
        cases.add(new Case("Q4_K", JAM.Q4_K, withScratch(Q4KKernel::gemm)));
        cases.add(new Case("Q5_K", JAM.Q5_K, withScratch(Q5KKernel::gemm)));
        cases.add(new Case("Q6_K", JAM.Q6_K, withScratch(Q6KKernel::gemm)));
        cases.add(new Case("MXFP4", JAM.MXFP4, withScratch(Mxfp4Kernel::gemm)));
        cases.add(new Case("NVFP4", JAM.NVFP4, withScratch(Nvfp4Kernel::gemm)));
        cases.add(new Case("Q1_0", JAM.Q1_0, withScratch(Q1Kernel::gemm)));

        Arena arena = Arena.ofShared();
        Random rng = new Random(7);
        System.out.printf(
                "%-6s %6s %6s %4s %10s %10s %10s%n",
                "dtype", "m", "k", "n", "best", "median", "wGB/s");
        for (Case c : cases) {
            if (!filter.equals("ALL") && !c.name.contains(filter)) continue;
            QuantWeights.Weight w = QuantWeights.encode(c.tag, m, k, arena, rng);
            long wBytes = w.seg().byteSize();
            for (String ns2 : ns) {
                int n = Integer.parseInt(ns2.trim());
                MemorySegment a = arena.allocate((long) n * k * 4, 64);
                float[] av = QuantWeights.gaussians(n * k, rng);
                for (int i = 0; i < av.length; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
                MemorySegment o = arena.allocate((long) n * m * 4, 64);
                MemorySegment g = VectorSupport.GLOBAL;
                long ab = a.address(), ob = o.address();

                double flops = 2.0 * m * n * k;
                for (int i = 0; i < warmup; i++) c.g.run(w.seg(), g, ab, g, ob, k, m, n, m, k, 0L);
                double[] ts = new double[reps];
                for (int i = 0; i < reps; i++) {
                    long t0 = System.nanoTime();
                    c.g.run(w.seg(), g, ab, g, ob, k, m, n, m, k, 0L);
                    ts[i] = (System.nanoTime() - t0) / 1e9;
                }
                if (Boolean.getBoolean("jam.bench.reps")) {
                    StringBuilder sb = new StringBuilder("reps GF/s:");
                    for (double t : ts) sb.append(String.format(" %.0f", flops / t / 1e9));
                    System.out.println(sb);
                }
                java.util.Arrays.sort(ts);
                double best = ts[0], med = ts[reps / 2];
                System.out.printf(
                        "%-6s %6d %6d %4d %10.1f %10.1f %10.1f%n",
                        c.name,
                        m,
                        k,
                        n,
                        flops / best / 1e9,
                        flops / med / 1e9,
                        wBytes / best / 1e9);
                sink += o.get(JAVA_FLOAT_UNALIGNED, 0);
            }
        }
        arena.close();
        if (sink == 42) System.out.println(sink);
    }

    private static volatile float sink;

    /**
     * Dump this process's rwx (JIT code heap) regions to /tmp/selfdump_<pid>_<i>.bin, with a .map
     * file recording each region's base address, so the JIT code can be disassembled offline.
     * Trigger with -Djam.selfdump=<seconds>.
     */
    static void dumpCodeHeaps() throws Exception {
        long pid = ProcessHandle.current().pid();
        List<long[]> regions = new ArrayList<>();
        for (String line :
                java.nio.file.Files.readAllLines(java.nio.file.Path.of("/proc/self/maps"))) {
            if (!line.contains(" rwxp ")) continue;
            String[] loHi = line.split(" ")[0].split("-");
            regions.add(
                    new long[] {
                        Long.parseUnsignedLong(loHi[0], 16), Long.parseUnsignedLong(loHi[1], 16)
                    });
        }
        try (var out = new java.io.PrintStream("/tmp/selfdump_" + pid + ".map")) {
            for (int i = 0; i < regions.size(); i++) {
                long[] r = regions.get(i);
                out.printf("%d %x %x%n", i, r[0], r[1]);
                MemorySegment heap =
                        MemorySegment.ofAddress(r[0])
                                .reinterpret(r[1] - r[0], Arena.ofAuto(), null);
                java.nio.file.Files.write(
                        java.nio.file.Path.of("/tmp/selfdump_" + pid + "_" + i + ".bin"),
                        heap.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            }
        }
        System.err.println("selfdump: " + regions.size() + " regions");
    }
}
