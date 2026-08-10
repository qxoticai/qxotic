package com.qxotic.jinfer.x;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The kernel cone's slice of jinfer's runtime flags (ported from jinfer-core {@code RuntimeFlags}):
 * only what the decode work-runners read. The rest (prompt lengths, cache, batch defaults) lands
 * with the boundary cone that owns it.
 */
final class RuntimeFlags {

    // decode runs at physical-core width on a spin-barrier pool (Parallel.onDecodePool /
    // SpinPool): decode is memory-bandwidth bound, so one thread per PHYSICAL core saturates DRAM
    // while a 2nd SMT sibling only contends for the core's load/store ports.
    // -Djinfer.decodeSpin=false forces the plain ForkJoin path.
    public static final int DECODE_THREADS =
            Integer.getInteger("jinfer.decodeThreads", physicalCoreCount());
    static final boolean DECODE_SPIN = !"false".equals(System.getProperty("jinfer.decodeSpin"));

    // Keys per flashDecode partition: below this there is nothing to gain from splitting the
    // attended range, so it falls through to rollingDecode.
    static final int DECODE_BLOCK_SIZE = Integer.getInteger("jinfer.decodeBlockSize", 512);

    /**
     * Best-effort physical-core count for sizing the bandwidth-bound decode pool. Linux reports SMT
     * state via sysfs (SMT on => 2 hardware threads per core => logical/2; off => logical).
     * macOS/Windows have no such file, so we assume 2-way SMT on x86 and none on ARM. Override with
     * -Djinfer.decodeThreads; read at run time so a native binary detects its host.
     */
    private static int physicalCoreCount() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            boolean smtOn =
                    !"0"
                            .equals(
                                    Files.readString(Path.of("/sys/devices/system/cpu/smt/active"))
                                            .trim());
            return smtOn ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            boolean noSmt = arch.contains("aarch64") || arch.contains("arm");
            return noSmt ? logical : Math.max(1, logical / 2);
        }
    }

    private RuntimeFlags() {}
}
