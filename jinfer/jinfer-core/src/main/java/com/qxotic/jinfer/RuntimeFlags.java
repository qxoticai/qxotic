package com.qxotic.jinfer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runtime flags read by the decode work-runners and the model boundary. Prompt and cache defaults
 * live with their owning modules.
 */
public final class RuntimeFlags {

    /** Default scratch width for {@code newState}: prefill batches up to this many tokens. */
    public static final int BATCH_CAPACITY = Integer.getInteger("jinfer.batchCapacity", 512);

    // decode runs at physical-core width on a spin-barrier pool (Parallel.onDecodePool /
    // SpinPool): decode is memory-bandwidth bound, so one thread per PHYSICAL core saturates DRAM
    // while a 2nd SMT sibling only contends for the core's load/store ports.
    // -Djinfer.decodeSpin=false forces the plain ForkJoin path.
    public static final int DECODE_THREADS =
            Integer.getInteger("jinfer.decodeThreads", physicalCoreCount());
    public static final boolean DECODE_SPIN =
            !"false".equals(System.getProperty("jinfer.decodeSpin"));

    // Keys per flashDecode partition: below this there is nothing to gain from splitting the
    // attended range, so it falls through to rollingDecode.
    public static final int DECODE_BLOCK_SIZE = Integer.getInteger("jinfer.decodeBlockSize", 512);

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
